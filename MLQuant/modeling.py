import pandas as pd
import MLQuant as MLQ
import numpy as np
import json, pickle, threading, glob, os

"""
实现模型滚动训练
数据格式要求：
特征features需要包含如下列 
date::int64, curTime::int64, symbol::string, legalData::bool
factor0, factor1, ..
预测标签Nbr需要包含如下列
date::int64, symbol::string, Nbr
"""


# ==============================
# 异步数据加载器（如果不在内存中直接输入Data，则从硬盘中异步读取数据）
# ==============================
class AsyncDataLoader:
    def __init__(self, data_dir, all_trade_dates):
        self.data_dir = data_dir
        self.all_dates = sorted(pd.to_datetime(all_trade_dates).date)
        self.cache_df = None
        self.available_up_to = min(self.all_dates) - pd.Timedelta(days=1)
        self.lock = threading.Lock()
        self.loading = False
    def _load_range(self, start_date, end_date):
        """根据日期范围，读取对应目录下的所有 .pqt 文件"""
        dfs = []
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        for dt in date_range:
            date_str = dt.strftime('%Y%m%d')
            day_dir = os.path.join(self.data_dir, date_str)
            if not os.path.exists(day_dir):
                continue
            # 读取该日所有 .pqt 文件（假设每个文件是一个时间片）
            pqt_files = glob.glob(os.path.join(day_dir, "*.pqt"))
            for f in pqt_files:
                try:
                    df_part = pd.read_parquet(f)
                    # 确保有 'date' 列（若没有，可用文件名推断）
                    if 'date' not in df_part.columns:
                        df_part['date'] = dt  # 或从文件名解析更细粒度时间
                    dfs.append(df_part)
                except Exception as e:
                    MLQ.io.log(f"Failed to read {f}: {e}", level='warn')
        if not dfs:
            df_combined = pd.DataFrame()
        else:
            df_combined = pd.concat(dfs, ignore_index=True)
            df_combined['date'] = pd.to_datetime(df_combined['date'])
        with self.lock:
            if self.cache_df is None:
                self.cache_df = df_combined
            else:
                self.cache_df = pd.concat([self.cache_df, df_combined], ignore_index=True).drop_duplicates().sort_values('date')
            self.available_up_to = max(self.available_up_to, pd.to_datetime(end_date))
    def ensure_loaded(self, required_end_date):
        with self.lock:
            if self.available_up_to >= pd.to_datetime(required_end_date):
                return
        if not self.loading:
            self.loading = True
            start_load = self.available_up_to + pd.Timedelta(days=1)
            thread = threading.Thread(
                target=self._load_range,
                args=(start_load, required_end_date),
                daemon=True
            )
            thread.start()
            thread.join()
            self.loading = False
    def get_data_slice(self, start_date, end_date):
        if self.cache_df is None or self.cache_df.empty:
            return pd.DataFrame()
        mask = (self.cache_df['date'] >= pd.to_datetime(start_date)) & \
               (self.cache_df['date'] <= pd.to_datetime(end_date))
        return self.cache_df[mask].copy()


# ==============================
# Modeling 主类
# ==============================
class Modeling():
    # Param 需包含"trainParam", "featureParam", "modelParam"三类参数，以及log的存储路径"logLoc"
    # Data pd.DataFrame格式, 
    #   如果未直接输入则会从Param["trainParam"]["featureDir"]路径下异步读取.
    # model 自定义的模型类，传入modelParam初始化，需实现train(), predict()功能。
    #   如果未直接输入则会直接import Param["modelParam"]["selectModel"]中字符串路径的类，
    #   例如参数为"mylib.myModels.model0"，等价于import mylib.myModels.model0
    # featureFilter 自定义特征筛选类，传入featureParam初始化，需实现filtFeature() 返回输入模型特征名的list。
    #   如果未直接输入则会直接import Param["modelParam"]["selectModel"]中字符串路径的类，
    #   例如"mylib.myFilters.filter0".
    def __init__(self, Param={"trainParam":{
                            "strResultDate":"20250101",
                            "endResultDate":"20260101",
                            "trainSetLen":600,
                            "testSetLen":20,
                            "stepTrainAndTest":3,
                            "featureDir":"./features/",
                            "predictDir":"./Nbr",
                            "predictLabel":"Nbr241",
                            "outPath":"./",
                            },
                            "featureParam":{"selectFilter":"MLQuant.sample.floatFilter"}, 
                            "modelParam":{"selectModel":"MLQuant.sample.lgbModel"}, 
                            "logLoc":"./"}, \
                Data=None, model=None, featureFilter=None):
        self.Param = Param

        if Data is None:
            raise NotImplementedError("当前仅支持传入Data参数，后续版本将支持异步从硬盘加载数据")
        else:
            self.Data = Data
        
        if featureFilter is not None:
            self.featureFilter = featureFilter
        else:
            self.featureFilter = MLQ.io.importMyClass(Param['featureParam']['selectFilter'])
        
        if model is not None:
            self.model = model
        else:
            self.model = MLQ.io.importMyClass(Param['modelParam']['selectModel'])
        
        self.logLoc = Param.get('logLoc', "./")
    # 日志函数
    def log(self, text):
        MLQ.io.log(text, logLoc=self.logLoc)
    # 生成滚动窗口
    def getRollingWindow(self, strResultDate, endResultDate, 
                         trainSetLen, testSetLen, stepTrainAndTest):
        tradeDates = sorted(os.listdir(self.Param['trainParam']['featureDir']))
        # 找出在 [startResultDate, endResultDate] 内的测试起点
        valid_test_starts = [d for d in tradeDates if strResultDate <= d <= endResultDate]
        if not valid_test_starts:
            self.log("!!! Error !!! 特征数据日期范围和目标result范围不匹配")
            raise ValueError
        # 每 testSetLen 天一个测试窗口起点
        testStarts = valid_test_starts[::testSetLen]
        testEnds = []
        for s in testStarts:
            idx = tradeDates.index(s)
            end_idx = min(idx + testSetLen - 1, len(tradeDates) - 1)
            testEnds.append(tradeDates[end_idx])
        def offset(date, n):
            preDates = [d for d in tradeDates if d < date]
            return tradeDates[0] if n > len(preDates) else preDates[-n]
        trainStarts = [offset(d, stepTrainAndTest + trainSetLen) for d in testStarts]
        trainEnds = [offset(d, stepTrainAndTest + 1) for d in testStarts]
        rollingWindows = list(zip(trainStarts, trainEnds, testStarts, testEnds))
        self.log(f"全部滚动窗口如下，共 {len(trainStarts)} 组" + 
                   "\n".join(["  ".join(i) for i in rollingWindows]))
        return rollingWindows
    # 处理单个模型
    def processOneModel(self, train_start, train_end, test_start, test_end):
        self.log(f"处理 train: {train_start} ~ {train_end}, test: {test_start} ~ {test_end}")
        modelLoc = os.path.join(self.Param['trainParam']['outPath'], "model", 
                                f"model_{test_start}_{test_end}")
        # 获取训练/测试数据
        trainData = self.Data[(self.Data['date']>=int(train_start)&
                               (self.Data['date']<=int(train_end)))]
        testData = self.Data[(self.Data['date']>=int(test_start)&
                              (self.Data['date']<=int(test_end)))]
        # 特征筛选
        filt = self.featureFilter(self.Param['featureParam'])
        feature_names = filt.filtFeature(trainData)
        # 保存筛选器

        # 训练模型

        # 测试模型 

        # 保存模型与评估结果
        self.saveModel(model)
        self.evaluateModel(model)
    # modeling 主流程
    def run(self): 
        MLQ.io.log("1. 构建滚动窗口") 
        self.rollingWindow = self.getRollingWindow(
            self.Param["trainParam"]["strResultDate"],
            self.Param["trainParam"]["endResultDate"],
            self.Param["trainParam"]['trainSetLen'],
            self.Param["trainParam"]['testSetLen'],
            self.Param["trainParam"]['stepTrainAndTest'])

        for train_start, train_end, test_start, test_end in self.rollingWindow:
            MLQ.io.log(f"train: {train_start} ~ {train_end}, test: {test_start} ~ {test_end}", logLoc=self.logLoc)

            # === 确保训练数据就绪（异步模式会触发加载，内存模式直接切片）===
            if self.use_async:
                self.async_loader.ensure_loaded(train_end)

            train_data = self._get_data_slice(train_start, train_end)
            if train_data.empty:
                MLQ.io.log(f"Empty training data in [{train_start}, {train_end}]", level='warn')
                continue

            # === 特征筛选与训练 ===
            filt = self.featureFilter(self.Param['featureParam'], train_start, train_end)
            featureNames = filt.filtFeature()
            if not featureNames:
                MLQ.io.log("No features selected.", level='warn')
                continue
            self.saveFilter(filt)

            Xi = train_data[featureNames]
            Yi = train_data[self.Param['predict_label']]
            self.Param['modelParam']['_trainstart'] = train_start
            self.Param['modelParam']['_trainend'] = train_end
            model = self.model(featureNames, self.Param['modelParam'])
            model.train(Xi, Yi)

            # === 测试阶段 ===
            if self.use_async:
                self.async_loader.ensure_loaded(test_end)
            test_data = self._get_data_slice(test_start, test_end)
            self.testModel(model, test_data)

        self.getReport()

    #def _get_all_trade_dates_from_disk(self, data_dir):
    #    """扫描 data_dir 下的所有日期目录，返回日期列表"""
    #    date_dirs = glob.glob(os.path.join(data_dir, "????????"))  # 匹配 YYYYMMDD
    #    dates = []
    #    for d in date_dirs:
    #        basename = os.path.basename(d)
    #        try:
    #            dt = datetime.strptime(basename, "%Y%m%d").date()
    #            dates.append(dt)
    #        except ValueError:
    #            continue
    #    return sorted(set(dates))
    #def loadData(self):
    #    if self.use_async:
    #        data_dir = self.Param.get('dataDir', 'data/')  # 建议在 Param 中配置
    #        all_dates = self._get_all_trade_dates_from_disk(data_dir)
    #        if not all_dates:
    #            raise ValueError(f"No date directories found in {data_dir}")
    #        self.async_loader = AsyncDataLoader(data_dir, all_dates)
    #        MLQ.io.log(f"Async loader initialized with {len(all_dates)} trade days.", logLoc=self.logLoc)
    #    else:
    #        # 检查 Data 是否为 DataFrame
    #        if not isinstance(self.Data, pd.DataFrame):
    #            raise TypeError("If Data is provided, it must be a pandas DataFrame.")
    #        if 'date' not in self.Data.columns:
    #            raise ValueError("Input DataFrame must contain a 'date' column.")
    #        self.Data['date'] = pd.to_datetime(self.Data['date'])
    #        MLQ.io.log("Using provided DataFrame for modeling.", logLoc=self.logLoc)
    #def _get_data_slice(self, start_date, end_date):
    #    """统一接口：无论异步还是内存 DataFrame，都返回切片"""
    #    if self.use_async:
    #        return self.async_loader.get_data_slice(start_date, end_date)
    #    else:
    #        mask = (self.Data['date'] >= pd.to_datetime(start_date)) & \
    #               (self.Data['date'] <= pd.to_datetime(end_date))
    #        return self.Data[mask].copy()



    # --- 以下方法保持不变（已修复 self）---
    def saveFilter(self): pass
    def saveModel(self): pass
    def evaluateModel(self): pass
    def testModel(self, model, test_data): pass
    def getReport(self): pass


# ==============================
# 因子筛选基类
# ==============================
class Filter:
    def __init__(self, featureParam={}):
        self.featureParam = featureParam #因子筛选所需的其他参数
        self.store = {}
    def saveFilter(self, modelLoc): # 储存因子筛选器
        os.makedirs(modelLoc, exist_ok=True)
        with open(os.path.join(modelLoc,"filter_store.json"), 'w') as f:
            json.dump(MLQ.io.converjson(self.store), f, indent=4)
        with open(os.path.join(modelLoc, 'filter.pkl'), 'wb') as f:
            pickle.dump(self, f)
    def restoreModel(self,modelDir): #加载模型
        with open(os.path.join(modelDir, 'filter.pkl'), 'rb') as f:
            model = pickle.load(f)
            self.__dict__.update(model.__dict__)
    #def filtFeature(self, data):  # 需要自定义因子筛选过程，输入训练数据DataFrame返回筛选后的特征名列表


# ==============================
# 模型基类
# ==============================
class Model:
    def __init__(self, modelParam={}):
        self.modelParam = modelParam
        self.store = {}
    def test(self, Xi): # 测试集测试
        return self.predict(Xi)
    def evaluate(self, Xi, Yi): # 评价模型
        from sklearn.metrics import mean_squared_error, r2_score
        y_pred = self.predict(Xi)
        mse = mean_squared_error(Yi, y_pred)
        r2 = r2_score(Yi, y_pred)
        self.store['evaluate'] = {'rmse':np.sqrt(mse),\
                    'R2':r2, 'ic':(lambda x: -np.sqrt(-x) if x<0 else np.sqrt(x))(r2),\
                    'Yi':Yi.tolist(), 'y_pred':y_pred.tolist(),\
                        'featureName':self.featureName}
    def saveModel(self, modelDir): # 储存模型
        os.makedirs(modelDir, exist_ok=True)
        with open(os.path.join(modelDir,"model_store.json"), 'w') as f:
            json.dump(MLQ.io.converjson(self.store), f, indent=4)
        with open(os.path.join(modelDir, 'model.pkl'), 'wb') as f:
            pickle.dump(self, f)
    def restoreModel(self,modelDir): #加载模型
        with open(os.path.join(modelDir, 'model.pkl'), 'rb') as f:
            model = pickle.load(f)
            self.__dict__.update(model.__dict__)
    #def train(self, Xi, Yi):  # 需要自定义训练过程
    #def predict(self, Xi): # 需要自定义预测过程 self.model即可

