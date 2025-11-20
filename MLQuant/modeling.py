import pandas as pd
import MLQuant as MLQ
import threading, glob, os

# 数据格式为 symbol::string, date::in64, curTime::int64, legalData::bool, factor0, factor1, ...


# ==============================
# 异步数据加载器（仅在 Data=None 时使用）
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
# Modeling 主类（支持 Data=None 或 Data=DataFrame）
# ==============================
class Modeling():
    # Param 字典格式,需包含"trainParam", "featureParam", "modelParam", "logLoc"
    # Data pd.DataFrame格式, 
    #   如果未直接输入则会从Param["trainParam"]["featureDir"]路径下异步读取.
    # model 自定义的模型类，需实现train, predict功能。
    #   如果未直接输入则会直接import Param["modelParam"]["selectModel"]中字符串路径的类，如"mylib.myModels.model0".
    # featureFilter 自定义模型类，需实现filtFeature() 返回输入模型特征名的list。
    #   如果未直接输入则会直接import Param["modelParam"]["selectModel"]中字符串路径的类，如"mylib.myFilters.filter0".
    def __init__(self, Param={"trainParam":{
                            "strResultDate":"20250101",
                            "endResultDate":"20260101",
                            "trainSetLen":600,
                            "testSetLen":20,
                            "testAndFitStep":3,
                            "predict_label":"Nbr241",
                            "outPath":"./",
                            },
                            "featureParam":{}, "modelParam":{}, "logLoc":"./"}, \
                Data=None, model=None, featureFilter=None):
        self.Param = Param

        self.Data = Data  # 可能是 DataFrame，也可能是 None
        self.async_loader = None
        self.use_async = Data is None  # 关键标志：是否启用异步加载
        
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

    def _get_all_trade_dates_from_disk(self, data_dir):
        """扫描 data_dir 下的所有日期目录，返回日期列表"""
        date_dirs = glob.glob(os.path.join(data_dir, "????????"))  # 匹配 YYYYMMDD
        dates = []
        for d in date_dirs:
            basename = os.path.basename(d)
            try:
                dt = datetime.strptime(basename, "%Y%m%d").date()
                dates.append(dt)
            except ValueError:
                continue
        return sorted(set(dates))

    def loadData(self):
        if self.use_async:
            data_dir = self.Param.get('dataDir', 'data/')  # 建议在 Param 中配置
            all_dates = self._get_all_trade_dates_from_disk(data_dir)
            if not all_dates:
                raise ValueError(f"No date directories found in {data_dir}")
            self.async_loader = AsyncDataLoader(data_dir, all_dates)
            MLQ.io.log(f"Async loader initialized with {len(all_dates)} trade days.", logLoc=self.logLoc)
        else:
            # 检查 Data 是否为 DataFrame
            if not isinstance(self.Data, pd.DataFrame):
                raise TypeError("If Data is provided, it must be a pandas DataFrame.")
            if 'date' not in self.Data.columns:
                raise ValueError("Input DataFrame must contain a 'date' column.")
            self.Data['date'] = pd.to_datetime(self.Data['date'])
            MLQ.io.log("Using provided DataFrame for modeling.", logLoc=self.logLoc)

    def _get_data_slice(self, start_date, end_date):
        """统一接口：无论异步还是内存 DataFrame，都返回切片"""
        if self.use_async:
            return self.async_loader.get_data_slice(start_date, end_date)
        else:
            mask = (self.Data['date'] >= pd.to_datetime(start_date)) & \
                   (self.Data['date'] <= pd.to_datetime(end_date))
            return self.Data[mask].copy()

    def run(self): 
        MLQ.io.log("加载数据或初始化异步加载器") 
        self.loadData()
        MLQ.io.log("开始滑动窗口建模") 
        
        # 获取用于生成窗口的日期列表
        if self.use_async:
            trade_dates = [pd.Timestamp(d) for d in self.async_loader.all_dates]
        else:
            trade_dates = sorted(self.Data['date'].dt.date.unique())
            trade_dates = [pd.Timestamp(d) for d in trade_dates]

        self.rollingWindow = self.getRollingWindow(
            pd.to_datetime(self.Param['trainParam']['startResultDate']),
            pd.to_datetime(self.Param['trainParam']['endResultDate']),
            self.Param['trainLen'],
            self.Param['testLen'],
            self.Param['step'],
            tradeDates=trade_dates
        )

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
            self.saveModel(model)
            self.evaluateModel(model)

            # === 测试阶段 ===
            if self.use_async:
                self.async_loader.ensure_loaded(test_end)
            test_data = self._get_data_slice(test_start, test_end)
            self.testModel(model, test_data)

        self.getReport()

    # --- 以下方法保持不变（已修复 self）---
    def getRollingWindow(self, startResultDate, endResultDate, 
                         trainLen=750, testLen=50, step=5, tradeDates=None):
        if tradeDates is None:
            raise ValueError("tradeDates must be provided.")
        tradeDates = sorted(tradeDates)
        # 找出在 [startResultDate, endResultDate] 内的测试起点
        valid_test_starts = [d for d in tradeDates if startResultDate <= d <= endResultDate]
        if not valid_test_starts:
            return []
        # 每 testLen 天一个测试窗口起点
        testStarts = valid_test_starts[::testLen]
        testEnds = []
        for s in testStarts:
            idx = tradeDates.index(s)
            end_idx = min(idx + testLen - 1, len(tradeDates) - 1)
            testEnds.append(tradeDates[end_idx])
        
        def offset(date, n):
            preDates = [d for d in tradeDates if d < date]
            return tradeDates[0] if n > len(preDates) else preDates[-n]
        
        trainStarts = [offset(d, step + trainLen) for d in testStarts]
        trainEnds = [offset(d, step + 1) for d in testStarts]
        return list(zip(trainStarts, trainEnds, testStarts, testEnds))

    def saveFilter(self): pass
    def saveModel(self): pass
    def evaluateModel(self): pass
    def testModel(self, model, test_data): pass
    def getReport(self): pass





## 滚动窗口训练, 样本内外测试
#class Modeling():
#    # 配置文件Param，数据、模型、特征筛选器、log地址等均可从配置文件中调用，
#    # 也可以在
#    def __init__(self, Param, Data=None, \
#                 model=None, featureFilter=None, logLoc=None):
#        self.Param = Param  # {'trainParam':***, 'featureParam':***, 'modelParam':***, 'logLoc':***}
#        self.Data = Data
#        if type(featureFilter)!=type(None):
#            self.featureFilter = featureFilter # class featureFilter 
#        else:
#            self.featureFilter = MLQ.io.importMyClass(Param['featureParam']['selectFilter'])
#        if type(model)!=type(None):
#            self.model = model # class Model
#        else:
#            self.model = MLQ.io.importMyClass(Param['modelParam']['selectModel'])
#        if 'logLoc' not in Param.keys():
#            self.logLoc = logLoc
#    def run(self): 
#        MLQ.io.log("加载数据") 
#        self.loadData()
#        MLQ.io.log("开始滑动窗口建模") 
#        # 1. 生成滑动训练/测试窗口;
#        self.rollingWindow = self.getRollingWindow(\
#            self.Param['trainParam']['startResultDate'], self.Param['trainParam']['endResultDate'], \
#                self.Param['trainLen'], self.Param['testLen'], self.Param['step'])
#        # 2. 滚动窗口样本内外训练测试;
#        for trainstart, trainend, testsart, testend in self.rollingwindow:
#            MLQ.io.log(f"train start at {trainstart} end at {trainend}, test start at {testsart} end at {testend}", logLoc=self.logLoc)
#            filt = self.featureFilter(self.Param['featureParam'], \
#                            trainstart, trainend)  #   a. 创建featureSelection对象
#            featureNames = filt.filtFeature()
#            self.saveFilter(filt)  # 保存因子过滤器self.store
#            self.Param['modelParam']['_trainstart'] = trainstart
#            self.Param['modelParam']['_trainend'] = trainend # 在Param['modelParam']中加入该窗口开始结束日期（为了兼容generalModeling）
#            model = self.model(featureNames, self.Param['modelParam']) #   b. 创建Model对象, 
#            Datai = self.Data[(self.Data['date']<=trainend)&\
#                              (self.Data['date']>=trainstart)] # 该段训练数据
#            Xi = Datai[featureNames]
#            Yi = Datai[self.Param['predict_label']]  # 获取该模型的训练数据 
#            model.train(Xi, Yi)      # 训练模型
#            self.saveModel(model) #   c. 保存self.model及self.store 
#            self.evaluateModel(model) #   c. 模型样本内评价 model 
#            self.testModel(model) #   d. 样本外模型预测 result
#        self.getReport()# 3. 生成模型报告
#    # 加载数据
#    def loadData(self):
#        if type(self.Data)==type(pd.Dataframe):
#            pass
#        else:
#            self.Data = pd.read_parquet(self.Param['featureDir']) # 从文件中读取数据
#    # 生成滑动训练窗口
#    def getRollingWindow(self, startResultDate, endResultDate, \
#                      trainLen=750, testLen=50, step=5, tradeDates=None):
#        if tradeDates==None:
#            tradeDates = sorted(self.Data['date'].unique())
#        testStarts = [d for d in tradeDates if d>=startResultDate][0:-1:testLen]
#        testStarts = [d for d in testStarts if d<=endResultDate]  # 所有测试区间开始点只需要在结果结束日期前即可
#        testEnds = [[d for d in tradeDates if d<s][-1] for s in testStarts[1:]]
#        testEnds.append([d for d in tradeDates if d<=endResultDate][-1])
#        def offset(date, n): # 前移n天,日期不足取第一天
#            preDates = [d for d in tradeDates if d<date]
#            return tradeDates[0] if n>len(preDates) else preDates[-n]
#        trainStarts = [offset(d, step+trainLen) for d in testStarts]
#        trainEnds = [offset(d, step+1) for d in testStarts]
#        return zip(trainStarts, trainEnds, testStarts, testEnds)  # 训练集开始结束日期,测试集开始日期
#    def saveFilter():
#        pass
#    def saveModel():
#        pass
#    def evaluateModel():
#        pass
#    def testModel():
#        pass
#    def getReport():
#        pass






