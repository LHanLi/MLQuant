import pandas as pd
import numpy as np
import MLQuant as MLQ
import json, pickle, time, threading, glob, os


class Modeling():
    def __init__(self, param={}, data=None, Filter=None, Model=None):
        if type(param)==type("outPath"): # 可以直接输入模型输出位置生成Modeling
            self.param = MLQ.io.readjson(os.path.join(param, "report", "param.json"))
            self.param["trainParam"]["outPath"] = param
        else:
            self.param = param
        self.data = data
        self.Filter = Filter
        self.Model = Model 
        self.dataFinished = []
    # 构建模型主流程 
    def main(self):
        self.prepare()
        if self.data is None:# 启动异步数据加载线程
            loader_thread = threading.Thread(target=self.loadData, daemon=True)
            loader_thread.start()
        else:
            if (self.data["date"].min()==self.rollingWindow[0][0])&(self.data["date"].max()==self.rollingWindow[-1][-1]):
                for i in range(len(self.rollingWindow)+1):
                    self.dataFinished.append(i) # 全部数据标记为已读
            else:
                raise ValueError("数据不匹配trainParam")
        for processNumber in range(len(self.rollingWindow)):
            self.train(processNumber)
        for processNumber in range(len(self.rollingWindow)):
            self.test(processNumber)
        self.report()
    # 创建工作目录,构建滑动窗口
    def prepare(self):
        # 模型构建在此目录构建
        os.makedirs(self.param["trainParam"]["outPath"], exist_ok=True)
        MLQ.io.savejson(self.param, os.path.join(self.param["trainParam"]["outPath"], "report", "param.json"))
        # 滑动窗口划分
        tradeDates = sorted([int(d) for d in os.listdir(self.param["trainParam"]["featureDir"])])
        valid_test_starts = [d for d in tradeDates if self.param["trainParam"]["strResultDate"] <= d\
                              <= self.param["trainParam"]["endResultDate"]]
        testStart_list = valid_test_starts[::self.param["trainParam"]["testSetLen"]]
        testStartEnd_list = list(zip(testStart_list, [[d for d in valid_test_starts if d < \
                nextStart][-1] for nextStart in testStart_list[1:]] + [valid_test_starts[-1]]))
        rollingWindow = []
        for testStart, testEnd in testStartEnd_list:
            testPreStart = tradeDates[max(tradeDates.index(testStart)\
                +1-self.param["trainParam"]["windowLen"], 0)]  # 间隔完整stepTrainAndTest天
            trainEnd = tradeDates[max(tradeDates.index(testStart)\
                -1-self.param["trainParam"]["stepTrainAndTest"], 0)]  # 间隔完整stepTrainAndTest天
            trainStart = tradeDates[max(tradeDates.index(trainEnd)\
                +1-self.param["trainParam"]["trainSetLen"], 0)] # 闭区间共trainSetLen天
            trainPreStart = tradeDates[max(tradeDates.index(trainStart)\
                +1-self.param["trainParam"]["windowLen"], 0)] # 闭区间共windowLen天
            rollingWindow.append([trainPreStart, trainStart, trainEnd, testPreStart, testStart, testEnd])    
        self.tradeDates = tradeDates
        self.rollingWindow = rollingWindow
        parts = [
            " " * 35 + f"第{i+1}滑动窗口 train:{rw[0]}-{rw[1]}-{rw[2]}, test:{rw[3]}-{rw[4]}-{rw[5]}"
            for i, rw in enumerate(rollingWindow)
        ]
        msg = "prepare,滚动窗口划分:\n" + ("\n").join(parts)
        self.log(msg)
    def loadData(self):
        # 包含元素1/2...表示对应滚动窗口的训练数据加载完毕,包含0表示全部数据加载完毕
        for i in range(len(self.rollingWindow)+1):
            if i==0:
                loadStart = self.rollingWindow[i][0]
                loadEnd = self.rollingWindow[i][2]
                self.log(f"loadData,加载第1个滚动窗口训练所需数据,{loadStart}-{loadEnd}")
            elif i==len(self.rollingWindow): 
                loadStart = max(self.rollingWindow[-1][0], [d for d in self.tradeDates if d>self.rollingWindow[i-1][2]][0])
                loadEnd = self.rollingWindow[-1][-1]
                self.log(f"loadData,加载全部剩余未加载数据,{loadStart}-{loadEnd}")
            else: 
                loadStart = max(self.rollingWindow[i][0], [d for d in self.tradeDates if d>self.rollingWindow[i-1][2]][0])
                loadEnd = self.rollingWindow[i][2]
                self.log(f"loadData,加载第{i+1}个滚动窗口训练所需未加载数据,{loadStart}-{loadEnd}")
            feature = MLQ.io.loadDataFrame(self.param["trainParam"]["featureDir"], \
                (loadStart, loadEnd), (93000000, 150000000))
            predict = MLQ.io.loadDataFrame(self.param["trainParam"]["predictDir"], \
                (loadStart, loadEnd), (93000000, 150000000)).fillna(0)
            if i==0:
                self.data = feature.merge(predict[["date", "curTime", "symbol", \
                    self.param["trainParam"]["predictLabel"]]], on=["date", "curTime", "symbol"]).\
                        sort_values(by=["date", "curTime", "symbol"]).reset_index(drop=True)  
            else:
                self.data = pd.concat([self.data, feature.merge(predict[["date", "curTime", "symbol", \
                    self.param["trainParam"]["predictLabel"]]], on=["date", "curTime", "symbol"])]).\
                        sort_values(by=["date", "curTime", "symbol"]).reset_index(drop=True)
            self.log(f"loadData,第{i+1}次数据加载完成,当前全部数据:{self.data.shape}, date:{self.data['date'].min()}-{self.data['date'].max()}, "+
                   f"curTime:{self.data['curTime'].min()}-{self.data['curTime'].max()}") # 数据全局有效，索引固定
            if i==len(self.rollingWindow):
                self.dataFinished.append(0)
            else:
                self.dataFinished.append(i+1)
    def loadRolllingWindowData(self, i):
        pass
    # 训练模型
    def train(self, processNumber):
        trainPreStart, trainStart, trainEnd, testPreStart, testStart, testEnd = self.rollingWindow[processNumber]
        while (processNumber+1) not in self.dataFinished:
            self.log(f"train,第{processNumber+1}个滚动窗口训练所需数据未完全加载,等待10s")
            time.sleep(10)
        self.log(f"train,第{processNumber+1}滚动窗口train({trainPreStart}-{trainStart}-{trainEnd})数据已加载完毕,开始train")
        # 创建该滑动窗口模型存储文件夹
        modelLoc = os.path.join(self.param["trainParam"]["outPath"], "model", f"{testStart}_startTest", f"{self.data['curTime'].min()}_{self.data['curTime'].max()}")
        os.makedirs(modelLoc, exist_ok=True)
        if "model.pkl" in os.listdir(modelLoc):
            self.log(f"train,第{processNumber+1}滚动窗口train已完成,跳过")
            return
        self.log("train,实例化Filter获取featureNames")
        if self.Filter is None:
            filter = MLQ.io.importMyClass(self.param["featureParam"]["selectFilter"])(self.param["featureParam"])
        else:
            filter = self.Filter(self.param["featureParam"])
        featureNames = filter.filtFeature(self.data.loc[\
            (self.data["date"]>=trainPreStart)&(self.data["date"]<=trainEnd)])
        featureNames = [f for f in featureNames if f!=self.param["trainParam"]["predictLabel"]]
        filter.store["window"] = self.rollingWindow[processNumber]
        filter.store["featureNames"] = featureNames
        filter.saveFilter(modelLoc)
        if len(featureNames)>10:
            self.log(f"train,特征筛选器选取{','.join(featureNames[:10])}, 等共{len(featureNames)}个因子")
        else:
            self.log(f"train,特征筛选器选取{','.join(featureNames)}, 共{len(featureNames)}个因子")
        self.log(f"train,获取训练集特征张量") 
        Xi, Yi, predictIndex = self.getTensor(self.data, featureNames, trainPreStart, trainStart, trainEnd)
        self.log(f"train,得到张量:Xi({Xi.shape}),Yi({Yi.shape}),实例化Model,开始训练模型") 
        if self.Model is None:
            model = MLQ.io.importMyClass(self.param["modelParam"]["selectModel"])(self.param["modelParam"])
        else:
            model = self.Model(self.param["modelParam"])
        # 模型训练
        model.train(Xi, Yi)
        model.store["window"] = self.rollingWindow[processNumber]
        self.log("train,模型训练完毕")
        df_train = self.data.iloc[predictIndex][["date", "curTime", "symbol", \
            self.param["trainParam"]["predictLabel"]]]
        df_train["predict"] = model.predict(Xi)
        model.store["train"] = df_train.to_dict()
        self.log(model.log)
        model.saveModel(modelLoc)
        model.cleanup()
        del model
        self.log("train,已保存模型及训练集结果")
    # 模型评价
    def test(self, processNumber):
        while 0 not in self.dataFinished:
            self.log("test,测试所需数据完全加载,等待10s")
            time.sleep(10)
        trainPreStart, trainStart, trainEnd, testPreStart, testStart, testEnd = self.rollingWindow[processNumber]
        self.log(f"test,开始读取第{processNumber+1}个滑动窗口已训练模型")
        # 读取该滑动窗口对应的特征筛选器及模型
        modelLoc = os.path.join(self.param["trainParam"]["outPath"], "model", f"{testStart}_startTest", f"{self.data['curTime'].min()}_{self.data['curTime'].max()}")
        filter = self.restoreFilter(modelLoc)
        model = self.restoreModel(modelLoc)
        df_train = pd.DataFrame(model.store["train"])
        self.log("test,全部数据已加载完毕,测试模型")
        # 所有模型测试从该滑动窗口测试集开始到最后一个滑动窗口测试集最后时间
        Xi_test, Yi_test, predictIndex_test = self.getTensor(self.data, filter.store["featureNames"], testPreStart, testStart, self.rollingWindow[-1][-1])
        self.log(f"test,得到测试数据{Xi_test.shape}")
        df_test = self.data.iloc[predictIndex_test][["date", "curTime", "symbol",\
            self.param["trainParam"]["predictLabel"]]]
        df_test["predict"] = model.predict(Xi_test) 
        from sklearn.metrics import mean_squared_error, r2_score
        # 日期IC
        ICdate_train = df_train.groupby("date")[["predict", self.param["trainParam"]["predictLabel"]]].corr().\
            iloc[0::2, 1].reset_index(level=1, drop=True)
        ICdate_test = df_test.groupby("date")[["predict", self.param["trainParam"]["predictLabel"]]].corr().\
            iloc[0::2, 1].reset_index(level=1, drop=True)
        rmse_train = np.sqrt(mean_squared_error(df_train[self.param["trainParam"]["predictLabel"]], df_train["predict"]))
        r2_train = r2_score(df_train[self.param["trainParam"]["predictLabel"]], df_train["predict"])
        ic_train = 100*np.sqrt(r2_train)  # 样本内
        rmse_test = np.sqrt(mean_squared_error(df_test[self.param["trainParam"]["predictLabel"]], df_test["predict"]))
        r2_test = r2_score(df_test[self.param["trainParam"]["predictLabel"]], df_test["predict"])
        ic_test = 100*np.sqrt(r2_test)  # 样本内
        plt, fig, ax = MLQ.post.matplot(2, 2)
        ax[0][0].scatter(df_train[self.param["trainParam"]["predictLabel"]], df_train["predict"])
        ax[0][0].set_title(f"rmse:{rmse_train:.5f}, R2:{r2_train:.2f}, IC:{ic_train:.2f}")
        ax[0][1].scatter(df_test[self.param["trainParam"]["predictLabel"]], df_test["predict"])
        ax[0][1].set_title(f"rmse:{rmse_test:.5f}, R2:{r2_test:.2f}, IC:{ic_test:.2f}")
        ax[1][0].plot(ICdate_train.cumsum().values)
        ax[1][0].set_title(f"IC:{100*ICdate_train.mean():.2f}, ICIR:{(ICdate_train.mean()/ICdate_test.std()):.2f}, "\
                        f"rollingICIR:{(ICdate_train.rolling(5).mean().mean()/ICdate_train.rolling(5).mean().std()):.2f}")
        ax[1][0].set_xlabel(f"从{df_train['date'].iloc[0]}到{df_train['date'].iloc[-1]}")
        ax[1][1].plot(ICdate_test.cumsum().values)
        ax[1][1].set_title(f"IC:{100*ICdate_test.mean():.2f}, ICIR:{(ICdate_test.mean()/ICdate_test.std()):.2f}, "\
                        f"rollingICIR:{(ICdate_test.rolling(5).mean().mean()/ICdate_test.rolling(5).mean().std()):.2f}")
        ax[1][1].set_xlabel(f"从{df_test['date'].iloc[0]}到{df_test['date'].iloc[-1]}")
        plt.savefig(os.path.join(modelLoc, "predict.png"))
        self.log("test,保存模型及测试结果")
        model.store["test"] = df_test.to_dict()
        MLQ.io.saveDataFrame(df_test[(df_test["date"]>=testStart)&(df_test["date"]<=testEnd)], os.path.join(self.param["trainParam"]["outPath"], "result"))
        model.saveModel(modelLoc)
        model.cleanup()
        del model
    # 训练完毕后生成模型报告
    def report(self):
        self.log("report,开始评价result")
        result = MLQ.io.loadDataFrame(os.path.join(self.param["trainParam"]["outPath"], "result")).sort_values(by=["date", "curTime", "symbol"])
        ICdate = result.groupby("date")[[self.param["trainParam"]["predictLabel"], "predict"]].corr().iloc[0::2, 1].reset_index(level=1, drop=True)
        plt, fig, ax = MLQ.post.matplot()
        ax.plot(ICdate.cumsum().values)
        ax.set_title(f"IC:{100*ICdate.mean():.2f}, ICIR:{(ICdate.mean()/ICdate.std()):.2f}, "\
                f"rollingICIR:{(ICdate.rolling(5).mean().mean()/ICdate.rolling(5).mean().std()):.2f}")
        plt.savefig(os.path.join(self.param["trainParam"]["outPath"], "report", "IC.png"))
    # 加载模型(包括特征筛选器)
    def restoreFilter(self, modelLoc):
        if self.Filter is None:
            filter = MLQ.io.importMyClass(self.param["featureParam"]["selectFilter"])(self.param["featureParam"])
        else:
            filter = self.Filter(self.param["featureParam"]) 
        filter.restoreFilter(modelLoc)
        return filter
    def restoreModel(self, modelLoc):
        if self.Model is None:
            model = MLQ.io.importMyClass(self.param["modelParam"]["selectModel"])(self.param["modelParam"])
        else:
            model = self.Model(self.param["modelParam"])
        model.restoreModel(modelLoc) 
        return model
    # 实盘调用模型预测数据(注意，默认使用最新模型，会自动丢弃legalData为False的数据)
    def predict(self, data):
        # 读取最新模型
        modelLoc = os.path.join(self.param["trainParam"]["outPath"], "model")
        modelLoc = os.path.join(modelLoc, os.listdir(modelLoc)[-1], f"{data['curTime'].min()}_{data['curTime'].max()}")
        filter = self.restoreFilter(modelLoc) 
        model = self.restoreModel(modelLoc)
        Xi, Yi, predictIndex = self.getTensor(data, filter.store["featureNames"], data["date"].unique()[0],\
            data["date"].unique()[self.param["trainParam"]["windowLen"]-1], data["date"].unique()[-1], False)
        return predictIndex, model.predict(Xi)
    # 返回B*T*m张量,B是legalData数量,回看窗口是T,m个特征(注意，回看窗口中可以包含illegal)
    def getTensor(self, data, featureNames, datePreStart, dateStart, dateEnd, predict=True):
        # 如果特征列包含date, curTime, symbol索引列 需要重命名
        for c in ["date", "curTime", "symbol"]:
            if c in featureNames:
                data[c+"1"] = data[c]
        featureNames = [c if c not in ["date", "curTime", "symbol"] else c+"1" for c in featureNames]
        # 获取包含时序信息的特征张量
        predictIndex = data[(data["date"]>=dateStart)&\
            (data["date"]<=dateEnd)&data["legalData"]].index
        featureIndex = data[(data["date"]>=datePreStart)&\
            (data["date"]<=dateEnd)].index
        # 预测目标值
        if predict:
            Yi = np.array(data.iloc[predictIndex]\
                [self.param["trainParam"]["predictLabel"]]).reshape(-1, 1)
        else:
            Yi = None
        # 特征数据图片 batch*seq*featureSize
        Xi_shift = data.iloc[featureIndex][["date", "curTime", "symbol"]+featureNames].set_index(\
            ['date', "curTime", 'symbol']) # 回看窗口为连续分钟,index去掉curTime则只取过去T个date
        Xi = np.array(Xi_shift).reshape(-1, 1, len(Xi_shift.columns))
        for i in range(self.param["trainParam"]["windowLen"]-1):
            Xi_shift = Xi_shift.groupby('symbol').shift().fillna(0) # 如果数据不足windowLen则补0
            Xi = np.concatenate(\
                (np.array(Xi_shift).reshape(-1, 1, len(Xi_shift.columns)), Xi), axis=1)
        # 获取对应predictIndex的featureIndex
        Xi_loc = []
        j = 0
        for i in range(len(featureIndex)):
            if featureIndex[i]==predictIndex[j]:
                Xi_loc.append(i)
                j += 1
        Xi = Xi[Xi_loc]
        if not self.param["trainParam"]["tensor"]: # 如果关闭张量模式则转化为DataFrame
            Xi = pd.DataFrame(Xi.reshape(Xi.shape[0], -1))
            if predict:
                Yi = pd.DataFrame(Yi.reshape(Yi.shape[0], -1))
        else:
            import torch  # 张量模式默认torch
            Xi = torch.from_numpy(Xi).float()   
            if predict:
                Yi = torch.from_numpy(Yi).float()
        return Xi, Yi, predictIndex
    # log函数
    def log(self, logstr):
        MLQ.io.log(logstr, logLoc=self.param["trainParam"]["outPath"])


# ==============================
# 因子筛选基类
# ==============================
class Filter:
    def __init__(self, featureParam):
        self.featureParam = featureParam #因子筛选所需的其他参数
        self.store = {}
        self.log = "" # 保存实例log,输出到Modeling工作目录
    def saveFilter(self, modelDir): # 储存因子筛选器
        os.makedirs(modelDir, exist_ok=True)
        with open(os.path.join(modelDir,"filter_store.json"), 'w') as f:
            json.dump(MLQ.io.converjson(self.store), f, indent=4)
        with open(os.path.join(modelDir, 'filter.pkl'), 'wb') as f:
            pickle.dump(self, f)
    def restoreFilter(self,modelDir): #加载模型
        with open(os.path.join(modelDir, 'filter.pkl'), 'rb') as f:
            model = pickle.load(f)
            self.__dict__.update(model.__dict__)
    #def filtFeature(self, data):  # 需要自定义因子筛选过程，返回筛选后的特征名列表

# ==============================
# 模型基类
# ==============================
class Model:
    def __init__(self, modelParam={}):
        self.modelParam = modelParam
        self.store = {}
        self.log = "" # 保存实例log,输出到Modeling工作目录
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
    def cleanup(self):
        """显式释放模型相关资源"""
        import torch
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            del self.optimizer
        # 清空 store 中可能的大对象
        self.store.clear()
        # 强制垃圾回收
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    #def train(self, Xi, Yi):  # 需要自定义训练过程
    #def predict(self, Xi): # 需要自定义预测过程 self.model即可


