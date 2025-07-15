import pandas as pd
import MLQuant as MLQ
import datetime, os


# 滚动窗口训练, 样本内外测试
class Modeling():
    # 配置文件Param，数据、模型、特征筛选器、log地址等均可从配置文件中调用，
    # 也可以在
    def __init__(self, Param, Data=None, \
                 model=None, featureFilter=None, logLoc=None):
        self.Param = Param  # {'trainParam':***, 'featureParam':***, 'modelParam':***, 'logLoc':***}
        self.Data = Data
        if type(featureFilter)!=type(None):
            self.featureFilter = featureFilter # class featureFilter 
        else:
            self.featureFilter = MLQ.io.importMyClass(Param['featureParam']['selectFilter'])
        if type(model)!=type(None):
            self.model = model # class Model
        else:
            self.model = MLQ.io.importMyClass(Param['modelParam']['selectModel'])
        if 'logLoc' not in Param.keys():
            self.logLoc = logLoc
    def run(self): 
        MLQ.io.log("加载数据") 
        self.loadData()
        MLQ.io.log("开始滑动窗口建模") 
        # 1. 生成滑动训练/测试窗口;
        self.rollingWindow = self.getRollingWindow(\
            self.Param['trainParam']['startResultDate'], self.Param['trainParam']['endResultDate'], \
                self.Param['trainLen'], self.Param['testLen'], self.Param['step'])
        # 2. 滚动窗口样本内外训练测试;
        for trainstart, trainend, testsart, testend in self.rollingwindow:
            MLQ.io.log(f"train start at {trainstart} end at {trainend}, test start at {testsart} end at {testend}", logLoc=self.logLoc)
            filt = self.featureFilter(self.Param['featureParam'], \
                            trainstart, trainend)  #   a. 创建featureSelection对象
            featureNames = filt.filtFeature()
            self.saveFilter(filt)  # 保存因子过滤器self.store
            self.Param['modelParam']['_trainstart'] = trainstart
            self.Param['modelParam']['_trainend'] = trainend # 在Param['modelParam']中加入该窗口开始结束日期（为了兼容generalModeling）
            model = self.model(featureNames, self.Param['modelParam']) #   b. 创建Model对象, 
            Datai = self.Data[(self.Data['date']<=trainend)&\
                              (self.Data['date']>=trainstart)] # 该段训练数据
            Xi = Datai[featureNames]
            Yi = Datai[self.Param['predict_label']]  # 获取该模型的训练数据 
            model.train(Xi, Yi)      # 训练模型
            self.saveModel(model) #   c. 保存self.model及self.store 
            self.evaluateModel(model) #   c. 模型样本内评价 model 
            self.testModel(model) #   d. 样本外模型预测 result
        self.getReport()# 3. 生成模型报告
    # 加载数据
    def loadData(self):
        if type(self.Data)==type(pd.Dataframe):
            pass
        else:
            self.Data = pd.read_parquet(self.Param['featureDir']) # 从文件中读取数据
    # 生成滑动训练窗口
    def getRollingWindow(self, startResultDate, endResultDate, \
                      trainLen=750, testLen=50, step=5, tradeDates=None):
        if tradeDates==None:
            tradeDates = sorted(self.Data['date'].unique())
        testStarts = [d for d in tradeDates if d>=startResultDate][0:-1:testLen]
        testStarts = [d for d in testStarts if d<=endResultDate]  # 所有测试区间开始点只需要在结果结束日期前即可
        testEnds = [[d for d in tradeDates if d<s][-1] for s in testStarts[1:]]
        testEnds.append([d for d in tradeDates if d<=endResultDate][-1])
        def offset(date, n): # 前移n天,日期不足取第一天
            preDates = [d for d in tradeDates if d<date]
            return tradeDates[0] if n>len(preDates) else preDates[-n]
        trainStarts = [offset(d, step+trainLen) for d in testStarts]
        trainEnds = [offset(d, step+1) for d in testStarts]
        return zip(trainStarts, trainEnds, testStarts, testEnds)  # 训练集开始结束日期,测试集开始日期
    def saveFilter():
        pass
    def saveModel():
        pass
    def evaluateModel():
        pass
    def testModel():
        pass
    def getReport():
        pass






