import pandas as pd
import datetime, os

def log(*txt, logLoc=''):
    try:
        f = open(os.path.join(logLoc,'log.txt'),'a+', encoding='utf-8')
        write_str = ('\n'+' '*35).join([str(i) for i in txt])
        f.write('%s,        %s\n' % \
            (datetime.datetime.now(), write_str))
        f.close()
    except PermissionError as e:
        print(f"Error: {e}. You don't have permission to access the specified file.")

# 滚动窗口训练, 样本内外测试
class Modeling():
    # 输入数据以及模型参数
    def __init__(self, Data, Param, model, featureFilter=None, logLoc='./'):
        self.Data = Data.reset_index(drop=True)  # 避免index不唯一
        self.Param = Param  # {'trainParam':***, 'featureParam':***, 'modelParam':***}
        self.featureFilter = featureFilter # class Filter
        self.model = model # class Model
        self.logLoc = logLoc
    def run(self): 
        log("开始Modeling") 
        # 1. 生成滑动训练/测试窗口;
        self.rollingWindow = self.getRollingWindow(\
            self.Param['trainParam']['startResultDate'], self.Param['trainParam']['endResultDate'], \
                self.Param['trainLen'], self.Param['testLen'], self.Param['step'])
        # 2. 滚动窗口样本内外训练测试;
        for trainstart, trainend, testsart, testend in self.rollingwindow:
            log(f"train start at {trainstart} end at {trainend}, test start at {testsart} end at {testend}", logLoc=self.logLoc)
            filter = self.featureFilter(self.Param['featureParam'], trainstart, trainend)  #   a. 创建featureSelection对象
            featureNames = filter.filtFeature()
            self.saveFilter(filter)  # 保存因子过滤器self.store
            model = self.model(featureNames, self.Param['modelParam']) #   b. 创建Model对象, 
            Xi = self.Data[featureNames]
            Yi = self.Data  # 获取该模型的训练数据 
            model.train(Xi, Yi)      # 训练模型
            self.saveModel(model) #   c. 保存self.model及self.store 
            self.evaluateModel(model) #   c. 模型样本内评价 model 
            self.testModel(model) #   d. 样本外模型预测 result
        self.getReport()# 3. 生成模型报告
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






