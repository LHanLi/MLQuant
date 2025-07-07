import pandas as pd


# 滚动窗口训练, 样本内外测试
class Modeling():
    # 输入数据以及模型参数
    def __init__(self, Data, Param, model, featureFilter=None):
        self.Data = Data  # 数据
        self.Param = Param  # {'trainParam':***, 'featureParam':***, 'modelParam':***}
        self.featureFilter = featureFilter # class Filter
        self.model = model # class Model
    def run(self): 
        self.rollingwindow = self.get_rollingwindow(\
            self.Param['trainParam']['startresultdate'], self.Param['trainParam']['endresultdate'], \
                self.Param['trainlen'], self.Param['testlen'], self.Param['step']) # 1. 生成滑动训练/测试窗口;
        # 2. 滚动窗口样本内外训练测试; 
        #   a. 创建featureSelection对象,featureFilter.filter 获取模型特征;
        #   b. 创建Model对象, model.train 训练模型 model
        #   c. model.saveModel 保存self.model及self.store 
        #   c. model.evaluate 模型样本内评价 model 
        #   d. model.test 样本外模型预测 result
        # 3. 生成模型报告
        pass
    # 生成滑动训练窗口
    def get_rollingwindow(self, startresultdate, endresultdate, \
                      trainlen=400, testlen=50, step=5, tradedates=None):
        if tradedates==None:
            tradedates = sorted(self.Data['date'].unique())
        teststarts = [d for d in tradedates if d>=startresultdate][0:-1:testlen]
        teststarts = [d for d in teststarts if d<=endresultdate]  # 所有测试区间开始点只需要在结果结束日期前即可
        testends = [[d for d in tradedates if d<s][-1] for s in teststarts[1:]]
        testends.append([d for d in tradedates if d<=endresultdate][-1])
        def offset(date, n): # 前移n天,日期不足取第一天
            predates = [d for d in tradedates if d<date]
            return tradedates[0] if n>len(predates) else predates[-n]
        trainstarts = [offset(d, step+trainlen) for d in teststarts]
        trainends = [offset(d, step+1) for d in teststarts]
        return zip(trainstarts, trainends, teststarts, testends)  # 训练集开始结束日期,测试集开始日期
    # train
    def train(self):
        for trains, traine, _, _ in self.rollingwindow:
            datai = aa27.data.readDates(self.data_loc, \
                        trains, traine, self.data_name, 'cc')
            x_names, y_names = self.featureFilter(datai)
            self.model.train(datai[x_names], datai[y_names])
    # test
    def test(self):
        pass



