import numpy as np
import pandas as pd

# 极简回测框架

# 输入全部交易品种的截面收益率DataFrame(某元素为从上一个index时刻到该index时刻该列资产收益率)
class BackTest():
    def __init__(self, dfReturns, param, Strategy):
        self.dfReturns = dfReturns.fillna(0)
        self.param = param
        self.save = {} # 用于保存回测中处理的变量
        self.strat = Strategy(param.get("stratParam", {}))
    def main(self):
        # 按信号交易
        self.save["pnl"] = [0] # 每日盈亏（单利计算）
        self.save["position"] = [self.param.get("initPosition", \
            {k:0 for k in self.dfReturns.columns})] # 每日市值(货值)仓位,底仓默认为0
        self.save["amount"] = [0] # 每日成交额
        tradeTimes = list(self.dfReturns.index)
        self.save["tradeTimes"] = sorted([t for t in tradeTimes if \
            (t>=pd.to_datetime(self.param.get("start", "20000101")))&\
                (t<=pd.to_datetime(self.param.get("end", "22000101")))])
        # ======================= 以上初始化, 开始遍历全部交易截面 ======================
        for t in self.save["tradeTimes"]:
            self.save["now"] = t
            # ======================= 以下运行策略逻辑 ======================
            targetPosition = self.strat.trade(self.save)  # 策略对象逻辑,返回目标交易仓位(归1化，和不为1部分为现金)
            # ======================= 以下执行交易,计算盈亏（百分比) ==========================
            ser = pd.Series(self.save["position"][-1])
            #cash = 1-sum(self.save["position"][-1].values())
            #chgPosition = (self.dfReturns.loc[t]*ser + ser).dropna() # 资产价格变动造成的仓位变动
            #self.save["pnl"].append(chgPosition.sum() + cash - 1)  # 上一交易时间仓位确定盈亏
            #if targetPosition is None: # 如果没有目标仓位则根据资产涨幅得到最新仓位
            #    self.save["position"].append((chgPosition/(chgPosition.sum()+cash)).to_dict())
            #    self.save["amount"].append(0)
            #else: # 如果有目标仓位则由最新仓位交易至目标仓位
            #    self.save["position"].append(targetPosition)
            #    all_keys = targetPosition.keys() | chgPosition.keys()
            #    self.save["amount"].append(sum(abs(targetPosition.get(k, 0)*(1+self.save["pnl"][-1]) \
            #                    - chgPosition.get(k, 0)) for k in all_keys)) # 从chgPosition变动到targetPosition*(1+涨跌幅)
            cash = 1-abs(ser).sum() # 做空做多均按100%保证金计算
            profitPosition = (self.dfReturns.loc[t]*ser).dropna() # 各持仓盈亏
            self.save["pnl"].append(profitPosition.sum())  # 上一交易时间仓位确定盈亏
            chgPosition = (np.sign(ser)*(profitPosition+abs(ser))/((profitPosition+abs(ser)).sum()+cash)).to_dict()
            if targetPosition is None: # 如果没有目标仓位则根据资产涨幅得到最新仓位
                self.save["position"].append(chgPosition)
                self.save["amount"].append(0)
            else: # 如果有目标仓位则由最新仓位交易至目标仓位
                self.save["position"].append(targetPosition)
                all_keys = targetPosition.keys() | chgPosition.keys()
                self.save["amount"].append(sum(abs(targetPosition.get(k, 0)*(1+self.save["pnl"][-1]) \
                                - chgPosition.get(k, 0)) for k in all_keys)) # 从chgPosition变动到targetPosition*(1+涨跌幅)
class Strategy():
    def __init__(self, param):
        self.param = param


