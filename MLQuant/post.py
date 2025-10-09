import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import copy

# matplot绘图
def matplot(r=1, c=1, sharex=False, sharey=False, w=13, d=7, hspace=0.3, wspace=0.2):
    # don't use sns style
    sns.reset_orig()
    #plot
    #run configuration 
    plt.rcParams['font.size']=14
    plt.rcParams['font.family'] = 'KaiTi'
    #plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif']=['Microsoft YaHei']
    plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题
    plt.rcParams['axes.linewidth']=1
    plt.rcParams['axes.grid']=True
    plt.rcParams['grid.linestyle']='--'
    plt.rcParams['grid.linewidth']=0.2
    plt.rcParams["savefig.transparent"]='True'
    plt.rcParams['lines.linewidth']=0.8
    plt.rcParams['lines.markersize'] = 1
    
    #保证图片完全展示
    plt.tight_layout()
        
    #subplot
    fig,ax = plt.subplots(r,c,sharex=sharex, sharey=sharey,figsize=(w,d))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace = hspace, wspace=wspace)
    plt.gcf().autofmt_xdate()
    return plt, fig, ax

class Post():
    def __init__(self, result, dailyAlpha, dailyPos):
        self.result = copy.copy(result)
        self.result["date"] = pd.to_datetime(self.result["date"].astype("string"))
        # 计算IC和benchmark
        self.sectionsCorr = self.result[["date", "curTime", "predict", "Nbr240"]].groupby(['date', 'curTime']).corr().loc[:, "predict"].loc[:, :, "Nbr240"]
        #benchmark = (result.groupby(["date", "curTime"])["Nbr240"].mean()+1).groupby("date").prod()-1 # 一天所有截面
        # 每日净值
        self.dailyAlpha = copy.copy(dailyAlpha)
        self.dailyAlpha["date"] = pd.to_datetime(self.dailyAlpha["date"].astype("string"))
        self.dailyAlpha = self.dailyAlpha.set_index("date")
        self.chgDay = self.dailyAlpha["value"]/self.dailyAlpha["value"].shift().fillna(1)-1
        # 持仓
        self.dailyPos = copy.copy(dailyPos)
        self.dailyPos["date"] = pd.to_datetime(self.dailyPos["date"].astype("string"))
        self.dailyPos = self.dailyPos.set_index("date")
    # 按天对净值进行分析
    def daily(self, benchmark, loc=""):
        self.benchmark = benchmark
        plt, fig, ax = matplot(2, 2, w=16, d=8)
        # 收益分析
        self.ann = self.dailyAlpha["value"].iloc[-1]**(250/len(self.dailyAlpha))-1
        self.sharpe = self.ann/(np.sqrt(250)*self.chgDay.std())
        self.drawdown = (self.dailyAlpha["value"]/self.dailyAlpha["value"].cummax()-1)
        # CAMP模型(很可能有很强的异方差和自相关问题)
        x = sm.add_constant(benchmark.values.reshape(-1, 1))
        y = self.chgDay.values
        capm = sm.OLS(y, x).fit()
        # 同方差检验
        import statsmodels.stats.api as sms
        from statsmodels.compat import lzip
        name = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
        test = sms.het_breuschpagan(capm.resid, capm.model.exog)
        print(lzip(name, test)) # p>0.05表示同方差
        # 自相关检验
        from statsmodels.stats.stattools import durbin_watson
        dw = durbin_watson(capm.resid)
        print(f"Durbin-Watson: {dw}")  # 接近 2 表示无自相关
        ax[0][0].set_title(f"年化: {100*self.ann:.2f}%, 夏普: {self.sharpe:.2f}, beta: {capm.params[1]:.2f}, alpha: {100*capm.params[0]:.2f}% ({test[1]:.2f}, {dw:.1f})\n"\
            f"每日一遇:{100*self.chgDay.quantile(0.5):.2f}%, 每周一遇:{100*self.chgDay.quantile(0.2):.2f}/{100*self.chgDay.quantile(0.8):.2f}%, 每月一遇:{100*self.chgDay.quantile(0.05):.2f}/{100*self.chgDay.quantile(0.95):.2f}%")
        ax[0][0].plot(np.log(self.dailyAlpha['value']), linewidth=2, c="C3")
        ax[0][0].plot(np.log((benchmark+1).cumprod()), c="C3")
        ax[0][0].vlines(self.drawdown.index[self.drawdown.argmin()], 0, np.log(self.dailyAlpha["value"].max()), linewidth=2, color="C3", linestyles="--")
        ax[0][0].vlines(self.chgDay.index[self.chgDay.argmin()], 0, np.log(self.dailyAlpha["value"].max()), color="C7", linestyles="--")
        ax[0][0].set_ylabel("对数净值")

        # 超额分析
        self.excess = self.chgDay-self.benchmark
        self.excess_ann = self.dailyAlpha["value"].iloc[-1]/(1+benchmark).prod()-1
        self.excess_sharpe = self.excess_ann/(np.sqrt(250)*self.excess.std())
        excess_cum = (1+self.excess).cumprod()
        excess_drawdown = (excess_cum/excess_cum.cummax()-1)
        ax[1][0].plot(np.log(self.excess+1).cumsum(), c="C2", linewidth=2)
        ax[1][0].set_title(f"年化: {100*self.excess_ann:.2f}, 夏普: {self.excess_sharpe:.2f}\n"
                f"每日一遇:{100*self.excess.quantile(0.5):.2f}%, 每周一遇:{100*self.excess.quantile(0.2):.2f}/{100*self.excess.quantile(0.8):.2f}%, 每月一遇:{100*self.excess.quantile(0.05):.2f}/{100*self.excess.quantile(0.95):.2f}%")
        ax[1][0].set_ylabel("对数超额净值")
        ax101 = ax[1][0].twinx()
        ax101.plot(self.sectionsCorr.groupby("date").mean().cumsum(), c="C0", linewidth=2)
        ax101.set_ylabel("累计IC")

        # 回撤分析
        ax[0][1].plot(self.drawdown, c="C3")
        ax[0][1].plot(excess_drawdown, c="C2")
        ax[0][1].set_title(f"最大回撤: {-100*self.drawdown.min():.2f}%, 单日最大回撤: {-100*self.chgDay.min():.2f}%\n"
                        f"超额最大回撤: {-100*excess_drawdown.min():.2f}%, 单日超额最大回撤: {-100*self.excess.min():.2f}%")
        ax[0][1].set_ylabel("回撤")

        # 持仓分析
        totalPosition = self.dailyPos.groupby('date')['marketValue'].sum()/(self.dailyAlpha["value"]*1e8)
        maxOnePosition = self.dailyPos.groupby("date")["marketValue"].max()/(self.dailyAlpha["value"]*1e8)
        meanOnePosition = self.dailyPos[self.dailyPos["marketValue"]!=0].groupby("date")["marketValue"].mean()/(self.dailyAlpha["value"]*1e8)
        ax[1][1].set_title(f"平均仓位: {100*totalPosition.mean():.2f}%, 最大单一持仓: {100*maxOnePosition.max():.2f}%, 日平均换手: {self.dailyAlpha['turnover'].mean():.2f}, \n"
                        f"平均持仓只数: {int(self.dailyPos[self.dailyPos["marketValue"]!=0].groupby("date")["symbol"].count().mean())}, 平均持仓仓位: {100*meanOnePosition.mean():.2f}%")
        ax[1][1].plot(totalPosition, c="C3")
        ax111 = ax[1][1].twinx()
        ax111.plot(maxOnePosition, c="C0")
        ax111.plot(meanOnePosition, c="C0", alpha=0.5)
        ax[1][1].set_ylabel("总仓位")
        ax111.set_ylabel("单一证券仓位")
        plt.tight_layout()
        if loc=="":
            plt.show()
        else:
            plt.savefig(loc)
            plt.show()


