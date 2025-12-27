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
    #plt.rcParams["savefig.transparent"]='True'
    plt.rcParams['lines.linewidth']=0.8
    plt.rcParams['lines.markersize'] = 1
    
    #保证图片完全展示
    plt.tight_layout()
        
    #subplot
    fig,ax = plt.subplots(r,c,sharex=sharex, sharey=sharey,figsize=(w,d))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace = hspace, wspace=wspace)
    plt.gcf().autofmt_xdate()
    return plt, fig, ax

# 模型样本内外预测效果
def plotPredict(result, ISStartDate, ISEndDate, OOSStartDate, OOSEndDate, namey, namey_pred): 
    from sklearn.metrics import mean_squared_error, r2_score
    df_IS = result[(result["date"]>=int(ISStartDate))&(result["date"]<int(ISEndDate))]
    datesicIS = df_IS[[namey, namey_pred, 'date']].groupby('date').\
                corr().iloc[0::2, 1].reset_index(level=1, drop=True)
    rmseIS = np.sqrt(mean_squared_error(df_IS[namey], df_IS[namey_pred]))
    r2IS = r2_score(df_IS[namey], df_IS[namey_pred])
    icIS = 100*np.sqrt(r2IS)  # 样本内

    df_OOS = result[(result["date"]>=int(OOSStartDate))&(result["date"]<int(OOSEndDate))]
    datesicOOS = df_OOS[[namey, namey_pred, 'date']].groupby('date').\
                corr().iloc[0::2, 1].reset_index(level=1, drop=True)
    rmseOOS = np.sqrt(mean_squared_error(df_OOS[namey], df_OOS[namey_pred]))
    r2OOS = r2_score(df_OOS[namey], df_OOS[namey_pred])
    icOOS = 100*np.sqrt(r2_score(df_OOS[namey], df_OOS[namey_pred]))  # 样本外

    plt, fig, ax = matplot(2, 2)
    # 样本内全截面
    ax[0][0].plot(np.linspace(min(min(df_IS[namey]), min(df_IS[namey_pred])),\
                    max(max(df_IS[namey]), max(df_IS[namey_pred]) ), 100),\
                   np.linspace(min(min(df_IS[namey]), min(df_IS[namey_pred])),\
                    max(max(df_IS[namey]), max(df_IS[namey_pred]) ), 100), c='C7')
    #ax[0][0].set_xlim(min(df_IS[namey]), max(df_IS[namey]))
    #ax[0][0].set_ylim(min(df_IS[namey_pred]), max(df_IS[namey_pred]))
    ax[0][0].scatter(df_IS[namey], df_IS[namey_pred])
    ax[0][0].set_xlabel('IS 目标值')
    ax[0][0].set_ylabel('全截面预测')
    ax[0][0].set_title(f"rmse:{rmseIS:.5f}, R2:{r2IS:.2f}, IC:{icIS:.2f}")
    # 样本外
    ax[0][1].plot(np.linspace(min(min(df_OOS[namey]), min(df_OOS[namey_pred])),\
                    max(max(df_OOS[namey]), max(df_OOS[namey_pred]) ), 100),\
                   np.linspace(min(min(df_OOS[namey]), min(df_OOS[namey_pred])),\
                    max(max(df_OOS[namey]), max(df_OOS[namey_pred]) ), 100), c='C7')
    #ax[0][1].set_xlim(min(df_OOS[namey]), max(df_OOS[namey]))
    #ax[0][1].set_ylim(min(df_OOS[namey_pred]), max(df_OOS[namey_pred]))
    ax[0][1].scatter(df_OOS[namey], df_OOS[namey_pred])
    ax[0][1].set_xlabel('OOS 目标值')
    ax[0][1].set_title(f"OOS rmse:{rmseOOS:.5f}, R2:{r2OOS:.2f}, IC:{icOOS:.2f}")
    # 样本内单截面累计
    ax[1][0].plot(datesicIS.cumsum().values)
    ax10_twinx = ax[1][0].twinx()
    ax10_twinx.plot(datesicIS.rolling(5).mean().values, c="C2")
    ax[1][0].set_title(f"IC:{100*datesicIS.mean():.2f}, ICIR:{(datesicIS.mean()/datesicIS.std()):.2f}, "\
                f"rollingICIR:{(datesicIS.rolling(5).mean().mean()/datesicIS.rolling(5).mean().std()):.2f}")
    ax[1][0].set_ylabel('单截面累计IC')
    ax[1][0].set_xlabel(f"从{df_IS['date'].iloc[0]}到{df_IS['date'].iloc[-1]}")
    # 样本外
    ax[1][1].plot(datesicOOS.cumsum().values)
    ax11_twinx = ax[1][1].twinx()
    ax11_twinx.plot(datesicOOS.rolling(5).mean().values, c="C2")
    ax[1][1].set_title(f"IC:{100*datesicOOS.mean():.2f}, ICIR:{(datesicOOS.mean()/datesicOOS.std()):.2f}, "\
                f"rollingICIR:{(datesicOOS.rolling(5).mean().mean()/datesicOOS.rolling(5).mean().std()):.2f}")
    ax[1][1].set_xlabel(f"从{df_OOS['date'].iloc[0]}到{df_OOS['date'].iloc[-1]}")
    return plt, fig, ax

# 模型衰减
def plotModelDecay(result, namey, namey_pred, modelStartList):
    IC = result.loc[result["legalData"]].groupby("date")\
        [[namey, namey_pred]].corr().loc[:, namey].loc[:, namey_pred]

    cutICs = [IC.loc[s:e].iloc[:-1].reset_index(drop=True) for s,e in \
        zip(modelStartList, modelStartList[1:]+[30000101])]
    cutICs = pd.concat(cutICs, axis=1)

    plt, fig, ax = matplot()
    ax.plot(cutICs.mean(axis=1), linewidth=2, c="C3")
    ax.set_ylabel("IC")
    ax1 = ax.twinx()
    ax1.plot(cutICs.std(axis=1), linewidth=5, alpha=0.3)
    ax1.set_ylabel("标准差")
    ax.set_title("模型衰减")

    return plt, fig, ax

# 日度收益率分析
def plotDay(returnDay, benchmark, loc=""):
    plt, fig, ax = matplot(2, 2, w=16, d=8)
    
    # CAMP模型(很可能有很强的异方差和自相关问题)
    x = sm.add_constant(benchmark.values.reshape(-1, 1))
    y = returnDay.values
    capm = sm.OLS(y, x).fit()
    import statsmodels.stats.api as sms
    from statsmodels.compat import lzip
    name = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
    test = sms.het_breuschpagan(capm.resid, capm.model.exog)
    from statsmodels.stats.stattools import durbin_watson
    dw = durbin_watson(capm.resid)
    fig.suptitle(f"CAPM模型, beta:{capm.params[1]:.2f}, alpha:{100*capm.params[0]:.2f}%, 残差收益率同方差p值:{test[1]:.2f}, 自相关检验:{dw:.1f}", fontsize=20)

    # === 策略和基准净值 ===
    net = (returnDay+1).cumprod()
    net_benchmark = (benchmark+1).cumprod()
    ann = net.iloc[-1]**(250/len(returnDay))-1
    sharpe = ann/(np.sqrt(250)*returnDay.std())
    drawdown = net/net.cummax()-1
    drawdown_benchmark = net_benchmark/net_benchmark.cummax()-1
    ax[0][0].plot(np.log(net), c="C3", linewidth=2)
    ax[0][0].plot(np.log(net_benchmark), c="C0")
    ax00_ = ax[0][0].twinx()
    ax00_.plot(100*drawdown, c="C3", alpha=0.5, linewidth=2)
    ax00_.plot(100*drawdown_benchmark, c="C0", alpha=0.5)
    ax[0][0].vlines(returnDay.index[returnDay.argmin()], np.log(min(net.min(), net_benchmark.min())), \
            np.log(max(net.max(), net_benchmark.max()))*1.1, \
                color="C3", linewidth=2, linestyles="--") # 单日最大回撤日期
    ax[0][0].set_title(f"年化: {100*ann:.2f}%, 夏普: {sharpe:.2f}, "\
        +f"最大回撤: {-100*drawdown.min():.1f}%, 单日最大回撤: {-100*returnDay.min():.1f}%") 
            # p<0.05表示显著异方差 范围(0, 4) 大于2表示负自相关，小于2表示正自相关
    ax[0][0].set_ylabel("对数净值")
    ax[0][0].tick_params(axis='x', labelbottom=True, rotation=30)
    import matplotlib.ticker as mticker # 日度收益直方图
    weights = np.ones_like(returnDay) / len(returnDay)
    n, bins, patches = ax[0][1].hist(returnDay, bins=20,  weights=weights, color='skyblue', edgecolor='black', alpha=0.7)
    ax[0][1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    ax[0][1].set_xticks(bin_centers)
    ax[0][1].set_xticklabels([f"{100*center:.2f}" for center in bin_centers], rotation=45, ha="right")
    ax[0][1].tick_params(axis='x', labelbottom=True)
    ax[0][1].set_title(f"每日一遇:{100*returnDay.quantile(0.5):.2f}%, 每周一遇:{100*returnDay.quantile(0.2):.2f}/{100*returnDay.quantile(0.8):.2f}%, 每月一遇:{100*returnDay.quantile(0.05):.2f}/{100*returnDay.quantile(0.95):.2f}%")
    ax[0][1].set_xlabel(f'日度收益率(%),胜率{(returnDay>0).mean()*100:.2f}%')
    ax[0][1].set_ylabel('区间收益占比(%)')

    ## === 超额收益 ===
    excess = returnDay-benchmark
    excess_ann = net.iloc[-1]/(1+benchmark).prod()-1
    excess_sharpe = excess_ann/(np.sqrt(250)*excess.std())
    excess_net = (1+excess).cumprod()
    excess_drawdown = (excess_net/excess_net.cummax()-1)
    ax[1][0].plot(np.log(excess_net), c="C3", linewidth=2)
    ax10_ = ax[1][0].twinx()
    ax10_.plot(100*excess_drawdown, c="C3", alpha=0.5, linewidth=2)
    ax[1][0].vlines(excess.index[excess.argmin()], np.log(excess_net.min()), np.log(excess_net.max())*1.1,\
                color="C3", linewidth=2, linestyles="--") # 单日最大回撤日期
    ax[1][0].set_title(f"年化: {100*excess_ann:.2f}%, 夏普: {excess_sharpe:.2f}, "\
        +f"最大回撤: {-100*excess_drawdown.min():.1f}%, 单日最大回撤: {-100*excess.min():.1f}%") 
    ax[1][0].set_ylabel("超额对数净值")
    import matplotlib.ticker as mticker # 日度收益直方图
    weights = np.ones_like(excess) / len(excess)
    n, bins, patches = ax[1][1].hist(excess, bins=20,  weights=weights, color='skyblue', edgecolor='black', alpha=0.7)
    ax[1][1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    ax[1][1].set_xticks(bin_centers)
    ax[1][1].set_xticklabels([f"{100*center:.2f}" for center in bin_centers], rotation=45, ha="right")
    ax[1][1].set_title(f"每日一遇:{100*excess.quantile(0.5):.2f}%, 每周一遇:{100*excess.quantile(0.2):.2f}/{100*excess.quantile(0.8):.2f}%, 每月一遇:{100*excess.quantile(0.05):.2f}/{100*excess.quantile(0.95):.2f}%")
    ax[1][1].set_xlabel(f'日度收益率(%),胜率{(excess>0).mean()*100:.2f}%')
    ax[1][1].set_ylabel('区间收益占比*5(%)')
    
    plt.tight_layout()
    if loc=="":
        plt.show()
    else:
        plt.savefig(loc)
        plt.show()



# 持仓分析
def plotHold():
    pass

class Post():
    def __init__(self, result, dailyAlpha, dailyPos, barAlpha=None):
        self.result = copy.copy(result)
        self.result["date"] = pd.to_datetime(self.result["date"].astype("string"))
        # 计算IC和benchmark
        self.sectionsCorr = self.result[["date", "curTime", "predict", "Nbr240"]].groupby(['date', 'curTime']).corr().loc[:, "predict"].loc[:, :, "Nbr240"]
        #benchmark = (result.groupby(["date", "curTime"])["Nbr240"].mean()+1).groupby("date").prod()-1 # 一天所有截面
        # 每日净值
        self.dailyAlpha = copy.copy(dailyAlpha)
        self.dailyAlpha["date"] = pd.to_datetime(self.dailyAlpha["date"].astype("string"))
        self.dailyAlpha = self.dailyAlpha.set_index("date")
        # 持仓
        self.dailyPos = copy.copy(dailyPos)
        self.dailyPos["date"] = pd.to_datetime(self.dailyPos["date"].astype("string"))
        self.dailyPos = self.dailyPos.set_index("date")
        # 逐bar净值
        if type(barAlpha)!=type(None):
            self.barAlpha = copy.copy(barAlpha)
            self.barAlpha["chg"] = (self.barAlpha["value"]/self.barAlpha["value"].shift()).fillna(1)-1
    # 按天对净值进行分析
    def daily(self, benchmark, dailyAlpha=None, loc=""):
        if type(dailyAlpha)==type(None):
            dailyAlpha = self.dailyAlpha
        plt, fig, ax = matplot(2, 2, w=16, d=8)
        # 收益分析
        self.chgDay = dailyAlpha["value"]/dailyAlpha["value"].shift().fillna(1)-1
        self.ann = dailyAlpha["value"].iloc[-1]**(250/len(dailyAlpha))-1
        self.sharpe = self.ann/(np.sqrt(250)*self.chgDay.std())
        self.drawdown = (dailyAlpha["value"]/dailyAlpha["value"].cummax()-1)
        # CAMP模型(很可能有很强的异方差和自相关问题)
        x = sm.add_constant(benchmark.values.reshape(-1, 1))
        y = self.chgDay.values
        self.capm = sm.OLS(y, x).fit()
        # 同方差检验
        import statsmodels.stats.api as sms
        from statsmodels.compat import lzip
        name = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
        test = sms.het_breuschpagan(self.capm.resid, self.capm.model.exog)
        print(lzip(name, test)) # p>0.05表示同方差
        # 自相关检验
        from statsmodels.stats.stattools import durbin_watson
        dw = durbin_watson(self.capm.resid)
        print(f"Durbin-Watson: {dw}")  # 接近 2 表示无自相关
        ax[0][0].set_title(f"年化: {100*self.ann:.2f}%, 夏普: {self.sharpe:.2f}, beta: {self.capm.params[1]:.2f}, alpha: {100*self.capm.params[0]:.2f}% ({test[1]:.2f}, {dw:.1f})\n"\
            f"每日一遇:{100*self.chgDay.quantile(0.5):.2f}%, 每周一遇:{100*self.chgDay.quantile(0.2):.2f}/{100*self.chgDay.quantile(0.8):.2f}%, 每月一遇:{100*self.chgDay.quantile(0.05):.2f}/{100*self.chgDay.quantile(0.95):.2f}%")
        ax[0][0].plot(np.log(dailyAlpha['value']), linewidth=2, c="C3")
        ax[0][0].plot(np.log((benchmark+1).cumprod()), c="C3")
        ax[0][0].vlines(self.drawdown.index[self.drawdown.argmin()], 0, np.log(dailyAlpha["value"].max()), linewidth=2, color="C3", linestyles="--")
        ax[0][0].vlines(self.chgDay.index[self.chgDay.argmin()], 0, np.log(dailyAlpha["value"].max()), color="C7", linestyles="--")
        ax[0][0].set_ylabel("对数净值")

        # 超额分析
        self.excess = self.chgDay-benchmark
        self.excess_ann = dailyAlpha["value"].iloc[-1]/(1+benchmark).prod()-1
        self.excess_sharpe = self.excess_ann/(np.sqrt(250)*self.excess.std())
        excess_cum = (1+self.excess).cumprod()
        excess_drawdown = (excess_cum/excess_cum.cummax()-1)
        ax[1][0].plot(np.log(self.excess+1).cumsum(), c="C2", linewidth=2)
        ax[1][0].set_title(f"年化: {100*self.excess_ann:.2f}%, 夏普: {self.excess_sharpe:.2f}, 胜率: {100*(self.excess>0).mean():.2f}, 绝对胜率: {100*(self.chgDay>0).mean():.2f}\n"
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
        totalPosition = self.dailyPos.groupby('date')['marketValue'].sum()/(dailyAlpha["value"]*1e8)
        maxOnePosition = self.dailyPos.groupby("date")["marketValue"].max()/(dailyAlpha["value"]*1e8)
        meanOnePosition = self.dailyPos[self.dailyPos["marketValue"]!=0].groupby("date")["marketValue"].mean()/(dailyAlpha["value"]*1e8)
        ax[1][1].set_title(f"平均仓位: {100*totalPosition.mean():.2f}%, 最大单一持仓: {100*maxOnePosition.max():.2f}%, 日平均换手: {dailyAlpha['turnover'].mean():.2f}, \n"
                        f"平均持仓只数: {int(self.dailyPos[self.dailyPos['marketValue']!=0].groupby('date')['symbol'].count().mean())}, 平均持仓仓位: {100*meanOnePosition.mean():.2f}%")
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
    # T0 日内净值变化
    def T0(self, loc=""):
        self.barChg = self.barAlpha.groupby("curTime")["chg"].mean()
        plt, fig, ax = matplot()
        ax.plot(np.hstack([self.barChg.loc[:113000000].values, np.zeros(30), self.barChg.loc[130000000:]]).cumsum())
        if loc=="":
            plt.show()
        else:
            plt.savefig(loc)
            plt.show()


