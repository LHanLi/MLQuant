


# 数据类型及格式: date:int64, curTime:int64, symbol:string, 因子:float/bool/string/int
# 因子命名格式：
# 算子：算子{参数（不同参数以$分隔）}、
# 例： MA{close$20}, corr{close$vol$20} 也可嵌套 MA{MA{close$20}$20}
# 可转债正股因子：@基础因子@
# 例： MA{@close@$20}
class Factor():
    def __init__(self, data):
        self.data = data
        checkfields = ['date', 'curTime', 'symbol']
        for i in checkfields:
            if i not in self.data.columns:
                raise ValueError("value must include %s"%i)
    # 计算/引用因子
    def cal(self, name, ifrecal=False):
        if (name not in self.data.columns)&(not ifrecal):  # 如果因子不存在或强制重新计算
            self.recursioncal(name)
        # 返回因子
        if name in self.data.columns:
            return self.data[name]
        else:
            print('failed cal ', name)
    # 定义因子 
    
    
    def optcal(self, opt, para):
        code = opt+'{'+'$'.join(para)+'}'
        # 一元
        if opt=='days':   # days{a} 距离上次a为True的天数
            dayscode = self.market[self.cal_factor(para[0])][[]].reset_index()
            dayscode['anndate'] = dayscode['date']
            alldayscode = self.market[[]].reset_index()
            days = alldayscode.merge(dayscode, on=['date', 'code'], how='left')
            days = days.set_index(['date', 'code']).groupby('code')['anndate'].ffill()
            days = days.reset_index()
            days[code] = ((days['date']-days['anndate']).dt.days+1).fillna(999)
            self.market[code] = days.set_index(['date', 'code'])[code]
        elif opt=='bars':  # 距离上次为True的交易bar数，之前没有True则为999
            mask = self.cal_factor(para[0]).astype(int)
            groups = mask.groupby('code').cumsum() # 每次为True重新标记分组
            self.market[code] = (self.cal_factor(para[0]).groupby(['code', groups]).\
                cumcount()+1).where(groups>0, 999)     # 每个分组
        elif opt=='meanCross':  # 截面平均  截面上全部code因子值相同
            meanRet = self.cal_factor(para[0]).groupby('date').mean()
            meanRet.name = 0
            self.market[code] = self.market[[]].reset_index().merge(meanRet.reset_index(), \
                on='date').set_index(['date', 'code'])[0] 
        # 二元
        elif opt in ['MA', 'EMA', 'Std', 'Skew', 'Kurt', 'Max', 'Min', 'quantile', 'Sum', 'Zscore']:
            self.market[code] = FB.my_pd.cal_ts(self.cal_factor(para[0]), opt, int(para[1]))
        elif opt=='dev':  # dev{close$20} close相对20日价偏离
            self.market[code] = -1 + self.cal_factor(para[0])/self.cal_factor('MA{%s$%s}'%(para[0], para[1]))
        elif opt=='ifhigher':
            self.market[code] = self.cal_factor(para[0])>float(para[1])
        elif opt=='iflower': 
            self.market[code] = self.cal_factor(para[0])<float(para[1])
        elif opt=='plus':
            self.market[code] = self.cal_factor(para[0])+self.cal_factor(para[1])
        elif opt=='minus':
            self.market[code] = self.cal_factor(para[0])-self.cal_factor(para[1])
        elif opt=='times':
            self.market[code] = self.cal_factor(para[0])*self.cal_factor(para[1])
        elif opt=='divide':
            self.market[code] = self.cal_factor(para[0])/self.cal_factor(para[1])
        # 三元
        elif opt in ['corr', 'slope', 'intercept', 'epsilon']: # 一元线性回归
            MAx = FB.my_pd.cal_ts(self.cal_factor(para[0]), 'MA', int(para[2]))
            deltax = self.cal_factor(para[0])-MAx
            MAy = FB.my_pd.cal_ts(self.cal_factor(para[1]), 'MA', int(para[2])) 
            deltay = self.cal_factor(para[1])-MAy
            if opt=='corr':
                self.market[code] = (FB.my_pd.cal_ts((deltax*deltay), 'Sum', int(para[2]))/\
                    (np.sqrt(FB.my_pd.cal_ts(deltax**2, 'Sum', int(para[2])))*\
                    np.sqrt(FB.my_pd.cal_ts(deltay**2, 'Sum', int(para[2]))))).fillna(0)
            elif opt=='slope':
                self.market[code] = FB.my_pd.cal_ts(deltax*deltay, 'Sum', \
                        int(para[2]))/FB.my_pd.cal_ts(deltax**2, 'Sum', int(para[2]))
            elif opt=='intercept':
                self.market[code] = MAy-(FB.my_pd.cal_ts(deltax*deltay, 'Sum', \
                        int(para[2]))/FB.my_pd.cal_ts(deltax**2, 'Sum', int(para[2])))*MAx
            elif opt=='epsilon':  # epsilon = y - (slope*x+intercept)
                self.market[code] = self.cal_factor(para[1])-\
                    ((FB.my_pd.cal_ts(deltax*deltay, 'Sum', \
                        int(para[2]))/FB.my_pd.cal_ts(deltax**2, 'Sum', int(para[2])))*self.cal_factor(para[0])\
                    +(MAy-(FB.my_pd.cal_ts(deltax*deltay, 'Sum', \
                        int(para[2]))/FB.my_pd.cal_ts(deltax**2, 'Sum', int(para[2])))*MAx))            
    # 从code得到他的算子以及参数列表
    def split(self, code):
        # 从opt{para}得到opt和para
        def split_optpara(code):
            find = re.findall(r'^(\w+)\{(.*?)\}$', code) # 用{}分离算子和参数
            if (len(find)!=1)|('{' in find)|('}' in find):
                print('!!!error!!! factor %s incorrect format'%code)
                return
            return find[0] # opt, para
        # 使用$分割para中的参数，同时忽略{}中包含的$，得到参数列表
        def split_para(para):
            result = []
            current_segment = []
            level = 0  # 跟踪大括号层级
            for char in para:
                if char == '{':
                    level += 1
                elif char == '}':
                    level = max(level - 1, 0)
                elif char == '$' and level == 0:
                    # 分割点：不在任何{}内
                    result.append(''.join(current_segment))
                    current_segment = []
                    continue
                current_segment.append(char)
            if current_segment:
                result.append(''.join(current_segment))
            return result
        opt,para = split_optpara(code) # 分隔算子和参数
        para = split_para(para) # 分隔参数列表
        return opt, para 
    # 递归计算基础因子
    def recursioncal(self, code):
        if not bool(re.compile(r'[{}]').search(code)):
            self.cal_basic_factor(code)
            return code
        else:
            opt,para = self.split(code) # 分隔算子和参数
            num = self.parafactornum[opt]
            parafactor, paraothers = para[:num], para[num:]
            for factor in parafactor:
                self.recursioncal(factor)
            return self.optcal(opt, para)
    # 递归给因子添加@@装饰符
    def decorate(self, code):
        if not bool(re.compile(r'[{}]').search(code)):
            return '@' + code + '@'
        else:
            opt,para = self.split(code) # 分隔算子和参数
            num = self.parafactornum[opt]
            parafactor, paraothers = para[:num], para[num:]
            return opt + '{' + '$'.join([self.decorate(i) for i in parafactor]+paraothers) + '}'
    # 向market中加入正股因子（或命名重叠） df中全为要添加元素字段
    def add_stk(self, dfSTK):  
        dfSTK = dfSTK[[i for i in dfSTK.columns \
            if self.decorate(i) not in self.market.columns]] # 只添加原有market中没有的数据列
        cut = self.market[['STKcode']].reset_index().merge(dfSTK.reset_index().\
                rename(columns={'code':'STKcode'}), on=['STKcode', 'date']).\
                set_index(['date', 'code']).drop(columns='STKcode') # 提取对应正股代码数据
        for i in cut.columns:
            if self.decorate(i) not in self.market.columns:
                self.market[self.decorate(i)] = cut[i]
    # 计算/引用基础因子
    def cal_basic_factor(self, code):
        # 因子分解为因子名和参数
        code_split = code.split('_')
        key = code_split[0]
        para = code_split[1:]
        ##############################################################################
        ########################### 交易规则类 rules ##################################
        ##############################################################################
        if key=='PriceLimit':  # 涨跌停
            def get_PriceLimit():
                # 使用长度为3的字符串表示涨跌停状态 
                # 开盘收盘：+表示涨停 -表示跌停 0表示正常
                # 盘中: 0盘中时表示非一直跌停或涨停，+或-表示存在涨停或跌停
                date = self.market.index.get_level_values(0)
                #global get_limit
                if self.type=='STK':
                    #mindelta = 0.01
                    #def get_limit(up=True):  # 获取涨跌幅度限制    简化处理，详细规则见:xueqiu.com/4610950050/276325750
                    #    return np.select([self.cal_factor('place_sector')=='上交所.科创板', self.cal_factor('place_sector')=='深交所.创业板', self.cal_factor('place_sector')=='北交所.北证'],\
                    #            [np.where(self.cal_factor('list_bars')<=5, 999 if up else 1, 0.2),\
                    #            np.where(date>=pd.to_datetime('2020-8-24'), np.where(self.cal_factor('list_bars')<=5, 999 if up else 1, 0.2),\
                    #                np.where(self.cal_factor('special_type').isin(['*ST', 'ST', 'delist']), 0.05, \
                    #                    np.where(date>=pd.to_datetime('2014-1-1'), np.where(self.cal_factor('list_bars')==1, 0.44 if up else 0.36, 0.1),\
                    #                        np.where(self.cal_factor('list_bars')==1, 999 if up else 1, 0.1)))),\
                    #            np.where(self.cal_factor('list_bars')==1, 999 if up else 1, 0.3)], \
                    #            np.where(self.cal_factor('special_type').isin(['*ST', 'ST', 'delist']), 0.05, \
                    #                np.where(date>=pd.to_datetime('2023-4-10'), np.where(self.cal_factor('list_bars')<=5, 999 if up else 1, 0.1),\
                    #                np.where(date>=pd.to_datetime('2014-1-1'), np.where(self.cal_factor('list_bars')<=5, 0.44 if up else 0.36, 0.1),\
                    #                    np.where(self.cal_factor('list_bars')==1, 999 if up else 1, 0.1)))))
                    uplimit = self.cal_factor('highLimit')
                    downlimit = self.cal_factor('lowLimit') 
                elif self.type=='CB':
                    mindelta = 1000
                    def get_limit(up=True):  # 获取涨跌幅度限制    简化处理，详细规则见:xueqiu.com/4610950050/276325750
                        return np.where(date<=pd.to_datetime('2022-8-1'), 999 if up else 1,\
                                np.where(self.cal_factor('listdays')==0, 0.3, 0.2))  # 上市首日仅考虑14:57前限制幅度即30%
                    uplimit = (0.5+self.cal_factor('preClose')*(1+get_limit())*mindelta).astype('int')/mindelta
                    downlimit = (0.5+self.cal_factor('preClose')*(1-get_limit(False))*mindelta).astype('int')/mindelta
                #uplimit = self.cal_factor('preClose')*(1+get_limit()) \
                #    if 'highLimit' not in self.market.columns else \
                #        pd.Series(np.where(self.market['vol']==0, 9e9,self.market['highLimit']), index=self.market.index) 
                #downlimit = self.cal_factor('preClose')*(1-get_limit(False)) \
                #    if 'lowLimit' not in self.market.columns else \
                #        pd.Series(np.where(self.market['vol']==0, 0,self.market['lowLimit']), index=self.market.index) 
                ifonceuplimit = self.cal_factor('high')==uplimit
                ifoncedownlimit = self.cal_factor('low')==downlimit
                ifalwaysuplimit = self.cal_factor('low')==uplimit
                ifalwaysdownlimit = self.cal_factor('high')==downlimit
                ifcloseuplimit = self.cal_factor('close')==uplimit
                ifclosedownlimit = self.cal_factor('close')==downlimit
                ifopenuplimit = self.cal_factor('open')==uplimit
                ifopendownlimit = self.cal_factor('open')==downlimit
                return uplimit, downlimit, pd.Series(np.where(ifopenuplimit, '+', np.where(ifopendownlimit, '-', 0)), index=self.market.index) +\
                    pd.Series(np.where((((~ifopendownlimit)&(~ifclosedownlimit)&ifoncedownlimit)|ifalwaysdownlimit), '-', \
                        np.where((((~ifopenuplimit)&(~ifcloseuplimit)&ifonceuplimit)|ifalwaysuplimit), '+', 0)), index=self.market.index) +\
                        pd.Series(np.where(ifcloseuplimit, '+', np.where(ifclosedownlimit, '-', 0)), index=self.market.index) # 开盘+盘中+收盘
            uplimit, downlimit, PriceLimit = get_PriceLimit()
            self.market['highLimit'] = uplimit
            self.market['lowLimit'] = downlimit
            self.market[code] = PriceLimit.astype('string')
        elif key=='listdays':  # list_days 上市天数  list_bars 上市交易日数
            #if para[0]=='days':
            self.market[code] =   (pd.Series(self.market.index.get_level_values(0), self.market.index)-\
                                pd.to_datetime(self.market['listDate'], format="%Y%m%d")).dt.days
            #elif para[0]=='bars':
            #    self.market['temp'] = self.cal_factor('list_days')==0  # 
            #    #index0 = pd.Series(self.market.index.get_level_values(0), self.market.index)  # 直接计算上市日，可能更快  
            #    #self.market['temp'] = self.market['listDate'] == index0.dt*10000 + index0.dt.month*100 + index0.dt.day
            #    self.market[code] = self.cal_factor('bars{temp}')
            self.market[code] = self.market[code].astype('int32')
        ##############################################################################
        ####################### 公告（bool） announcements ############################
        ##############################################################################
        #elif key=='special_type':
        #    def get_special_type(name):
        #        if '*ST' in name:
        #            return '*ST'
        #        elif 'ST' in name:
        #            return 'ST'
        #        elif '退' in name:
        #            return 'delist'
        #        else:
        #            return 'Normal'
        #    self.market[code] = self.cal_factor('name').map(lambda x: get_special_type(x))
        #elif key=='status':   # status.a.d  某类型状态，重置时间周期
        #    if self.type=='bond':
        #        if para[0]=='下修':   # 下修公告前20日均价指的是20日的VWAP
        #            self.market[code] = pd.Series(np.select([(self.cal_factor('提示下修-tradedays')<self.cal_factor('提议下修-tradedays'))&\
        #                                (self.cal_factor('提示下修-tradedays')<self.cal_factor('下修-tradedays'))&\
        #                                (self.cal_factor('提示下修-tradedays')<self.cal_factor('不下修-tradedays'))&(self.cal_factor('提示下修-tradedays')<int(para[1])),\
        #                                (self.cal_factor('提议下修-tradedays')<self.cal_factor('提示下修-tradedays'))&\
        #                                (self.cal_factor('提议下修-tradedays')<self.cal_factor('下修-tradedays'))&\
        #                                (self.cal_factor('提议下修-tradedays')<self.cal_factor('不下修-tradedays'))&(self.cal_factor('提议下修-tradedays')<int(para[1])),\
        #                                (self.cal_factor('下修-tradedays')<self.cal_factor('提示下修-tradedays'))&\
        #                                (self.cal_factor('下修-tradedays')<self.cal_factor('提议下修-tradedays'))&\
        #                                (self.cal_factor('下修-tradedays')<self.cal_factor('不下修-tradedays'))&(self.cal_factor('下修-tradedays')<int(para[1])),\
        #                                (self.cal_factor('不下修-tradedays')<self.cal_factor('提示下修-tradedays'))&\
        #                                (self.cal_factor('不下修-tradedays')<self.cal_factor('提议下修-tradedays'))&\
        #                                (self.cal_factor('不下修-tradedays')<self.cal_factor('下修-tradedays'))&(self.cal_factor('不下修-tradedays')<int(para[1]))], \
        #                                ['提示下修', '提议下修', '下修', '不下修']), index=self.market.index).replace('0', np.nan).\
        #                                    astype('object').groupby('code').fillna('空')
        #        elif para[0]=='强赎':
        #            self.market[code] = pd.Series(np.select([(self.cal_factor('提示强赎-tradedays')<self.cal_factor('强赎-tradedays'))&\
        #                                (self.cal_factor('提示强赎-tradedays')<self.cal_factor('不强赎-tradedays'))&(self.cal_factor('提示强赎-tradedays')<int(para[1])),\
        #                                (self.cal_factor('强赎-tradedays')<self.cal_factor('提示强赎-tradedays'))&\
        #                                (self.cal_factor('强赎-tradedays')<self.cal_factor('不强赎-tradedays'))&(self.cal_factor('强赎-tradedays')<int(para[1])),\
        #                                (self.cal_factor('不强赎-tradedays')<self.cal_factor('提示强赎-tradedays'))&\
        #                                (self.cal_factor('不强赎-tradedays')<self.cal_factor('强赎-tradedays'))&(self.cal_factor('不强赎-tradedays')<int(para[1]))], \
        #                                ['提示强赎', '强赎', '不强赎']), index=self.market.index).replace('0', np.nan).\
        #                                    astype('object').groupby('code').fillna('空')
        ###############################################################################
        ################################ 市值 Cap ##################################
        ##############################################################################
        elif key=='Cap':
            if self.type=='STK':
                #self.market[code] = self.cal_factor('totalShares')*self.cal_factor('close')
                self.market[code] = self.cal_factor('times{totalShares$close}')
            elif self.type=='CB':
                self.market[code] = (self.cal_factor('raiseFund')*(1-self.cal_factor('CVRate')))\
                    *self.cal_factor('close')/100
        elif key=='balance': # 剩余规模
            self.market[code] = self.cal_factor('raiseFund')*(1-self.cal_factor('CVRate'))
        #elif key=='freeCap':
        #    #self.market[code] = self.cal_factor('freeShares')*self.cal_factor('close')
        #    self.market[code] = self.cal_factor('times{freeShares$close}')
        #elif key=='freeCapratio':
        #    self.market[code] = self.cal_factor('freeCap')/self.cal_factor('Cap')
        #elif key=='CBCapratio':
        #    self.market[code] = self.cal_factor('Cap')/self.cal_factor('@freeCap@')
        ##############################################################################
        ####################### 财务数据 finance 股票专属 #############################
        ##############################################################################
        #elif key=='现金':
        #    self.market['现金'] = self.cal_factor('currency')/1e8
        #elif key=='总资产':
        #    self.market['总资产'] = self.cal_factor('asset')/1e8
        #elif key=='BP':
        #    self.market['BP'] = self.cal_factor('equity')/(self.cal_factor('close')*\
        #        self.cal_factor('total_shares'))  # 市净率倒数
        #elif key=='EP':
        #    self.market['EP'] = self.cal_factor('profit')/(self.cal_factor('close')*self.cal_factor('total_shares'))  # 市盈率倒数 ttm
        #elif key=='SP':
        #    self.market['SP'] = self.cal_factor('revenue')/(self.cal_factor('close')*self.cal_factor('total_shares'))  # 市销率倒数 ttm
        #elif key=='ROE':
        #    self.market['ROE'] = self.cal_factor('profit-ttm')/self.cal_factor('equity')
        #elif key=='ROA':
        #    self.market['ROA'] = self.cal_factor('profit-ttm')/self.cal_factor('asset')
        #elif key=='资产负债率':
        #    self.market['资产负债率'] = self.cal_factor('borrow')/self.cal_factor('asset')  # 资产负债率
        #elif key=='有形资产负债率':
        #    self.market['有形资产负债率'] = self.cal_factor('borrow')/\
        #        (self.cal_factor('asset')-self.cal_factor('intangible'))  # 资产负债率
        #elif key=='股息率':
        #    self.market['股息率'] = self.cal_factor('dividends_ttm')/(self.cal_factor('close')*self.cal_factor('total_shares')) # 股息率
        #elif key=='流动比率':
        #    self.market['流动比率'] = self.cal_factor('current_asset')/self.cal_factor('current_borrow')
        #elif key=='速动比率':
        #    self.market['速动比率'] = (self.cal_factor('current_asset')-self.cal_factor('inventory'))/self.cal_factor('current_borrow')
        #elif key=='存货周转天数':
        #    self.market['存货周转天数'] = 360*self.cal_factor('inventory-ttm')/self.cal_factor('revenue-ttm')
        #elif key=='应收账款周转天数':
        #    self.market['应收账款周转天数'] = 360*self.cal_factor('receivable-ttm')/self.cal_factor('revenue-ttm')
        #elif key=='营业周期':
        #    self.market['营业周期'] = self.cal_factor('存货周转天数') + self.cal_factor('应收账款周转天数')
        #elif key=='流动资产周转率':
        #    self.market['流动资产周转率'] = self.cal_factor('revenue-ttm')/self.cal_factor('current_asset-ttm')
        #elif key=='总资产周转率':
        #    self.market['总资产周转率'] = self.cal_factor('revenue-ttm')/self.cal_factor('asset-ttm')
        #elif key=='已获利息倍数':
        #    self.market['已获利息倍数'] = self.cal_factor('profit-ttm')/self.cal_factor('dividends_ttm')
        #elif key=='净利率':
        #    self.market['净利率'] = self.cal_factor('profit-ttm')/self.cal_factor('revenue-ttm')
        #elif key=='毛利率':
        #    self.market['毛利率'] = (self.cal_factor('revenue-ttm') - self.cal_factor('operating_cost-ttm'))/self.cal_factor('revenue-ttm')
        #elif key=='现金到期债务比':
        #    self.market['现金到期债务比'] = (self.cal_factor('OAcashin-ttm')-self.cal_factor('OAcashout-ttm'))/\
        #        (self.cal_factor('st_borrow')+self.cal_factor('1y_borrow'))
        #elif key=='现金流动负债比':
        #    self.market['现金流动负债比'] = (self.cal_factor('OAcashin-ttm')-self.cal_factor('OAcashout-ttm'))/(self.cal_factor('current_borrow'))
        #elif key=='现金负债比':
        #    self.market['现金负债比'] = (self.cal_factor('OAcashin-ttm')-self.cal_factor('OAcashout-ttm'))/(self.cal_factor('borrow'))
        #elif key=='现金销售比':
        #    self.market['现金销售比'] = (self.cal_factor('OAcashin-ttm')-self.cal_factor('OAcashout-ttm'))/(self.cal_factor('revenue'))
        #elif key=='全部资产现金回收率':
        #    self.market['全部资产现金回收率'] = (self.cal_factor('OAcashin-ttm')-self.cal_factor('OAcashout-ttm'))/(self.cal_factor('asset'))
        #elif key=='营收同比':
        #    self.market['营收同比'] = self.cal_factor('revenue-YOY')
        #elif key=='净利润同比':
        #    self.market['净利润同比'] = self.cal_factor('profit-YOY')
        ##############################################################################
        ######################### 纯债数据 bond 债券专属 ##############################
        ##############################################################################
        #elif key=='ncredit':
        #    if self.type=='bond':
        #        replace_dict = {'AAA': 0, 'AA+': 1, 'AA+u':1, 'AA': 2, 'AA-': 3, 'A+': 4, 'A+k':4, 'A': 5, 'A-': 6,\
        #            'BBB+': 7, 'BBB': 8, 'BBB-': 9, 'BB+': 10, 'BB': 11, 'BB-': 12,\
        #                'B+': 13, 'B': 14, 'B-': 15, 'CCC': 16, 'CC': 17, 'C':18}
        #        credit = self.market['credit'].fillna(self.market['credit'].mode().iloc[0]) # 众数填充
        #        self.market['ncredit'] = credit.replace(replace_dict)
        #elif key=='bond_prem':
        #    self.market[code] = self.cal_factor('close')/self.cal_factor('pure_bond')-1
        #elif key=='init_balance':
        #    temp = self.market[[]].reset_index().merge(self.issurance[['code', 'total_amount']]\
        #                            , on='code').set_index(['date', 'code'])
        #    self.market['init_balance'] = temp['total_amount']/1e8
        #elif key=='remain_ratio':
        #    self.market[code] = self.cal_factor('balance')/self.cal_factor('init_balance')
        #elif key=='balance_cash':
        #    self.market[code] = self.cal_factor('balance')/self.cal_factor('@现金@')
        #elif key=='balance_asset':
        #    self.market[code] = self.cal_factor('balance')/self.cal_factor('@总资产@')
        #elif key=='hold_money':        # 持有到期获得现金（不包含中间利息
        #    def get_return_money(string):
        #        try:
        #            n = int(re.findall(r"[0-9]*\.?[0-9]+", string)[-1])
        #            if n>100:
        #                return n
        #            else:
        #                return 100+n
        #        except:
        #            return 100
        #    self.issurance['hold_money'] = self.issurance['compen_interest'].map(\
        #        lambda x: get_return_money(x))
        #    temp = self.market[[]].reset_index().merge(self.issurance[['code', 'hold_money']],\
        #                on='code').set_index(['date', 'code'])
        #    self.market[code] = temp['hold_money']
        #elif key=='hold_ratio':  # 持有到期税前收益
        #    self.market[code] = self.cal_factor('hold_money')/self.cal_factor('close')-1
        #elif key=='hold_rate': # 持有到期收益率
        #    self.market[code] = (1+self.cal_factor('hold_ratio'))**\
        #                                    (365/self.cal_factor('last_days'))-1
        ##############################################################################
        ######################### 期权数据 option 转债专属 ##############################
        ##############################################################################
        elif key=='CVValue':  # 转股价值
             self.market[code] = self.cal_factor('@close@')*100/self.cal_factor('CVPrice')
        elif key=='CVprem':   # 转股溢价率
            self.market[code] = self.cal_factor('close')/self.cal_factor('CVValue')-1
        elif key=='correctCVprem': # 修正转股溢价率 截面上转股溢价率对转股价值幂函数回归的残差
            def fitfunc(x, a, b):
                return a*(x**b)
            result = []
            for d,g in self.market.groupby(level='date'):
                x = g['Pc']
                y = g['CVprem']
                params, _ = curve_fit(fitfunc, x, y, p0=[1, 1])  # 初始值设为a=1, b=1
                a, b = params
                y_pred = fitfunc(x, a, b)
                residuals = y - y_pred   # 计算预测值和残差
                result.append(residuals)
            self.market[code] = pd.concat(result)
        elif key=='dblow':    # dblow_a  双低值 以a为参数 双低值中a一般为100 
            self.market[code] = self.cal_factor('CVprem')*int(para[0])+self.cal_factor('close')
        #elif key=='PcvB':    # 纯债转股溢价率
        #    self.market[code] = (self.cal_factor('Pc')-self.cal_factor('pure_bond'))/\
        #                                                self.cal_factor('pure_bond')
        #elif key in ['call', 'put']:   # call.d 使用d日波动率
        #    def BSM(S, K, T, sigma, r=0, option='call'):
        #        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        #        d2 = (np.log(S/K) + (r - 0.5*sigma**2)*T)/(sigma * np.sqrt(T))
        #        if option == 'call':
        #            p = (S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2))
        #        elif option == 'put':
        #            p = (K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1))
        #        return p
        #    self.market[code] = BSM(self.cal_factor('Pc'), self.cal_factor('hold_money'),\
        #        self.cal_factor('last_tradedays'), self.cal_factor('@Ret@-Std-%s'%para[0]),\
        #            option=key)
        #elif key=='delta':   # delta.call.d  # call
        #    def Delta(S, K, T, sigma, r=0, option='call'):
        #        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        #        if option == 'call':
        #            delta = norm.cdf(d1)
        #        elif option == 'put':
        #            delta = norm.cdf(d1) - 1
        #        return delta
        #    self.market[code] = Delta(self.cal_factor('Pc'), self.cal_factor('hold_money'),\
        #        self.cal_factor('last_tradedays'), self.cal_factor('@Ret@-Std-%s'%para[1]),\
        #            option=para[0])
        #elif key=='Gamma':   # Gamma.d
        #    def Gamma(S, K, T, sigma, r=0):
        #        # 利率全部统一到交易日
        #        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        #        gamma = (np.exp(-d1**2/2)/(np.sqrt(2*math.pi)))/(S*sigma*np.sqrt(T))
        #        return gamma
        #    self.market[code] = Gamma(self.cal_factor('Pc'), self.cal_factor('hold_money'),\
        #        self.cal_factor('last_tradedays'), self.cal_factor('@Ret@-Std-%s'%para[0]))
        #elif key=='Theta':    # Theta.call.d
        #    def Theta(S, K, T, sigma, r=0, option='call'):
        #        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        #        d2 = (np.log(S/K) + (r - 0.5*sigma**2)*T)/(sigma * np.sqrt(T))

        #        if option == 'call':
        #            theta = -S*sigma*(np.exp(-d1**2/2)/(np.sqrt(2*math.pi)))/(2*np.sqrt(T)) \
        #                - r*K*np.exp(-r*T)*norm.cdf(d2)
        #        elif option == 'put':
        #            theta = -S*sigma*(np.exp(-d1**2/2)/(np.sqrt(2*math.pi)))/(2*np.sqrt(T)) \
        #                + r*K*np.exp(-r*T)*norm.cdf(-d2)
        #        return theta
        #    self.market[code] = Theta(self.cal_factor('Pc'), self.cal_factor('hold_money'),\
        #        self.cal_factor('last_tradedays'), self.cal_factor('@Ret@-Std-%s'%para[1]),\
        #            option=para[0])
        #elif key=='Vega':   # Vega.d
        #    def Vega(S, K, T, sigma, r=0):
        #        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        #        vega = S*np.sqrt(T)*np.exp(-d1**2/2)/np.sqrt(2*math.pi)  
        #        return vega
        #    self.market[code] = Vega(self.cal_factor('Pc'), self.cal_factor('hold_money'),\
        #        self.cal_factor('last_tradedays'), self.cal_factor('@Ret@-Std-%s'%para[0]))
        #elif key=='IV':    # IV.call
        #    # 隐含波动率（可以为负值）,  P 期权价格
        #    def BSM(S, K, T, sigma, r=0, option='call'):
        #        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        #        d2 = (np.log(S/K) + (r - 0.5*sigma**2)*T)/(sigma * np.sqrt(T))
        #        if option == 'call':
        #            p = (S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2))
        #        elif option == 'put':
        #            p = (K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1))
        #        return p
        #    def IV(P,S,K,T,r=0, option='call'):      #从0.001 - 1.000进行二分查找
        #        sigma_min = 1e-3           # 设定波动率初始最小值
        #        sigma_max = 3           # 设定波动率初始最大值(能计算出值的最大基本在2.7，之后就往无限大走了)
        #    #    sigma_mid = (sigma_min + sigma_max) / 2
        #        V_min = BSM(S, K, T, sigma_min, r, option)
        #        V_max = BSM(S, K, T, sigma_max, r, option)
        #    #    V_mid = BSM(S,K,sigma_mid,r,T, option)
        #        if P < V_min:
        #            #print('IV less than sigma_min')
        #            return sigma_min                            # 隐波记为0
        #        elif P > V_max:
        #            #print('IV big than sigma_max')
        #            return sigma_max
        #        # 波动率差值收敛到0.01%为止
        #        diff = sigma_max - sigma_min
        #        while abs(diff) > 0.0001:
        #            sigma_mid = (sigma_min + sigma_max) / 2
        #            V_mid = BSM(S, K, T, sigma_mid, r, option)
        #            # V_mid小于价格，说明sigma_mid小于隐含波动率
        #            if P > V_mid:
        #                sigma_min = sigma_mid
        #            else:
        #                sigma_max = sigma_mid
        #            diff = sigma_max - sigma_min
        #        return sigma_mid
        #    self.cal_factor('hold_money')
        #    self.market[code] = self.market.apply(lambda x: IV(x['close']-x['hold_money'],\
        #        x['Pc'], x['hold_money'], x['last_tradedays'], option=para[0]), axis=1)
        #elif key=='theory_prem':  # 理论溢价率  theory_prem.call.d
        #    if para[0]=='call':
        #        self.market[code] = self.cal_factor('close')/\
        #            (self.cal_factor('pure_bond')+self.cal_factor('call.'+para[1]))-1 
        #    elif para[0]=='put':
        #        self.market[code] = self.cal_factor('close')/\
        #            (self.cal_factor('Pc')+self.cal_factor('put.'+para[1]))-1 
        
        #==============================================================================# 
        #============================= 量价因子 =========================================# 
        #==============================================================================# 
        
        ##############################################################################
        ################################ 价格因子 price ##############################
        ##############################################################################
        elif key=='vwap':    # vwap_n 默认1日 n日vawp，注意n日vwap和MA{vwap.n}不同
            if para==[]:
                self.market[code] = (self.cal_factor('amount')/self.cal_factor('vol')).\
                    fillna(self.market['close'])
            else:
                self.market[code] = (self.cal_factor('Sum{amount$%s}'%para[0])/\
                                        self.cal_factor('Sum{vol$%s}'%para[0])).\
                                            fillna(self.market['close'])
        elif key=='exfactor':    # 复权因子
            exfactor = (self.cal_factor('tradeClose').groupby('code').shift()/\
                self.cal_factor('preClose')).fillna(1)
            self.market[code] = exfactor.groupby('code').cumprod()
        elif key in ['exclose', 'exopen', 'exhigh', 'exlow']:
            self.market[code] = self.cal_factor(code[2:])*self.cal_factor('exfactor')
        ##############################################################################
        ################################ k线形态 kbar ################################
        ##############################################################################
        elif key=='Ret':      
            self.market[code] = self.cal_factor('close')/self.cal_factor('preClose')-1
        elif key=='correctRet':      # 修正收益率（上市首日修正为0）
            correctRet = self.cal_factor('Ret').copy()
            correctRet[self.cal_factor('listdays')==0] = 0
            self.market[code] = correctRet
        elif key=='highRet':     
            self.market[code] = self.cal_factor('high')/self.cal_factor('preClose')-1
        elif key=='lowRet':     
            self.market[code] = self.cal_factor('low')/self.cal_factor('preClose')-1
        elif key=='nightRet':     # 隔夜收益（开盘跳空
            self.market[code] = self.cal_factor('open')/self.cal_factor('preClose')-1
        elif key=='T0Ret':
            self.market['T0Ret'] = self.cal_factor('close')/self.cal_factor('open')-1
        elif key=='maxRet':
            self.market['maxRet'] = self.cal_factor('high')/self.cal_factor('low')-1
        elif key=='bodyRatio':   # k线实体占比
            self.market[code] = abs(self.cal_factor('close')-self.cal_factor('open'))/\
                (self.cal_factor('high')-self.cal_factor('low'))
        elif key=='oc2hl':   # 开收重心到低高重心变动
            self.market[code] = (self.cal_factor('high')+self.cal_factor('low'))/\
                (self.cal_factor('open')+self.cal_factor('close')) 
        elif key=='MACD':   #  MACD_12_26_9 
            EMAslow = FB.my_pd.cal_ts(self.cal_factor('exclose'), 'EMA', int(para[1])) 
            EMAfast = FB.my_pd.cal_ts(self.cal_factor('exclose'), 'EMA', int(para[0]))
            DIF = EMAfast-EMAslow
            DEM = FB.my_pd.cal_ts(DIF, 'EMA', int(para[2]))
            self.market[code] = DIF-DEM
        elif key=='RSI':    # RSI_d  d日RSI 
            up = pd.Series(np.where(self.cal_factor('Ret')>0, self.cal_factor('Ret'), 0), \
                           index=self.market.index)
            down = pd.Series(np.where(self.cal_factor('Ret')<0, -self.cal_factor('Ret'), 0), \
                             index=self.market.index)
            sumup = FB.my_pd.cal_ts(up, 'Sum', int(para[0]))
            sumdown = FB.my_pd.cal_ts(down, 'Sum', int(para[0]))
            rsi = sumup/(sumup+sumdown)
            self.market[code] = rsi
        #elif key=='RSRS':   slope{exlow.exhigh.d}   # high/low 回归斜率
        ##############################################################################
        ########h##################### 动量反转 mom ##################################
        ##############################################################################
        elif key=='DrawDown':        # 唐奇安通道 上轨距离   DrawDown_d
            max_c = FB.my_pd.cal_ts(self.cal_factor('exhigh'), 'Max', int(para[0]))
            self.market[code] = 1-self.cal_factor('exclose')/max_c
        elif key=='DrawUp':   # 唐奇安通道 下轨距离  DrawUp_d
            min_c = FB.my_pd.cal_ts(self.cal_factor('exlow'), 'Min', int(para[0]))
            self.market[code] = self.cal_factor('exclose')/min_c-1 
        ##############################################################################
        ########################### 波动率 volatility ################################
        ##############################################################################
        elif key=='AMP':   # 振幅   AMP_d
            if para==[]:
                exhigh_Max = FB.my_pd.cal_ts(self.cal_factor('exhigh'), 'Max', 1)
                exlow_Min = FB.my_pd.cal_ts(self.cal_factor('exlow'), 'Min', 1)
            else:
                exhigh_Max = FB.my_pd.cal_ts(self.cal_factor('exhigh'), 'Max', int(para[0]))
                exlow_Min = FB.my_pd.cal_ts(self.cal_factor('exlow'), 'Min', int(para[0]))
            self.market[code] = (exhigh_Max - exlow_Min)/exlow_Min
        elif key=='TR':    # 真实波动
            self.market[code] = pd.concat([self.cal_factor('high')-self.cal_factor('low'),\
                abs(self.cal_factor('preClose')-self.cal_factor('high')), \
                abs(self.cal_factor('preClose')-self.cal_factor('low'))], axis=1).max(axis=1) 
        #elif key=='ifUp': # ifUp.a
        #    self.market[code] = self.cal_factor('highRet')>(float(para[0])/100)
        #elif key=='ifDown':
        #    self.market[code] = self.cal_factor('lowRet')<(float(para[0])/100)
        #elif key=='ifAMP':
        #    self.market[code] = self.cal_factor('AMP')>(float(para[0])/100)
        #elif key=='ifLimit':
        #    self.market[code] = self.cal_factor('PriceLimit').map(lambda x: ('+' in x)|('-' in x))
        elif key=='everLimit':
            self.market[code] = self.cal_factor('PriceLimit').map(lambda x: x!='000')
        elif key=='everhighLimit':
            self.market[code] = self.cal_factor('PriceLimit').map(lambda x: ('-' in x))
        elif key=='everlowLimit':
            self.market[code] = self.cal_factor('PriceLimit').map(lambda x: ('+' in x))
        #elif key=='UpTimes':      # UpTimes.a.d 过去d日涨幅超过a%天数   ifUp.a
        #    self.market[code] = FB.my_pd.cal_ts((self.cal_factor('highRet')>float(para[0])/100),\
        #                                'Sum', int(para[1]))
        #elif key=='DownTimes':      # *.a.d 过去d日最大跌幅超过a%天数
        #    self.market[code] = FB.my_pd.cal_ts((self.cal_factor('lowRet')<float(para[0])/100),\
        #                                'Sum', int(para[1]))
        #elif key=='AMPTimes':      # *.a.d 过去d日最大波动超过a%天数
        #    self.market[code] = FB.my_pd.cal_ts(self.cal_factor('maxRet')>float(para[0]),\
        #                                'Sum', int(para[1]))
        #elif key=='LimitTimes':      # LimitTimes.d d日涨跌停天数
        #    self.market[code] = FB.my_pd.cal_ts(self.cal_factor('PriceLimit').\
        #            map(lambda x: ('+' in x)|('-' in x)), 'Sum', int(para[0]))
        ##############################################################################
        ######################## 交易活跃度、流动性 volhot #############################
        ##############################################################################
        elif key=='turnover':               # turnover_d 
            if self.type=='STK':
                if para==[]:
                    self.market[code] = self.cal_factor('vol')/self.cal_factor('freeShares')
                else:
                    self.market[code] = self.cal_factor('Sum{vol$%s}'%para[0])/self.cal_factor('freeShares')
            elif self.type=='CB':
                if para==[]:
                    self.market[code] = self.cal_factor('vol')/(self.cal_factor('balance')/100)
                else:
                    self.market[code] = self.cal_factor('Sum{vol$%s}'%para[0])/(self.cal_factor('balance')/100)
        elif key=='Amihud':  # 流动性  Amihud_d
            if para==[]:
                self.market[code] = abs(self.cal_factor('Ret'))/self.cal_factor('amount')
            else:
                self.market[code] = abs(self.cal_factor('Sum{Ret$%s}'%para[0]))/self.cal_factor('Sum{amount$%s}'%para[0])
        ##############################################################################
        ################################ 相关性 corr ################################
        ##############################################################################
        # 量价相关性  corr{vol$vwap$20}
        ##############################################################################
        ################################# 截面 cross #################################
        ##############################################################################
        # 超额收益率 minus{Ret$meanCross{Ret}}  
        ##############################################################################
        ############################# 经济学理论 model ################################
        ##############################################################################
        # CAMP理论 beta slope{meanCross{Ret}$Ret$20}
        ##############################################################################
        ############################# 创新因子 fancy ################################
        ##############################################################################
        if code not in self.market.columns:
            print('warning! not basic factor %s'%code)


 