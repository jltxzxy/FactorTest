from FactorTest.FactorTestPara import *
from FactorTest.FactorTestBox import *


class FactorTest():
    def __init__(self):
        self.startdate=20000101
        self.enddate=21000101
        self.factorlist=[]
        self.FactorDataBase={'v':pd.DataFrame(columns=['time','code'])}
        self.filterStockDF='FULL'
        self.retData = getRetData() #统一为time、code  time为int  code 为str 
        self.ICList={}
        self.portfolioList={}
        self.ICAns={}
        self.portfolioAns={}
        self.portfolioGroup = pd.DataFrame(columns=['time', 'code'])
        self.annualTurnover = {}
        self.year_performance={}
        self.WR={}
        self.PL={}
        pd.options.mode.use_inf_as_na = True  #剔除inf
        self.dataProcess=dataProcess(self.FactorDataBase)

    def getFactor(self,Factor):
        #考虑：频率统一为月度 ,sql型数据
        if('month' in Factor):
            Factor.rename(columns={'month':'time'},inplace=True)
        if('date' in Factor):
            Factor.rename(columns={'date':'time'},inplace=True)
        factorList=Factor.columns        
        if(len(factorList)<=2):
            print('error')
            return Factor
        else:
            factorList=getfactorname(Factor,['code','time'])
        for factorname in factorList:
            if(factorname in self.factorlist):#如果重复则先删除信息再重新载入
                rest=pd.Series(self.factorlist)
                self.factorlist=rest[rest!=factorname].tolist()
                del self.FactorDataBase['v'][factorname]
            self.FactorDataBase['v']=self.FactorDataBase['v'].merge(Factor[['time','code',factorname]],on=['time','code'],how='outer')
            self.factorlist=self.factorlist+[factorname]
    
    def calcIC(self,factorlist='',startMonth='',endMonth=''):
        if(factorlist==''):
            factorlist=self.factorlist
        if(type(factorlist)==str):
            factorlist=[factorlist]
        if(startMonth==''):
            startMonth=int(str(self.startdate)[:6])
        if(endMonth==''):
            endMonth=int(str(self.enddate)[:6])
        RetData=self.retData
        RetData=RetData[RetData.time>=startMonth]
        RetData=RetData[RetData.time<=endMonth]
        if(type(self.filterStockDF)==pd.DataFrame):
            RetData=setStockPool(RetData,self.filterStockDF)
        for facname in factorlist:
            Mer=self.FactorDataBase['v'][['time','code',facname]].merge(RetData,on=['time','code'],how='outer').dropna()        
            self.ICList[facname],self.ICAns[facname]=calcIC(Mer,facname)
            if(len(factorlist)==1):
                print(facname+':')
                print(self.ICAns[facname])
        if(len(factorlist)>1):
            print(self.ICDF)

    def calcLongShort(self,factorlist='',startMonth='',endMonth='',t=5,asc=''):
        if(factorlist==''):
            factorlist=self.factorlist
        if(type(factorlist)==str):
            factorlist=[factorlist]
        if(startMonth==''):
            startMonth=int(str(self.startdate)[:6])
        if(endMonth==''):
            endMonth=int(str(self.enddate)[:6])
        RetData=self.retData
        RetData=RetData[RetData.time>=startMonth]
        RetData=RetData[RetData.time<=endMonth]
        if(type(self.filterStockDF)==pd.DataFrame):
            RetData=setStockPool(RetData,self.filterStockDF)
        for facname in factorlist:
            Mer=self.FactorDataBase['v'][['time','code',facname]].merge(RetData,on=['time','code'],how='outer').dropna()
            if(asc!=''):
                ascloc=asc
            else:
                ascloc=False
                if(facname in self.ICAns):
                    if(self.ICAns[facname]['IC:']<0):
                        ascloc=True
            Mer = Mer.groupby('time').apply(lambda x: isinGroupT(x, facname, asc=ascloc, t=t)).reset_index(drop=True)
            ls_ret = calcGroupRet(Mer,facname,RetData)
            ls_ret['多空组合'] = ls_ret[1] - ls_ret[t]  # 第一组-第五组
            if (facname in self.portfolioGroup.columns):  # 如果重复则先删除信息再重新载入
                self.portfolioGroup = self.portfolioGroup.drop(columns=facname)
            self.portfolioGroup = self.portfolioGroup.merge(Mer[['time','code',facname]], on=['time', 'code'], how='outer').dropna()
            self.portfolioList[facname]=ls_ret
            self.portfolioAns[facname]=evaluatePortfolioRet(ls_ret[1]-ls_ret[t])
            self.annualTurnover[facname] = calcAnnualTurnover(self.portfolioGroup, facname)
            if(len(factorlist)==1):
                print(facname+':')
                ls_ret1=ls_ret.reset_index().copy()
                ls_ret1['time']=ls_ret1['time'].apply(lambda x:str(x))
                ls_ret1.set_index('time').apply(lambda x:x+1).cumprod().plot()
                print(self.portfolioAns[facname])
            plt.show()
        if(len(factorlist)>1):
            print(self.portfolioDF)

        
    #常规测试流程
    def autotest(self,factorlist='',startMonth='',endMonth='',t=5,asc=''):
        self.calcIC(factorlist,startMonth,endMonth)
        self.calcLongShort(factorlist,startMonth,endMonth,t,asc)
        
    #计算按因子值排名前K
    def calcTopK(self,factorlist='',startMonth='',endMonth='',k=30,asc='',base=''):
        if(factorlist==''):
            factorlist=self.factorlist
        if(type(factorlist)==str):
            factorlist=[factorlist]
        if(startMonth==''):
            startMonth=int(str(self.startdate)[:6])
        if(endMonth==''):
            endMonth=int(str(self.enddate)[:6])
        RetData=self.retData
        RetData=RetData[RetData.time>=startMonth]
        RetData=RetData[RetData.time<=endMonth]
        if(type(self.filterStockDF)==pd.DataFrame):
            RetData=setStockPool(RetData,self.filterStockDF)
        if ((base != '') & (base in self.portfolioGroup.columns)):
            factorDB = self.portfolioGroup[self.portfolioGroup[base] == 1][['time', 'code']].merge(self.FactorDataBase['v'],on=['time', 'code'],how='inner').dropna()
        elif (base == ''):
            factorDB = self.FactorDataBase['v']
        else:
            print('error')
            return factorlist

        for facname in factorlist:
            Mer=factorDB[['time','code',facname]].merge(RetData,on=['time','code'],how='outer').dropna()
            if(asc!=''):
                ascloc=asc
            else:
                ascloc=False
                if(facname in self.ICAns):
                    if(self.ICAns[facname]['IC:']<0):
                        ascloc=True
            Mer = Mer.groupby('time').apply(lambda x: isinTopK(x, facname, ascloc, k=k)).reset_index(drop=True)
            topk_list = calcGroupRet(Mer,facname,RetData)
            if (facname in self.portfolioGroup.columns):  # 如果重复则先删除信息再重新载入
                self.portfolioGroup = self.portfolioGroup.drop(columns=facname)
            # portfoliogroup为1，表明按asc排序该股票的因子值在前k之内，为2表明因子值在倒数k个之内
            self.portfolioGroup = self.portfolioGroup.merge(Mer[['time','code',facname]], on=['time', 'code'], how='outer').fillna(0)
            self.portfolioList[facname]=topk_list
            self.portfolioAns[facname]=evaluatePortfolioRet(topk_list[1]-topk_list[0])
            self.annualTurnover[facname] = calcAnnualTurnover(self.portfolioGroup, facname)
            if(len(factorlist)==1):
                print(facname+':')
                topk_list['ls']=topk_list[1]-topk_list[0]
                calc_plot(topk_list.apply(lambda x:x+1).cumprod())
                print(self.portfolioAns[facname])
            plt.show()
        if(len(factorlist)>1):
            print(self.portfolioDF)


    def calcFutureRet(self,factorlist='',startMonth='',endMonth='',L=36,t=5,asc=''):
        '''
        Parameters
        ----------
        factorlist : TYPE, optional
            需要测试的因子  'factor1' 或 ['factor1','factor2'] 可留空
        startMonth :int 起始月份 201001   可留空
        endMonth : int 终止月份 形如202201 可留空
        L : 向后看的月数，默认36个月
        t : int 分组数，默认为5.
        asc : T or F 方向， 默认为True 从小到大 False为从大到小
        Returns
        -------
        返回每个月向后未来1到36个月的收益均值，存储在Test.FutureRet里面
        '''
        if(factorlist==''):
            factorlist=self.factorlist
        if(type(factorlist)==str):
            factorlist=[factorlist]
        if(startMonth==''):
            startMonth=int(str(self.startdate)[:6])
        if(endMonth==''):
            endMonth=int(str(self.enddate)[:6])
        RetData=self.retData.pivot(index='time',columns='code',values='ret')
        self.FutureRet=pd.DataFrame(columns=factorlist)
        RetData=RetData.apply(lambda x:np.log(x+1))
        for i in tqdm(range(1,L+1)):
            Ret_loc=RetData.rolling(window=i).sum().apply(lambda x:np.e**x-1).shift(-1*i+1).dropna(how='all').stack().reset_index()
            Ret_loc.columns=['time','code','ret']
            Ret_loc=Ret_loc[Ret_loc.time>=startMonth]
            Ret_loc=Ret_loc[Ret_loc.time<=endMonth]
            if(type(self.filterStockDF)==pd.DataFrame):
                Ret_loc=setStockPool(RetData,self.filterStockDF)
            for facname in factorlist:
                if(asc!=''):
                    ascloc=asc
                else:
                    ascloc=False
                    if(facname in self.ICAns):
                        if(self.ICAns[facname]['IC:']<0):
                            ascloc=True
                Mer=self.FactorDataBase['v'][['time','code',facname]].merge(Ret_loc,on=['time','code'],how='outer').dropna()        
                Mer=Mer.groupby('time').apply(isinGroupT,facname,asc=ascloc,t=t).reset_index(drop=True)
                ls_ret=calcGroupRet(Mer,facname,Ret_loc).reset_index()
                self.FutureRet.loc[i,facname]=(ls_ret[1]-ls_ret[t]).mean()#第一组-第五组
        self.FutureRet.plot()
    
    #计算胜率赔率
    def displayWinRate(self,factorlist=''):
        if(factorlist==''):
            factorlist=self.portfolioList.keys()
        for facname in factorlist:
            Mer=self.portfolioGroup[['time','code',facname]].merge(self.retData,on=['time','code'],how='outer').dropna()        
            L=Mer.groupby(['time']).apply(calcGroupWR,facname,self.retData)
            self.WR[facname]=L.mean()['WR']
            self.PL[facname]=L.mean()['PL']
        print(pd.concat([pd.Series(self.WR,name='WR'),pd.Series(self.PL,name='PL')],axis=1))
    #展示年度收益
    def displayYearPerformance(self,factorlist='',t=5):
        '''
        分年度打印：
            一、五组业绩
            一-五 收益率、信息比例、月胜率、最大回撤FB.evaluatePortfolioRet
        '''
        if(factorlist==''):
            factorlist=self.portfolioList.keys()
        if(type(factorlist)==str):
            factorlist=[factorlist]
        for facname in factorlist:
            portfolio=self.portfolioList[facname].reset_index()
            portfolio['time']=portfolio['time'].apply(lambda x:str(x)[:4])
            portfolioyear=portfolio.groupby('time')
            ans=pd.DataFrame()
            for year in portfolio.time.sort_values().unique():
                portfolio_loc=portfolioyear.get_group(year).set_index('time')
                ans1=evaluatePortfolioRet(portfolio_loc['多空组合'])
                ans1.loc[1]=(portfolio_loc[1]+1).prod()-1
                ans1.loc[t]=(portfolio_loc[t]+1).prod()-1
                ans1.name=year
                ans=ans.append(ans1)
            self.year_performance[facname]=ans 
    #计算相关性矩阵 1.因子值矩阵 2.IC矩阵
    def calcCorrMatrix(self,CorType=stats.spearmanr):
        '''
        self.factorCorr 因子相关性  ICCorr IC序列相关性
        默认使用 stats.spearmanr
        可换成stats.pearsonr

        Parameters
        ----------
        CorType : TYPE, optional
            DESCRIPTION. The default is stats.spearmanr.

        Returns
        -------
        None.

        '''
        self.factorCorr=pd.DataFrame([],index=self.factorlist,columns=self.factorlist)
        self.ICCorr=pd.DataFrame([],index=self.factorlist,columns=self.factorlist)
        for i in range(len(self.factorlist)):
            for j in  range(len(self.factorlist)):    
                if(i<j):
                    fac=self.FactorDataBase['v'][['time','code',self.factorlist[i],self.factorlist[j]]].dropna()
                    fac=fac.groupby('time').apply(lambda x:CorType(x[self.factorlist[i]],x[self.factorlist[j]])[0])
                    self.factorCorr.loc[self.factorlist[i],self.factorlist[j]]=fac.mean()
                    if(self.factorlist[i] in self.ICList and self.factorlist[j] in self.ICList):
                        A=pd.DataFrame(self.ICList[self.factorlist[i]],columns=[1])
                        A[2]=self.ICList[self.factorlist[j]]
                        A=A.dropna()
                        self.ICCorr.loc[self.factorlist[i],self.factorlist[j]]=CorType(A[1],A[2])[0]
        print('因子相关性:')
        print(self.factorCorr)
        print('IC相关性：')
        print(self.ICCorr.dropna(how='all').dropna(how='all',axis=1))
    #测试与Barra因子
    def calcCorrBarra(self,factorlist=''):
        if(factorlist==''):
            factorlist=self.factorlist
        if(type(factorlist)==str):
            factorlist=[factorlist]
        factor_tmp,Barra_list=addXBarra(self.FactorDataBase['v'][['time','code']+factorlist])
        Corr_Mat=pd.DataFrame(index=factorlist,columns=Barra_list)
        for fac in factorlist:
            for barra in Barra_list:
                corr_loc=factor_tmp[['time',fac,barra]].dropna()
                Corr_Mat.loc[fac,barra]=corr_loc.groupby('time').apply(lambda x:stats.spearmanr(x[fac],x[barra])[0]).mean()
        self.Corr_Mat=Corr_Mat
        print('与Barra相关性')
        print(self.Corr_Mat.T)                
        
    
    #得到纯因子，后缀为 +_pure
    def calcPureFactor(self,factorlist=''):
        if(factorlist==''):
            factorlist=self.factorlist
        if(type(factorlist)==str):
            factorlist=[factorlist]
        for fac in factorlist:
            factorDF=calcNeuBarra(self.FactorDataBase['v'], fac)
            self.getFactor(factorDF[['time','code',fac+'_pure']].dropna())
                
    @property
    def ICDF(self):
        return pd.DataFrame(self.ICAns).T
    @property
    def portfolioDF(self):
        return pd.DataFrame(self.portfolioAns).T
    

class IndTest(FactorTest):
    def __init__(self):
        self.startdate=20000101
        self.enddate=21000101
        self.factorlist=[]
        self.FactorDataBase={'v':pd.DataFrame(columns=['time','code'])}
        self.filterStockDF='FULL'
        self.retData = getIndRetData() #统一为time、code  time为int  code 为str 
        self.ICList={}
        self.portfolioList={}
        self.ICAns={}
        self.portfolioAns={}
        self.portfolioGroup = pd.DataFrame(columns=['time', 'code'])
        self.annualTurnover = {}
        self.year_performance={}
        self.indStatus=pd.read_csv(filepathtestdata+'sw1.csv').set_index('申万代码')
        pd.options.mode.use_inf_as_na = True  #剔除inf
        self.dataProcess=dataProcess(self.FactorDataBase)

    #将个股转换为行业数据
    @staticmethod
    def convertStocktoInd(Factor,func=lambda x:x.mean()):
        if('month' in Factor):
            Factor.rename(columns={'month':'time'},inplace=True)
        if('date' in Factor):
            Factor.rename(columns={'date':'time'},inplace=True)
        factorList=Factor.columns        
        if(len(factorList)<=2):
            print('error')
            return Factor
        else:
            factorList=getfactorname(Factor,['code','time'])
        DF=pd.DataFrame(columns=['time','code'])
        indStatus=pd.read_csv(filepathtestdata+'sw1.csv').set_index('申万代码')
        for facname in factorList:
            DataLoc=Factor[['time','code',facname]]
            DataLoc=DataLoc.pipe(getSWIndustry,freq='month')
            DataLoc=DataLoc.groupby(['time','SWind']).mean().reset_index()
            A=DataLoc.groupby('SWind')
            for ind in DataLoc['SWind'].unique():
                DataLoc.loc[A.get_group(ind)['SWind'].index,'code']=indStatus.loc[ind,'代码']
            DF=DF.merge(DataLoc[['time','code',facname]],on=['time','code'],how='outer')
        return DF


