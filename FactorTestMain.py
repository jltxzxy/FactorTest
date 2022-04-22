from FactorTest.FactorTestBox import *
from FactorTest.FactorTestPara import *


class FactorTest():
    def __init__(self):
        self.startdate=20000101
        self.enddate=21000101
        self.factorlist=[]
        self.FactorDataBase=pd.DataFrame(columns=['time','code'])
        self.filterStockDF='FULL'
        self.retData = getRetData() #统一为time、code  time为int  code 为str 
        self.ICList={}
        self.portfolioList={}
        self.ICAns={}
        self.portfolioAns={}
        self.portfolioGroup = pd.DataFrame(columns=['time', 'code'])
        
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
                del self.FactorDataBase[factorname]
            self.FactorDataBase=self.FactorDataBase.merge(Factor[['time','code',factorname]],on=['time','code'],how='outer')
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
            Mer=self.FactorDataBase[['time','code',facname]].merge(RetData,on=['time','code'],how='outer').dropna()        
            self.ICList[facname],self.ICAns[facname]=calcIC(Mer,facname)
            if(len(factorlist)==1):
                print(facname+':')
                print(self.ICAns[facname])
        if(len(factorlist)>1):
            print(self.ICDF)

    def calcLongShort(self,factorlist='',startMonth='',endMonth='',t=5,asc=True):
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
            Mer=self.FactorDataBase[['time','code',facname]].merge(RetData,on=['time','code'],how='outer').dropna()
            ls_ret = Mer.groupby('time').apply(lambda x: longshortfive(x, facname, asc=asc, t=t)[0]).dropna()
            ls_ret['多空组合'] = ls_ret[1] - ls_ret[t]  # 第一组-第五组
            isingroupt = Mer.groupby('time').apply(lambda x: longshortfive(x, facname, asc=asc, t=t)[1]).reset_index(drop=True)
            self.portfolioGroup = self.portfolioGroup.merge(isingroupt, on=['time', 'code'], how='outer').dropna()

            self.portfolioList[facname]=ls_ret
            self.portfolioAns[facname]=evaluatePortfolioRet(ls_ret[1]-ls_ret[t]) 
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
    def autotest(self,factorlist='',startMonth='',endMonth='',t=5,asc=True):
        self.calcIC(factorlist,startMonth,endMonth)
        self.calcLongShort(factorlist,startMonth,endMonth,t,asc)
        
    #计算按因子值排名前K
    def calcTopK(self,factorlist='',startMonth='',endMonth='',k=30,asc=True,base=''):
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
            factorDB = self.portfolioGroup[self.portfolioGroup[base] == 1][['time', 'code']].merge(self.FactorDataBase,on=['time', 'code'],how='inner').dropna()
        elif (base == ''):
            factorDB = self.FactorDataBase
        else:
            print('error')
            return factorlist

        for facname in factorlist:
            Mer=factorDB[['time','code',facname]].merge(RetData,on=['time','code'],how='outer').dropna()
            Mer['time']=Mer['time'].apply(lambda x:str(x))
            topk_list = Mer.groupby('time').apply(lambda x: selecttopK(x, facname, asc, k=k)[0]).reset_index()
            isintopk = Mer.groupby('time').apply(lambda x: selecttopK(x, facname, asc, k=k)[1]).reset_index(drop=True)
            self.portfolioGroup = self.portfolioGroup.merge(isintopk, on=['time', 'code'], how='outer').fillna(0)
            self.portfolioList[facname]=topk_list
            self.portfolioAns[facname]=evaluatePortfolioRet(topk_list['up'])
            if(len(factorlist)==1):
                print(facname+':')
                topk_list['up'].apply(lambda x:x+1).cumprod().plot()
                print(self.portfolioAns[facname])
            plt.show()
        if(len(factorlist)>1):
            print(self.portfolioDF)


    def calcFutureRet(self,factorlist='',startMonth='',endMonth='',L=36,t=5,asc=True):
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
        for i in range(1,L+1):
            Ret_loc=RetData.rolling(window=i).sum().apply(lambda x:np.e**x-1).shift(-1*i+1).dropna(how='all').stack().reset_index()
            Ret_loc.columns=['time','code','ret']
            Ret_loc=Ret_loc[Ret_loc.time>=startMonth]
            Ret_loc=Ret_loc[Ret_loc.time<=endMonth]
            if(type(self.filterStockDF)==pd.DataFrame):
                Ret_loc=setStockPool(RetData,self.filterStockDF)
            
            for facname in factorlist:
                Mer=self.FactorDataBase[['time','code',facname]].merge(Ret_loc,on=['time','code'],how='outer').dropna()        
                ls_ret=Mer.groupby('time').apply(longshortfive,facname,asc=asc,t=t).dropna()
                self.FutureRet.loc[i,facname]=(ls_ret[1]-ls_ret[t]).mean()#第一组-第五组
        self.FutureRet.plot()
    #双分组，打印热力图、想一想如何返回值
    def calcDoubleSort(self):
        pass
    #计算日度多空收益
    def calcDailyLongShort(self):
        pass
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
                     fac=self.FactorwDataBase[['time','code',self.factorlist[i],self.factorlist[j]]]
                     fac=fac.groupby('time').apply(lambda x:CorType(x[self.factorlist[i]],x[self.factorlist[j]])[0])
                     self.factorCorr.loc[self.factorlist[i],self.factorlist[j]]=fac.mean()
                     self.ICCorr.loc[self.factorlist[i],self.factorlist[j]]=CorType(self.ICList[self.factorlist[i]],self.ICList[self.factorlist[j]])[0]
        print('因子相关性:')
        print(self.factorCorr)
        print('IC相关性：')
        print(self.ICCorr)
    #测试与Barra因子
    def calcCorrBarra(self,factorlist=''):
        if(factorlist==''):
            factorlist=self.factorlist
        if(type(factorlist)==str):
            factorlist=[factorlist]
        factor_tmp,Barra_list=addXBarra(self.FactorDataBase[['time','code']+factorlist])
        Corr_Mat=pd.DataFrame(index=factorlist,columns=Barra_list)
        for fac in factorlist:
            for barra in Barra_list:
                corr_loc=factor_tmp[['time',fac,barra]].dropna()
                Corr_Mat.loc[fac,barra]=corr_loc.groupby('time').apply(lambda x:stats.spearmanr(x[fac],x[barra])[0]).mean()
        self.Corr_Mat=Corr_Mat
        print('与Barra相关性')
        print(self.Corr_Mat)                
        
    
    #得到纯因子，后缀为 +_pure
    def calcPureFactor(self,factorlist=''):
        if(factorlist==''):
            factorlist=self.factorlist
        if(type(factorlist)==str):
            factorlist=[factorlist]
        for fac in factorlist:
            factorDF=calcNeuBarra(self.FactorDataBase, fac)
            self.getFactor(factorDF[['time','code',fac+'_pure']].dropna())
                
    @property
    def ICDF(self):
        return pd.DataFrame(self.ICAns).T
    @property
    def portfolioDF(self):
        return pd.DataFrame(self.portfolioAns).T