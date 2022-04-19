import FactorTest.FactorTestBox as FB
import FactorTest.FactorTestMain as FM
from FactorTest.FactorTestPara import *
from FactorTest.FactorTestBox import *
import pandas as pd

#获取指数成分股数据
def Indexdata(data,indexname='wind'):
    if(indexname=='300'):
        Component = getIndexComponent(indexname=300)[['time','code']]
    if(indexname=='500'):
        Component = getIndexComponent(indexname=500)
        Component = Component[Component['isComponent'] == 1]
        Component=Component[['time', 'code']]
    if(indexname=='800'):
        HS300 = getIndexComponent(indexname=300)
        CSI500 = getIndexComponent(indexname=500)
        CSI500 = CSI500[CSI500['isComponent'] == 1]
        Component = pd.concat([HS300, CSI500], ignore_index=True)
        Component = Component[['time', 'code']]
    if(indexname=='1000'):
        Component = getIndexComponent(indexname=1000)
        Component = Component[Component['isComponent'] == 1]
        Component = Component[['time', 'code']]
    if(indexname=='wind'):
        Component = getIndexComponent(indexname='wind')[['time','code']]
    Component=Component.reset_index(drop=True)
    mer = Component.merge(data, how='inner', on=['time', 'code']).dropna()
    return mer


#计算股息率数据
dividend=pd.read_csv(Datapath+'dividend.csv')[['S_INFO_WINDCODE','CASH_DVD_PER_SH_PRE_TAX','REPORT_PERIOD']].dropna()
dividend.columns=['code','DPS','time']
dividend['year']=dividend['time'].apply(lambda x: int(str(x)[:4]))
dividend=dividend.groupby(['year','code'])['DPS'].sum().reset_index()
dividend['time']=dividend['year']*10000+1231
eps=read_feather(Datapath + 'BasicFactor_S_Fa_Eps_Basic.txt').set_index('time').stack().reset_index()
eps.columns=['time','code','EPS']
dividend=dividend.merge(eps,on=['code','time'])
dividend['payoutRatio']=dividend['DPS']/dividend['EPS']
dividend['rank']=dividend.groupby('time')['payoutRatio'].rank(ascending=True,pct=True)
dividend=dividend[dividend['rank']<=0.95]
dividend=dividend[~(dividend['payoutRatio']<0)]
dividend['preDPS']=dividend.sort_values(by='time').groupby('code')['DPS'].shift(1)
dividend=dividend.sort_values(by=['code','time'])
dividend['growthRate']=dividend['DPS']/dividend['preDPS']-1
dividend=dividend[dividend['growthRate']>=0]
dividend['months']='1/2/3/4/5/6/7/8/9/10/11/12'
dividend.loc[:,['months']]=dividend['months'].str.split('/')
dividend=dividend.explode(column='months')
dividend['time']=dividend['year']*100+dividend['months'].astype(int)
DVD=Indexdata(dividend[['time','code','payoutRatio']],indexname='500')


#计算波动数据
close = read_feather(Datapath + 'BasicFactor_Close.txt').set_index('time')
adjfactor = read_feather(Datapath + 'BasicFactor_AdjFactor.txt').set_index('time')
adjclose = close * adjfactor
adjopen = read_feather(Datapath + 'BasicFactor_Open.txt').set_index('time') * adjfactor  # 调整后的开盘价和收盘价
adjclose = adjclose.fillna(method='ffill')
adjopen = adjopen.fillna(method='ffill')
DailyYield = adjclose / adjopen - 1  # 算出收益率
DailyYield = DailyYield.stack().reset_index()
DailyYield.columns = ['date', 'code', 'ret']
DailyYield['time']=DailyYield['date'].apply(lambda x: int(str(x)[:6]))
del close, adjclose, adjopen, adjfactor
indexRet=Indexdata(DailyYield,indexname='500')
#剔除ST、上市60天以内、停牌股(该程序中只对波动率的计算做了剔除，因为最后波动率与红利数据合成时会剔除掉红利数据中有异常情况的部分)
ValidDF=read_feather(Datapath + 'BasicFactor_ValidDF.txt')
ValidDF=ValidDF.set_index('time').stack().reset_index()
ValidDF.columns=['date','code','valid']
indexRet = indexRet.merge(ValidDF, on=['code', 'date'])
indexRet = indexRet[indexRet.valid == True]
del indexRet['valid']
indexRet=indexRet.reset_index(drop=True)
indexRet['vol20']=indexRet.sort_values(by='date').groupby('code').rolling(20,min_periods=1)['ret'].std().reset_index(drop=True)
indexRet=indexRet.dropna().reset_index(drop=True)
indexRet['AVG_vol']=indexRet.sort_values(by='date').groupby('code').rolling(252,min_periods=1)['vol20'].mean().reset_index(drop=True)
VOL=indexRet.drop_duplicates(subset=['time', 'code'], keep='last')[['time','code','AVG_vol']].reset_index(drop=True)

#运用FM筛选低波高红利股票
Test=FM.FactorTest()
data=DVD.merge(VOL,on=['time','code'],how='inner')
Test.getFactor(data)
Test.calcTopKpct('payoutRatio',k=0.3,asc=False)
Test.calcTopK('AVG_vol',k=50,asc=True,base='payoutRatio')


