import os
import FactorTest.FactorTestBox as FB
from FactorTest.FactorTestPara import *
import pandas as pd
import numpy as np
from tqdm import tqdm
#

#A股日行情（放在第一列更新,同时会维护tradedate)
def getAShareEODPrices(infoDF):
    starttime=FB.getUpdateStartTime(infoDF['最新时间'])
    sqlData=FB.getSql('select S_INFO_WINDCODE,TRADE_DT,'+','.join(infoDF['数据库键'])+' from wind.AShareEODPrices where TRADE_DT>='+str(int(starttime)))
    sqlData.rename(columns={'S_INFO_WINDCODE':'code','TRADE_DT':'time'},inplace=True)
    FB.saveDailyData(sqlData,infoDF)


#A股日行情1（放在第一列更新,同时会维护tradedate)
def getAShareEODPrices1(infoDF):
    starttime=FB.getUpdateStartTime(infoDF['最新时间'])
    sqlData=FB.getSql('select S_INFO_WINDCODE,TRADE_DT,'+','.join(infoDF['数据库键'])+' from wind.AShareEODPrices where TRADE_DT>='+str(int(starttime)))
    sqlData.rename(columns={'S_INFO_WINDCODE':'code','TRADE_DT':'time'},inplace=True)
    FB.saveDailyData(sqlData,infoDF)


#A股日行情衍生指标
def AShareEODDerivativeIndicator(infoDF):
    for ind in tqdm(FB.partition(infoDF.index,5)):
        info_loc=infoDF.loc[ind]
        starttime=FB.getUpdateStartTime(info_loc['最新时间'])
        sqlData=FB.getSql('select S_INFO_WINDCODE,TRADE_DT,'+','.join(info_loc['数据库键'])+' from wind.AShareEODDerivativeIndicator where TRADE_DT>='+str(int(starttime)))
        sqlData.rename(columns={'S_INFO_WINDCODE':'code','TRADE_DT':'time'},inplace=True)
        FB.saveDailyData(sqlData,info_loc)



#A股财务数据
def AShareFinancialIndicator(infoDF):
    starttime=FB.getUpdateStartTime(infoDF['最新时间'])
    sqlData=FB.getSql('select S_INFO_WINDCODE,REPORT_PERIOD,'+','.join(infoDF['数据库键'])+' from wind.AShareFinancialIndicator where REPORT_PERIOD>='+str(int(starttime)))
    sqlData.rename(columns={'S_INFO_WINDCODE':'code','REPORT_PERIOD':'time'},inplace=True)
    FB.saveFinData(sqlData,infoDF)

#sw指数成分股
def AShareSWIndustriesClass(infoDF):
    starttime=FB.getUpdateStartTime(infoDF['最新时间'])
    sqlData=FB.getSql('select S_INFO_WINDCODE,SW_IND_CODE,ENTRY_DT,REMOVE_DT from wind.AShareSWIndustriesClass where ENTRY_DT>='+str(int(starttime)))
    sqlData.columns=['code',infoDF['数据库键'].iloc[0],'time','eddate']
    FB.saveIndData(sqlData,infoDF)


#sw指数价格
def ASWSIndexEOD(infoDF):
    starttime=FB.getUpdateStartTime(infoDF['最新时间'])
    sqlData=FB.getSql('select S_INFO_WINDCODE,TRADE_DT,'+','.join(infoDF['数据库键'])+' from wind.ASWSIndexEOD where TRADE_DT>='+str(int(starttime)))
    sqlData.rename(columns={'S_INFO_WINDCODE':'code','TRADE_DT':'time'},inplace=True)
    FB.saveDailyData(sqlData,infoDF)


def getHS300Weight(infoDF):
    #AIndexHS300CloseWeight
    #AIndexCSI500WeightWeight
    starttime=FB.getUpdateStartTime(infoDF['最新时间'])
    sqlData=FB.getSql('select S_CON_WINDCODE,TRADE_DT,I_WEIGHT \
                      from wind.AIndexHS300CloseWeight where TRADE_DT>'+str(int(starttime)))
    sqlData.rename(columns={'S_CON_WINDCODE':'code','TRADE_DT':'time'},inplace=True)    
    FB.saveDailyData(sqlData,infoDF)

def readWindData(ans,ind='Times',col='Codes'):
    return pd.DataFrame(ans.Data,index=getattr(ans,ind),columns=getattr(ans,col))

def getZZ500EWeight(infoDF):
    starttime=str(FB.getUpdateStartTime(infoDF['最新时间']))
    starttime=starttime[:4]+'-'+starttime[4:6]+'-'+starttime[6:]
    from WindPy import w
    w.start()
    ans=w.wset("indexhistory","startdate="+starttime+";enddate=2100-12-31;windcode=000905.SH")
    w.close()
    try:
        ans=readWindData(ans,'Fields','Codes').T[['tradedate','tradecode','tradestatus']]
    except:
        ans=pd.DataFrame(index=['tradedate','tradecode','tradestatus']).T
    FB.saveIndexComponentData(ans, infoDF)
    


def getZZ1000EWeight(infoDF):
    starttime=str(FB.getUpdateStartTime(infoDF['最新时间']))
    starttime=starttime[:4]+'-'+starttime[4:6]+'-'+starttime[6:]
    from WindPy import w
    w.start()
    ans=w.wset("indexhistory","startdate="+starttime+";enddate=2100-12-31;windcode=000852.SH")
    w.close()
    try:
        ans=readWindData(ans,'Fields','Codes').T[['tradedate','tradecode','tradestatus']]
    except:
        ans=pd.DataFrame(index=['tradedate','tradecode','tradestatus']).T
    FB.saveIndexComponentData(ans, infoDF)


def getWindAEWeight(infoDF):
    starttime=str(FB.getUpdateStartTime(infoDF['最新时间']))
    starttime=starttime[:4]+'-'+starttime[4:6]+'-'+starttime[6:]
    from WindPy import w
    w.start()
    ans=w.wset("indexhistory","startdate="+starttime+";enddate=2100-12-31;windcode=881001.WI")
    w.close()
    try:
        ans=readWindData(ans,'Fields','Codes').T[['tradedate','tradecode','tradestatus']]
    except:
        ans=pd.DataFrame(index=['tradedate','tradecode','tradestatus']).T
    FB.saveIndexComponentData(ans, infoDF)

def getValidData(infoDF):
    #ST
    sqlData=FB.getSql('select S_INFO_WINDCODE,S_TYPE_ST,ENTRY_DT,REMOVE_DT from wind.AShareST ')
    sqlData=sqlData[sqlData.S_TYPE_ST=='S']
    del sqlData['S_TYPE_ST']
    sqlData.columns=['code','time','eddate']
    sqlData1=sqlData[['time','code']]
    sqlData1['ST']=1
    sqlData2=sqlData[['eddate','code']].dropna()
    sqlData2.columns=['time','code']
    sqlData2['out']=-1
    sqlData=sqlData1.merge(sqlData2,on=['time','code'],how='outer').drop_duplicates(subset=['time','code'],keep='first')
    sqlData['time']=sqlData['time'].apply(lambda x:int(x))
    sqlData.loc[sqlData['ST']!=sqlData['ST'],'ST']=sqlData.loc[sqlData['ST']!=sqlData['ST'],'out']
    del sqlData['out']
    data0=sqlData.pivot(index='time',columns='code',values='ST').reindex(index=FB.getTradeDateList(),columns=FB.getStockList()).fillna(method='ffill').reset_index()
    data0[data0==-1]=np.nan
    FB.save_feather(Datapath+infoDF.iloc[0]['存储地址'],data0)
    #停牌
    TradeStatus=FB.read_feather(Datapath+'BasicFactor_TradeStatus.txt').set_index('time').stack().reset_index()
    TradeStatus.columns=['time','code','TS']
    TradeStatus.loc[TradeStatus.TS!='停牌','TS']=0
    TradeStatus.loc[TradeStatus.TS=='停牌','TS']=1
    TradeStatus=TradeStatus.pivot(index='time',columns='code',values='TS')
    FB.save_feather(Datapath+'BasicFactor_StockHaltDay.txt',TradeStatus.reset_index())
    
    Close=FB.read_feather(Datapath+'BasicFactor_Close.txt').set_index('time')
    Close[Close==Close]=1
    #上市时间
    Close=Close.fillna(method='ffill').cumsum()
    FB.save_feather(Datapath+infoDF.iloc[2]['存储地址'],Close.reset_index())
    
    STData=FB.read_feather(Datapath + 'BasicFactor_isST.txt').set_index('time')
    LTData=FB.read_feather(Datapath + 'BasicFactor_StockHaltDay.txt').set_index('time')
    HDData=FB.read_feather(Datapath + 'BasicFactor_ListedTime.txt').set_index('time')
    windData=FB.read_feather(Datapath + 'WINDAComponent.txt').set_index('time')
    ValidDF=(STData!=1) * (LTData!=1)* ( HDData>=60)*(windData==1)
    FB.save_feather(Datapath+'BasicFactor_ValidDF.txt',ValidDF.reset_index())
    
def getDailyRetData(infoDF):
    FB.updateStockDailyRet()
    
def getDailyIndexData(infoDF): 
    #从Sql提取数据
    starttime=FB.getUpdateStartTime(infoDF['最新时间'])
    sqlData=FB.getSql('select S_INFO_WINDCODE,TRADE_DT,'+','.join(infoDF['数据库键'])+' from wind.AIndexEODPrices where TRADE_DT>='+str(int(starttime)))
    sqlData.rename(columns={'S_INFO_WINDCODE':'code','TRADE_DT':'time'},inplace=True)
    FB.saveDailyData(sqlData,infoDF)
    
def getAnalystData(infoDF):
    infoDF=infoDF[infoDF['数据库键']!='OBJECT_ID']
    infoDF=infoDF[infoDF['数据库键']!='EST_DT']
    for ind in tqdm(infoDF['存储地址'].unique()):
        info_loc=infoDF[infoDF['存储地址']==ind]
        starttime=FB.getUpdateStartTime(info_loc['最新时间'])
        sqlData=FB.getSql('select OBJECT_ID ,EST_DT,'+','.join(info_loc['数据库键'])+'\
                      from wind.AShareConsensusData where EST_DT>='+str(int(starttime)))
        sqlData.rename(columns={'OBJECT_ID':'id','EST_DT':'time'},inplace=True)
        FB.saveSqlData(sqlData,info_loc)
    
    
    
    