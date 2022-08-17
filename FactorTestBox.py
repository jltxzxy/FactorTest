import pandas as pd
import numpy as np
from math import *
from scipy.stats import norm
from scipy.special import erfinv
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy
import scipy.stats as stats
import os
import pymysql
#import cx_Oracle
import pickle
import sys
import seaborn
import time
from tqdm import tqdm
import h5py
import warnings
from FactorTest.FactorTestPara import *
warnings.filterwarnings('ignore')
import datetime
import zipfile
import shutil
import pyfinance.ols as po
import functools

#设置图片保存格式和字体
import matplotlib as mpl
# mpl.rcParams['font.sans-serif'] = ['PingFang HK']  # 设置默认字体
# mpl.rcParams['axes.unicode_minus'] = False #解决保存图像是负号显示为方块问题
#读入pickle文件
def read_pickle(filename):
    '''
    功能：读入pickle文件
    输入：pickle文件路径
    输出：sql数据
    '''
    with open(filename, 'rb') as f:
        sql_data = pickle.load(f)
    return sql_data
def save_pickle(filename,data):
    '''
    功能：保存数据至pickle文件
    输入：保存文件的路径，待保存的数据
    输出：无
    '''
    with open(filename,'wb') as f:
        '''
            保存pickle文件到指定目录，这里每次保存前先将pickle清空
            当保存dict文件的时候，需要设置highest_protocal=True
            '''
        pick=pickle.Pickler(f)
        pick.clear_memo()
        pick.dump(data)
def read_feather(filename):
    '''
    功能：读取feather文件（后缀为.txt)
    输入：feather文件路径
    输出：feather文件中储存的object
    '''
    # sql_data = feather.read_feather(filename)
    return pd.read_feather(filename)
def save_feather(filename,data):
    """
    功能：将数据保存在feather文件中
    输入：保存文件的路径和数据
    输出：无
    注意：要求存储内容保持一致
    """
    # feather.write_feather(data,filename)
    data.reset_index(drop=True).to_feather(filename)
# 保存hdf5文件，格式可以是df,也可以是很多df构成的dict，只支持一层嵌套
def save_hdf5(df_or_dict,file_full_root,subgroup=None):
    '''
    保存hdf5文件，格式可以是df,也可以是很多df构成的dict，只支持一层嵌套
    :param df_or_dict: 一个df或者dict{df}
    :param file_full_root: h5文件的全路径
    :param subgroup: 如果是None，则保存在hdf5文件最外层的目录下，否则，整个df或者dict保存在对应的subgroup内
                     如果输入的是dict,那么保存在subgroup内，如果输入的是df，则将df保存在对应的subgroup内
    :return:
    '''
    file_h = pd.HDFStore(file_full_root,'a')
    file_keys_raw = file_h.keys()
    file_keys = [x[1:] for x in file_keys_raw]
    file_h.close()
    if isinstance(df_or_dict,pd.DataFrame) or isinstance(df_or_dict,pd.Series):
        if subgroup is None:
            df_or_dict.to_hdf(file_full_root,key='_data')
        else:
            df_or_dict.to_hdf(file_full_root,key=subgroup)
    elif isinstance(df_or_dict,dict):
        file_h = pd.HDFStore(file_full_root, 'a')
        if subgroup is None:
            for sub_df_name in df_or_dict:
                file_h[sub_df_name] = df_or_dict[sub_df_name]
        else:
            for sub_df_name in df_or_dict:
                file_h[subgroup][sub_df_name] = df_or_dict[sub_df_name]
        file_h.close()
    else:
        raise Exception('wrong input type, only df or dict{df} supported!')
    print(file_full_root+' save finished!')
# 读取hdf5文件，h5文件可以是里面['_data']是df，也可以是一整个dict{df},或者选定要读取哪个subgroup
def read_hdf5(file_full_root,subgroup=None):
    '''
    功能：读取hdf5文件，h5文件可以是里面['_data']是df，也可以是一整个dict{df},或者选定要读取哪个subgroup
    输入：文件路径，选定读取的subgroup(默认读整个）
    输出：hdf5文件内的df或df字典信息
    '''
    if os.path.exists(file_full_root):
        file_h = pd.HDFStore(file_full_root,'r')
        file_keys_raw = file_h.keys()
        file_keys = [x[1:] for x in file_keys_raw]
        if len(file_keys)==1 and subgroup is None:
            result = file_h[file_keys[0]]
        elif len(file_keys)==1 and subgroup is not None:
            if subgroup in file_keys:
                result = file_h[subgroup]
            else:
                result = pd.DataFrame()
        elif len(file_keys)>1 and subgroup is not None:
            try:
                result = file_h[subgroup]
            except:
                result = pd.DataFrame()
        elif len(file_keys)>1 and subgroup is None:
            result = {}
            for i in file_keys:
                result[i] = file_h[i]
        else:
            # 原始文件是空的
            if subgroup is not None:
                result = pd.DataFrame()
            else:
                result = None
        file_h.close()
    else:
        result = None
    return result


# 从东吴在线落地数据库提取数据
def getSql(sql_req):
    """
    功能：从东吴在线落地数据库提取数据
    输入：
    @param sql_req:sql代码
    @return:DF格式
    输出：读取的sql信息
    """
    os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
    Oracle_userName = 'sjcj'
    Oracle_password = 'dwzqsjcj'
    Oracle_sid = 'sidwdzx1'
    Oracle_ip = '180.168.106.62'
    Oracle_port = '20001'
    dsn_tns = cx_Oracle.makedsn(Oracle_ip, Oracle_port, Oracle_sid)
    db_oracle = cx_Oracle.connect(Oracle_userName, Oracle_password, dsn_tns)
    data = pd.read_sql(sql_req, con=db_oracle)
    return data

def get_value(df,n):
    '''很多因子计算时，会一次性生成很多值，使用时只取出一个值'''
    def get_value_single(x,n):
        try:
            return x[n]
        except Exception:
            return np.nan
    df=df.applymap(lambda x:get_value_single(x,n))
    return df

#计算最大回撤
def maxDrawDown(Rev_seq):
    """
    功能:求最大回撤率
    输入:return_list:Series格式月度收益率
    输出:最大回撤（值在0-1之间）
    """
    Rev_list=(Rev_seq+1).cumprod()
    return (Rev_list/Rev_list.cummax()-1).min()
#修改——自动多样式
def toTime(x):
    '''
    功能:将字符串格式转化为datetime格式
    输入:包含日期字符串list或其他iterable格式
    输出:包含日期datetime的序列
    '''
    if(len(str(pd.Series(x).iloc[0]))==6):
        return pd.Series(x).apply(lambda x:pd.to_datetime(str(int(x))+'01'))
    return pd.Series(x).apply(lambda x:pd.to_datetime(str(int(x))))

def fromTime(x,strtype='%Y%m%d'):
    '''
    功能：将日期格式转化为int格式
    输入：包含日期格式的iterable格式,str格式
    输出：包含int格式日期的序列
    '''
    return pd.Series(x).apply(lambda x:int(x.strftime(strtype)))

#回归：x可以选择是否加截距项
def regress(y,x,con=True,weight=1):
    """
    功能：回归——注意不能出现空值
    输入：
    @param y:   应变量
    @param x:   自变量
    @param con: 是否加入截距项（涉及到行业信息的不加）
    @param weight: 加权参数，默认为1
    输出: 返回res  残差：res.resid
    """
    import statsmodels.api as sm # 最小二乘
    from statsmodels.stats.outliers_influence import summary_table # 获得汇总信息
    if(con):
        x=sm.add_constant(x)
    regr = sm.WLS(y, x,weight=weight) # 加权最小二乘模型，ordinary least square model
    res = regr.fit()    #res.model.endog
    return res
#更新数据 获得最新交易日历、计算收益序列数据  输出到一张表中：各指标最新日期
def updateStockDailyRet():
    '''
    功能：更新数据，获得最新交易日历、计算收益序列数据
    输入：无
    输出：
    @stockret.csv 更新后的数据表
    @stockret.txt 更新后的数据feather文件
    '''
    close=read_feather(Datapath+'BasicFactor_Close.txt').set_index('time')
    adjfactor=read_feather(Datapath+'BasicFactor_AdjFactor.txt').set_index('time')
    adjclose=close*adjfactor
    adjopen=read_feather(Datapath+'BasicFactor_Open.txt').set_index('time')*adjfactor
    
    adjclose=adjclose.reset_index().fillna(method='ffill')
    adjclose['time']=adjclose['time'].apply(lambda x:int(str(x)[:6]))
    adjclose=adjclose.drop_duplicates(subset=['time'],keep='last').set_index('time')

    adjopen=adjopen.reset_index()
    adjopen['time']=adjopen['time'].apply(lambda x:int(str(x)[:6]))
    adjopen=adjopen.drop_duplicates(subset=['time'],keep='first').set_index('time')

    adjclose=adjclose/adjopen-1

    adjclose=adjclose.stack().reset_index()   
    adjclose.columns=['time','code','ret']
    def shift1(x):
        x=int(x)-1
        if(x%100==0):
            x=x-88
        return x  
    adjclose['time']=adjclose['time'].apply(shift1)
    adjclose=kickout(adjclose)
    save_feather(Datapath+'stockret.txt',adjclose.reset_index(drop=True))
#获得交易日历
def getTradeDateList(period='date'):
    '''
    功能:获得交易日历
    输入:交易周期，默认为日度，可修改为月度
    输出:pandas series包含交易日期
    '''
    data=read_feather(Datapath+'BasicFactor_close.txt')
    if(period=='date'):
        return data['time']
    elif(period=='month'):
        return pd.Series(data['time'].apply(lambda x:int(str(x)[:6])).unique(),name='time')
def getFisicalList(period='Season'):
    '''
    功能：获得财务公告日历
    输入：数据周期，默认为季度，可改为年度
    输出：pandas series包含公告日期
    '''
    data=read_feather(Datapath+'BasicFactor_AShareFinancialIndicator_ANN_DT.txt')
    year_list=pd.Series(data['time'].apply(lambda x:int(str(x)[:4])).unique(),name='time')
    if(period=='year'):
        return year_list.apply(lambda x:int(str(x)+'1231'))
    else:
        monthlist=[]
        for year in year_list:
            for season in ['0331','0630','0930','1231']:
                monthlist.append(int(str(year)+season))
        monthlist=pd.Series(monthlist,name='time')
        return monthlist[monthlist>=19891231]
#获取股票序列
def getStockList():
    '''
    功能：获取股票序列数据
    输入：无
    输出：pandas series包含数据库中所有股票
    '''
    data=read_feather(Datapath+'BasicFactor_close.txt').set_index('time')
    return pd.Series(data.columns,name='code')
#获取收益序列数据
def getRetData():
    '''
    功能：获取收益学咧数据
    输入：无
    输出：pandas dataframe包含数据库中的股票收益
    '''
    retData=read_feather(Datapath+'stockret.txt').dropna()
    return retData

#读取指数行情数据 改 缺指数收益数据
def getIndexData(code):
    """
    功能：获取指数行情数据
    输入：股票代码
    示例：code=["000300.SH",'000905.SH','000985.SH']
    输出：指数行情数据
    """
    indexData=pd.read_csv(filepathtestdata+'index_data.csv',index_col=0)
    indexData.columns=['code','time','ret']
    indexData=indexData[indexData.code==code]
    indexData=indexData.sort_values(by='time')
    return indexData
#读取指数成分股列表 改  缺 500、1000成分股数据
def getIndexComponent(indexname='wind'):
    """
    功能：获取指数成分股列表
    输入：指数名称，默认万得，可改为恒生300，中证500，恒生300+中证500，中证1000
    indexname={300,500,800,1000}
    输出：dataframe包含成份股列表信息
    """
    if(indexname=='wind'):
        IndexComponent=readLocalData('WINDAComponent.txt').dropna()
        IndexComponent=IndexComponent[IndexComponent.isComponent!=0]

    if(indexname==300):
        IndexComponent=readLocalData('HS300Component.txt').dropna()
    if(indexname==500):
        IndexComponent=readLocalData('CSI500Component.txt').dropna().reset_index(drop=True)
        IndexComponent=IndexComponent[IndexComponent['000905.SH']!=0]
    if(indexname==800):
        IndexComponent=readLocalData('HS300Component.txt').dropna().reset_index(drop=True)
        IndexComponent1=readLocalData('CSI500Component.txt').dropna().reset_index(drop=True)
        IndexComponent=IndexComponent.append(IndexComponent1)
        IndexComponent=IndexComponent[IndexComponent['000905.SH']!=0]

    if(indexname==1000):
        IndexComponent=readLocalData('CSI1000Component.txt').dropna().reset_index(drop=True)
        IndexComponent=IndexComponent[IndexComponent['000852.SH']!=0]

    IndexComponent['time']=IndexComponent['time'].apply(lambda x:int(str(x)[:6]))
    IndexComponent=IndexComponent.drop_duplicates(subset=['time','code'],keep='last')
    return IndexComponent

#筛选股票 
def setStockPool(DF,DFfilter):
    """
    功能：将DFfilter中的股票在DF中筛选出来
    stockpool = ["000300.SH",'000905.SH','000985.SH']
    返回：筛选后的DF  columns=[code,date,xx]
    """
    DF=DF.merge(DFfilter[['time','code']],on=['code','time'])
    return DF

#得到申万行业分类数据 
@functools.lru_cache()
def getSWIndustryData(level=1,freq='month',fill=False):
    """
    功能：获取申万行业分类数据
    输入：行业分类级数，默认为一级，可以改为二级
        频率，默认为月
        fill是否填充空值，默认为否
    输出：dataframe包含时间、股票和对应行业分类信息
    """
    ind_data=pd.read_feather(Datapath+'BasicFactor_Swind_Component.txt').set_index('time')
    if(freq=='month'):
        ind_data=ind_data.pipe(applyindex,toTime).resample('m').last().pipe(applyindex,fromTime,'%Y%m')
        
    ind_data=ind_data.stack().reset_index()
    ind_data.columns=['time','code','SWind']
    if(level==1):
        ind_data['SWind']=ind_data['SWind'].apply(lambda x:str(x)[:4])
    if(level==2):
        ind_data['SWind']=ind_data['SWind'].apply(lambda x:str(x)[:6])
    if(fill==False):
        return ind_data
    ind_data=ind_data.sort_values(by='time')
    ind_data=ind_data.pivot(index='time',columns='code',values=['SWind'])
    ind_data=ind_data.reindex(getTradeDateList()).fillna(method='ffill').stack().reset_index()
    return ind_data
#加上行业数据列 
def addSWIndustry(DF,level=1,freq='month'):
    """
    功能：给数据dataframe加上行业数据列
    输入：已有数据dataframe
        行业分类级数，默认为一级，可改为二级
        频率，默认为月
    输出：加上申万行业数据的dataframe
    """
    SWData=getSWIndustryData(level,freq=freq)
    DF=DF.merge(SWData,on=['time','code'],how='left')
    return DF

#加上板块信息

def getSWSector(freq='month'):
    '''
    功能：获取申万板块信息
    输入：数据频率
    输出：申万板块数据dataframe
    '''
    sector = pd.read_csv(filepathtestdata+'sw1.csv')[['申万代码','板块']]
    swind=getSWIndustryData(freq=freq).merge(sector, left_on='SWind', right_on='申万代码')
    swind=swind.drop(['SWind','申万代码'], axis=1).rename(columns={'板块': 'sector'})
    if(freq=='month'):
        swind=swind.pivot(index='time',columns='code',values='sector')        
        swind=swind.pipe(applyindex,toTime).resample('m').last().pipe(applyindex,fromTime,'%Y%m')
        swind=swind.stack().reset_index().rename(columns={0: 'sector'})
    return swind

def addSWSector(DF, freq='month'):
    '''
    功能：给现有数据dataframe加上申万板块数据
    输入：DF: 三列标准型
        freq: 'day', 'month' 可选日频月频
    输出：加上申万板块数据后的dataframe
    '''
    sector=getSWSector(freq=freq)
    DF = DF.merge(sector, on=['time', 'code'], how='left')
    return DF


def getIndRetData():
    '''
    功能：获取行业收益信息
    输入：无
    输出：行业收益dataframe三列式
    '''
    ret=read_feather(Datapath+'BasicFactorSW_S_dq_Close.txt').set_index('time')
    ret=ret.stack().reset_index()
    ret.columns=['time','code','ret']
    ret=dayToMonth(ret)
    ret=ret.pipe(toShortForm).ffill()
    ret=ret/ret.shift(1)-1
    ret=ret.pipe(toLongForm)
    def shift1(x):
        x=int(x)-1
        if(x%100==0):
            x=x-88
        return x  
    ret['time']=ret['time'].apply(shift1)
    return ret

#合并大类行业  待改(直接改到数据更新模块)
def industryAggregate(IndInfo):
    """
        六个行业大类 消费 能源 周期 金融 TMT 其他
    """
    IndustryAggregationInfo=pd.Series(index=['纺织服装', '家用电器', '商业贸易', '农林牧渔', '食品饮料', '休闲服务', '医药生物','银行','非银金融','钢铁','有色金属','采掘','化工','建筑材料','建筑装饰','电气设备','机械设备','轻工制造','国防军工','汽车','公用事业','电子','通信','计算机','传媒','综合','房地产','交通运输'])
    IndustryAggregationInfo[['纺织服装', '家用电器', '商业贸易', '农林牧渔', '食品饮料', '休闲服务', '医药生物']]='消费'
    IndustryAggregationInfo[['钢铁','有色金属','采掘']]='能源冶金'
    IndustryAggregationInfo[['化工','建筑材料','建筑装饰','电气设备','机械设备','轻工制造','国防军工','汽车','公用事业']]='工业周期'
    IndustryAggregationInfo[['银行','非银金融']]='金融'
    IndustryAggregationInfo[['电子','通信','计算机','传媒']]='TMT'
    IndustryAggregationInfo[['综合','房地产','交通运输']]='其它'
    for ind in IndustryAggregationInfo.index:
        IndInfo.loc[IndInfo['SW1']==ind,'IndC']=IndustryAggregationInfo[ind]
    del IndInfo['SW1']
    return IndInfo


#读取barra因子
def getBarraData(startdate='',enddate='',freq='month'):
    """
    功能：读取Barra因子
    输入：开始日期和截止日期，startdate 格式例20200731 int
         freq频率，默认为月频
    输出：Barra因子DataFrame，列为 time code barra_factor (注:日频)
    """
    BarraData = read_hdf5(filepathtestdata + 'FactorLoading_Style.h5')
    BarraDataDF = pd.DataFrame([], columns=['time', 'code'])
    for fac in BarraData.keys():
        factorData = BarraData[fac].stack().reset_index()
        BarraData[fac] = pd.DataFrame([])
        factorData.columns = ['time', 'code', fac]  
        if(startdate!=''):
            factorData=factorData[factorData.time>=startdate]
        if(enddate!=''):
            factorData=factorData[factorData.time<=enddate]
        fac = factorData.fillna(0)
        BarraDataDF = BarraDataDF.merge(factorData, on=['time', 'code'], how='outer')
        
    if(freq=='month'):
        BarraDataDF=dayToMonth(BarraDataDF)
        
    if(freq=='day'):
        BarraDataDF['time']=BarraDataDF['time'].apply(lambda x:int(x))
    return BarraDataDF

#模块转换——因子
def factorStack(factor,factorname):
    '''
    功能：将因子数据dataframe堆叠转化
    输入：factor因子矩阵，factorname:因子名,string
    输出：转化后的因子数据
    '''
    factor=factor.stack().reset_index()
    factor.columns=['time','code',factorname]
    factor['time']=factor['time'].apply(lambda x:int(str(x)))
    return factor

#剔除ST、上市60天以内、停牌股
def kickout(mer):
    '''
    功能：剔除ST、上市60天以内、停牌股
    输入：待剔除的数据dataframe
    输出：剔除后的dataframe
    '''
    ValidDF=read_feather(Datapath + 'BasicFactor_ValidDF.txt')
    ValidDF['time']=ValidDF['time'].apply(lambda x:int(str(x)[:6]))
    ValidDF=ValidDF.drop_duplicates(subset=['time'],keep='first').set_index('time').stack().reset_index()
    ValidDF.columns=['time','code','valid']
    mer = mer.merge(ValidDF, on=['code', 'time'])
    mer = mer[mer.valid == True]
    del mer['valid']
    return mer

def zscore(x):
    '''
    功能：求series的z-score值
    输入：pandas series
    输出：z-score值
    '''
    return (x-x.dropna().mean())/x.dropna().std()

def zscorefac(x,fac):
    '''
    功能：求dataframe数据中某因子的z-score值
    输入：待测数据dataframe
    输出：z-score改造后的数据dataframe
    '''
    x[fac]=(x[fac]-x[fac].dropna().mean())/x[fac].dropna().std()
    return x

def dePCT(x,fac,k1=0.01):
    '''
    功能：将数据dataframe中因子值上、下k分位数以外的值修改为上、下k分位数值
    输入：数据dataframe，待修改的因子值列名称，分位数（默认为1%）
    输出：修改极端值后的数据dataframe
    '''
    x.loc[x[fac]>x[fac].quantile(1-k1),fac]=x[fac].quantile(1-k1)
    x.loc[x[fac]<x[fac].quantile(k1),fac]=x[fac].quantile(k1)
    return x

def deSTD(x,fac,k1=2):
    '''
    功能：将数据dataframe中偏离平均值k倍标准差以上的值修改为偏离平均值k倍的值
    输入：数据dataframe，待修改的因子值列名称，k倍数（默认为2）
    输出：修改极端值后的数据dataframe
    '''
    x.loc[x[fac]>x[fac].mean()+x[fac].std()*k1,fac]=x[fac].mean()+x[fac].std()*k1
    x.loc[x[fac]<x[fac].mean()-x[fac].std()*k1,fac]=x[fac].mean()-x[fac].std()*k1
    return x
def deMAD(x,fac,k1=5):
    '''
    功能：将数据dataframe中偏离中位数值k倍MAD以上的值修改为偏离中位数值k倍MAD的值
    输入：数据dataframe，待修改的因子值列名称，k倍数（默认为5）
    输出：修改极端值后的数据dataframe
    '''
    xmedian=x[fac].median()
    newmad=(x[fac]-xmedian).abs().median()
    x[fac]=np.clip(x[fac],xmedian-k1*newmad,xmedian+k1*newmad)
    return x

def normBoxCox(x,fac):
    '''
    功能：将数据正态分布化
    输入：数据dataframe和待修改的因子列名称
    输出：修改后的数据dataframe
    '''
    if(x[fac].dropna().count()>1):
        x[fac]=scipy.stats.boxcox(x[fac]-x[fac].min()+1)[0]
    return x
    
    

def fillmean(x,fac):
    '''
    功能：以平均值填补数据dataframe中因子值列的空缺
    输入：数据dataframe，待填补的因子值列名称
    输出：修改后的dataframe
    '''
    x[fac]=x[fac].fillna(x.mean())
    return x

def fillmedian(x,fac):
    '''
    功能：以中位数值填补数据dataframe中因子值列的空缺
    输入：数据dataframe，待填补的因子值列名称
    输出：修改后的dataframe
    '''
    x[fac]=x[fac].fillna(x.median())
    return x
#因子初始化处理

def factorInit(factor):
    '''
    功能：因子初始化处理，去除极端值
    输入：待处理的因子数据
    输出：处理后的因子数据
    '''
    factor_list=getfactorname(factor,['time','code'])
    for facname in factor_list:
        factor=factor.groupby('time').apply(deMAD)
        factor=factor.groupby('time').apply(zscore,facname)
    return factor


#获得因子列表
def getfactorname(x,L=['code','time']):
    '''
	功能：获取因子名称
	输入：因子矩阵
	返回：因子名称的list  ['Ret20',...]
	'''
    factor_list=pd.Series(x.columns)
    for l in L:
        try:
            factor_list=factor_list[factor_list!=l]
        except:
            continue
    return factor_list.to_list()

#日频转换为月频率
def dayToMonth(DF,factor_list='',method='last'):
    """
    功能：将日频数据dataframe转化为月频
    输入：
        @param DF:待转化的数据dataframe
        @param method: 保留数据（月末值，平均值，当月总和）{'last','mean','sum'}
    输出：转化为月频后的数据dataframe
    """
    x=DF.copy()
    if(factor_list==''):
        factor_list=getfactorname(x)
    x=x.pivot(index='time',columns=['code'],values=factor_list)
    x_tmp=pd.DataFrame(index=['time','code']).T
    for fac in factor_list:
        x1=x[fac].reset_index()
        x1['time']=x1['time'].apply(lambda x:int(str(x)[:6]))
        if(method=='sum'):
            x1=x1.groupby('time').sum().stack().reset_index()
            x1.columns=['time','code',fac]
        elif(method=='last'):
            x1=x1.groupby('time').apply(lambda x:x.ffill().iloc[-1]).set_index('time').stack().reset_index()
            x1['time']=x1['time'].apply(lambda x:int(x))
            x1.columns=['time','code',fac]
        elif(method=='mean'):
            x1=x1.groupby('time').mean().stack().reset_index()
            x1.columns=['time','code',fac]
        elif(method=='first'):
            x1=x1.groupby('time').apply(lambda x:x.iloc[0]).set_index('time').stack().reset_index()
            x1['time']=x1['time'].apply(lambda x:int(x))
            x1.columns=['time','code',fac]
        else:
            raise Exception('没有找到该方法')
        x_tmp=x_tmp.merge(x1,on=['time','code'],how='outer')
    return x_tmp

#多空t分组,并判断各股票的指标在哪一组
def isinGroupT(x,factor_name,asc=True,t=5):
    '''
    功能：判断各股票的指标在多空分中的哪一组
    输入：数据dataframe，因子名称，升/降序，分组组数
    输出：因子指标转化为分组组数后的数据dataframe
    '''
    x[factor_name]=pd.qcut(x[factor_name], t,labels=False,duplicates='drop')+1
    if(asc==False):
        x[factor_name]=t-x[factor_name]+1
    return x

#判断各股票的指标是否在前k（k%）
def isinTopK(x,factor_name,asc=True,k=30):
    '''
    功能：判断各股票的指标知否在前k
    输入：数据dataframe，因子名称，升/降序，前k数量（默认为30）
    输出：因子指标转化为判断是否在前k（1表示在，0表示不在）后的数据dataframe
    '''
    num = int(x.shape[0] * k) if k < 1 else k
    if(x.shape[0] < k):
        x[factor_name] = 0
        return x
    x[factor_name] = x[factor_name].rank(ascending = asc)
    x[factor_name] = x[factor_name].apply(lambda y: 1 if y <= num else 0)
    return x

#组合业绩评估
def calcGroupRet(x,factor_name,RetData=getRetData(),MVweight=False):
    '''
    功能：计算组合收益
    输入：数据dataframe，因子名称，收益数据，是否按市值加权
    输出：组合收益dataframe，索引为时间，一列为因子名
    '''
    if(not('ret' in x.columns)):
        x = x.merge(RetData,on = ['time','code'])
    if(MVweight==False):
        y = x.groupby(['time', factor_name])['ret'].mean().reset_index()
    else:
        x = addXSize(x,norm = 'dont')
        y = x.groupby(['time', factor_name]).apply(lambda x:calcWeightedMean(x['ret'],x['CMV'])).reset_index()
        
    y = y.pivot(index = 'time', columns = factor_name)
    y.columns = y.columns.droplevel()
    return y


#计算胜率、赔率
def calcWRPL(x):
    '''
    功能：计算胜率、赔率
    输入：包含收益数据ret列的数据dataframe
    输出：包含胜率赔率两个数据的序列
    '''
    return pd.Series([x[x.ret>0].count().ret*1.0/x.count().ret,x[x.ret>0].mean().ret/x[x.ret<0].mean().ret*-0.1],index=['WR','PL'])

def calcGroupWR(x,by='group',RetData=getRetData()):
    '''
    功能：计算组合胜率和赔率
    输入：数据dataframe,
    '''
    xname = x.name
    if(not('ret' in x.columns)):
        x = x.merge(RetData, on=['time', 'code'], how='outer')
    x['ret']=x['ret']-x['ret'].median()
    x = x[x[by] != 0]
    t = len(x[by].unique())
    if (t == 0):
        return 0
    if (t == 1):
        groupWeight = [1]
    else:
        groupWeight = [-1+i*(2.0/(t-1)) for i in range(t)]
    Ans=x.groupby(by).apply(calcWRPL).sort_index(ascending=False)
    Ans['w']=groupWeight
    Ans.loc[Ans.w<0,'WR']=1-Ans.loc[Ans.w<0,'WR']
    Ans.loc[Ans.w<0,'PL']=1/Ans.loc[Ans.w<0,'PL']
    WR=(Ans['WR']*(Ans['w'].abs())).sum()/(Ans['w'].abs()).sum()
    PL=(Ans['PL']*(Ans['w'].abs())).sum()/(Ans['w'].abs()).sum()
    return pd.Series([WR,PL],name=xname,index=['WR','PL'])

    
#取残差
def calcResid(y,x,intercept=True,retBeta=False):
    '''
    功能：求回归残差
    输入：
       x、y必须是DataFrame 格式，不能是Series
       用公式快速求解，需要更多信息用FB.regress
       retBeta='True'
    输出：储存回归残差信息的序列
    '''    
    x=pd.DataFrame(x).reset_index(drop=True)
    y=pd.DataFrame(y).reset_index(drop=True)
    if(intercept==True):
        x['intercept']=1
    x['y']=y
    x=x.dropna()
    y1=np.matrix(x[['y']])
    x1=np.matrix(x.drop(columns='y'))
    beta=np.linalg.pinv(x1.T.dot(x1)).dot(x1.T).dot(y1)
    y_pred=x1*beta
    resid=y1-y_pred
    if(retBeta):
        return {'beta':pd.DataFrame(beta,index=pd.Series(x.drop(columns='y').columns))[0],'resid':np.array(resid)}
    return np.array(resid)

#加入sw行业
def addXSWindDum(DF,freq='month'):
    '''
    功能：加入月度申万行业信息
    输入：待修改的数据dataframe（月频）
    输出：加入后的数据dataframe
    '''
    ind_tmp=getSWIndustryData(freq=freq)
    if('SWind' in DF):
        del DF['SWind']
    DF=DF.merge(ind_tmp,on=['time','code'],how='left')
    return DF

#获取中信行业数据,待完善
def getZXIndustryData(level = 1, freq = "month",fill = False):
    """
        功能：获取中信行业分类数据
        输入：行业分类级数，默认为一级，可以改为二级和三级
            频率，默认为月
            fill是否填充空值，默认为否
        输出：dataframe包含时间、股票和对应行业分类信息
    """
    ind_data = pd.read_feather(Datapath + '').set_index('time')
    if (freq == 'month'):
        ind_data = ind_data.pipe(applyindex, toTime).resample('m').last().pipe(applyindex, fromTime, '%Y%m')
    ind_data = ind_data.stack().reset_index()
    ind_data.columns = ['time', 'code', 'ZX']
    if (level == 1):
        ind_data['ZX'] = ind_data['ZX'].apply(lambda x: str(x)[:2])
    if (level == 2):
        ind_data['ZX'] = ind_data['ZX'].apply(lambda x: str(x)[:4])
    if (level == 3):
        ind_data['ZX'] = ind_data['ZX'].apply(lambda x: str(x)[:6])
    if (fill == False):
        return ind_data
    ind_data = ind_data.sort_values(by='time')
    ind_data = ind_data.pivot(index='time', columns='code', values=['ZX'])
    ind_data = ind_data.reindex(getTradeDateList()).fillna(method='ffill').stack().reset_index()
    return ind_data

#加入中信行业
def addXZXind(DF,freq='month'):
    '''
    功能：将中信行业数据加入现有dataframe
    输入：现有标准三列dataframe
    输出：加入中信行业数据后的四列dataframe
    '''
    ind_tmp=getZXIndustryData(level=1,freq=freq,fill = False)
    if('ZXind' in DF):
        del DF['ZXind']
    DF=DF.merge(ind_tmp,on=['time','code'],how='left')
    return DF


@functools.lru_cache()
def getCMV(freq='month'):
    '''
    功能：获取市值数据
    输入：数据频率（默认为月）
    输出：标准三列dataframe包含市值数据
    '''
    Size_data=pd.read_feather(Datapath+'BasicFactor_DqMV.txt')
    if(freq=='month'):
        Size_data['time']=Size_data['time'].apply(lambda x:int(str(x)[:6]))
        Size_data=Size_data.fillna(method='ffill').drop_duplicates(['time'],keep='last')
    Size_data=Size_data.set_index('time')
    Size_data=Size_data.stack().reset_index()
    Size_data.columns=['time','code','CMV']
    return Size_data

def addXSize(DF,freq='month',norm='boxcox'):
    '''
    功能：加入月度规模信息
    输入：待修改的标准三列数据dataframe,freq默认月频，norm默认为boxcox标准化，可选dont不做标准化
    输出：加入后的数据四列dataframe
    '''
    Size_data=getCMV(freq=freq)
    def boxcoxcmv(x):
        if(len(x['CMV'].unique())>1):
            x['CMV']=scipy.stats.boxcox(x['CMV']+1)[0]
        return x
    if(norm=='boxcox'):
        Size_data=Size_data.groupby('time').apply(boxcoxcmv)
    if(norm=='ln'):
        Size_data['CMV']=Size_data['CMV'].apply(lambda x:np.log(x))
    if('CMV' in DF):
        del DF['CMV']
    DF=DF.merge(Size_data,on=['time','code'],how='left')
    return DF


def RegbyindSize(x,name):
    '''
    功能：计算现有因子和行业市值因子的回归残差，用于中性化
    输入：因子信息，因子名称，因子列表
    输出：回归残差
    '''
    def zscore(x):
        return (x-x.dropna().mean())/x.dropna().std()
    x_tmp=x[['time','code',name,'SWind','CMV']]
    x_tmp['CMV']=zscore(deMAD(x_tmp,'CMV')['CMV']).fillna(0)
    x_tmp=x_tmp.dropna()    
    if(x_tmp.shape[0]==0):
        x[name+'_neu']=np.nan
        return x
    y=x_tmp[[name]]

    x1=pd.get_dummies(x_tmp[['CMV','SWind']],'SWind')
    # x1=x_tmp[['CMV']]

    # for ind in x['SWind'].unique():
    #     x_tmp[ind]=0
    #     x_tmp.loc[x_tmp.SWind==ind,ind]=1
    #     x1[ind]=x_tmp[ind]  

    x_tmp[name+'_neu']=calcResid(y, x1)
    x=x.merge(x_tmp[['code',name+'_neu']],on='code',how='outer')
    return x

#计算行业市值中性
def calcNeuIndSize(factor,factor_name,freq='month'):
    '''
    功能：计算行业市值中性，排除行业市值对因子的影响
    输入：因子信息和因子名称，评率默认为月，可选日
    输出：中性化后的因子信息
    '''
    factor_tmp=factor.copy()
    if('CMV' not in factor):
        factor_tmp=addXSize(factor_tmp,freq=freq)
    if('SWind' not in factor):
        factor_tmp=addXSWindDum(factor_tmp,freq=freq)
    if(factor_name+'_neu' in factor_tmp):
        del factor_tmp[factor_name+'_neu']
    factor_tmp=factor_tmp.groupby('time').apply(RegbyindSize,factor_name).reset_index(drop=True)
    if(factor_name+'_neu' in factor):
        factor=factor.drop(columns=factor_name+'_neu')
    factor=factor.merge(factor_tmp[['time','code',factor_name+'_neu']],on=['time','code'])
    return factor



def RegbySize(x,name):
    '''
    功能：计算现有因子和行业市值因子的回归残差，用于中性化
    输入：因子信息，因子名称
    输出：回归残差
    '''
    def zscore(x):
        return (x-x.dropna().mean())/x.dropna().std()
    x_tmp=x[['time','code',name,'CMV']]
    x_tmp['CMV']=zscore(deMAD(x_tmp,'CMV')['CMV']).fillna(0)
    x_tmp=x_tmp.dropna()    
    if(x_tmp.shape[0]==0):
        x[name+'_neu']=np.nan
        return x
    y=x_tmp[[name]]
    x1=x_tmp[['CMV']]
    x1['const']=1    
    x_tmp[name+'_neu']=calcResid(y, x1)
    x=x.merge(x_tmp[['code',name+'_neu']],on='code',how='outer')
    return x

#计算市值中性
def calcNeuSize(factor,factor_name,freq='month'):
    '''
    功能：计算市值中性，排除市值对因子的影响
    输入：因子信息和因子名称，数据频率（默认为月）
    输出：中性化后的因子信息
    '''
    factor_tmp=factor.copy()
    if('CMV' not in factor):
        factor_tmp=addXSize(factor_tmp,freq=freq)
    if(factor_name+'_neu' in factor_tmp):
        del factor_tmp[factor_name+'_neu']
    factor_tmp=factor_tmp.groupby('time').apply(RegbySize,factor_name).reset_index(drop=True)
    if(factor_name+'_neu' in factor):
        factor=factor.drop(columns=factor_name+'_neu')
    factor=factor.merge(factor_tmp[['time','code',factor_name+'_neu']],on=['time','code'])
    return factor



def addXBarra(DF,Barra_list='',freq='month'):
    '''
    功能：加入Barra因子数据
    输入：待修改的数据dataframe，barra因子列表，数据频率（默认为月度）
    输出：加入后的数据dataframe
    '''
    barra_tmp=getBarraData(freq=freq)
    if(Barra_list==''):
        Barra_list=getfactorname(barra_tmp)        
    if(type(Barra_list)==str):
        Barra_list=[Barra_list]
    DF=DF.merge(barra_tmp[['time','code']+Barra_list],on=['time','code'],how='left')
    return DF,Barra_list

def RegbyBarra(x,name,factor_list):
    '''
    功能：计算现有因子和barra因子的回归残差，用于中性化
    输入：因子信息，barra因子名称，因子列表
    输出：回归残差
    '''
    #代码补齐
    def zscore(x):
        return (x-x.dropna().mean())/x.dropna().std()
    x_tmp=x[['time','code','SWind',name]+factor_list]
    for fac in factor_list:
        x_tmp[fac]=zscore(deMAD(x_tmp,fac)[fac]).fillna(0)
    x_tmp=x_tmp.dropna()    
    if(x_tmp.shape[0]==0):
        x[name+'_pure']=np.nan
        return x
    y=x_tmp[[name]]
    x1=pd.get_dummies(x_tmp[factor_list+['SWind']],'SWind')
    # x1=x_tmp[factor_list]
    # for ind in x['SWind'].unique():
    #     x_tmp[ind]=0
    #     x_tmp.loc[x_tmp.SWind==ind,ind]=1
    #     x1[ind]=x_tmp[ind]  
    x_tmp[name+'_pure']=calcResid(y, x1)
    x=x.merge(x_tmp[['code',name+'_pure']],on='code',how='outer')
    return x

#纯因子
def calcNeuBarra(factor,factor_name,factor_list='',freq='month'):
    '''
    功能：计算Barra因子中性，排除barra因子对待处理因子的影响
    输入：因子信息，因子名称，barra因子列表，数据频率（默认为月）
    输出：中性化后的因子信息
    '''
    factor_tmp,factor_list=addXBarra(factor,Barra_list=factor_list,freq=freq)
    if('SWind' not in factor):
        factor_tmp=addXSWindDum(factor_tmp,freq=freq)
    if(factor_name+'_pure' in factor_tmp):
        del factor_tmp[factor_name+'_pure']
    factor_tmp=factor_tmp.groupby('time').apply(RegbyBarra,factor_name,factor_list).reset_index(drop=True)
    if(factor_name+'_pure' in factor):
        factor=factor.drop(columns=factor_name+'_pure')
    factor=factor.merge(factor_tmp[['time','code',factor_name+'_pure']],on=['time','code'])
    return factor

def calcIC(mer, factor_name):
    '''
    功能：用pearsonr和spearmanr方法计算相关性（IC、rankIC）
    输入：包含因子值和回报的数据dataframe，因子名称
    输出：包含IC,ICIR,rankIC,rankICIR,t_value的序列
    '''
    sp1 = mer.groupby('time').apply(lambda x: stats.pearsonr(x['ret'], x[factor_name])[0])
    IC = sp1.mean()
    ICIR = (sp1.mean()) / sp1.std() * 12 ** 0.5
   
    sp2 = mer.groupby('time').apply(lambda x: stats.spearmanr(x['ret'], x[factor_name])[0])
    rankIC = sp2.mean()
    rankICIR = (sp2.mean()) / sp2.std() * 12 ** 0.5
    t_v = sp2.mean() / sp2.std() * (sp2.count() - 1) ** 0.5
    return sp1,sp2,pd.Series([IC,ICIR,rankIC,rankICIR, t_v],index=['IC:','ICIR:','rankIC:','rankICIR:','t_value:'])

#计算组合业绩

def calcIClist(DF):
    IC = sp.mean()
    ICIR =x

def ZipLocalFiles(file_path,save_path,zipname='',t=5):
    '''
    功能：将本地文件打包
    输入：t是分成的压缩包数量默认为5,file_path是需要压缩的文件夹路径,save_path是储存路径
    '''
    def zip_files(file_path,files, zip_name ):
        zip = zipfile.ZipFile( zip_name, 'w', zipfile.ZIP_DEFLATED )
        for file in files:
            filefull=os.path.join(file_path,file)
            zip.write(filefull,file)
        zip.close()
    def get_file_list(file_path): #获取按时间顺序排列的文件名
        dir_list = os.listdir(file_path)
        dir_list = sorted(dir_list,key=lambda x: os.path.getmtime(os.path.join(file_path, x)))
        return dir_list
    def getFileSize(file_path,size=0): #获取文件夹大小
        file_list=get_file_list(file_path)
        for f in file_list:
            size += os.path.getsize(file_path+f)
        return size
    ff=[]
    fff=[]
    file_list=get_file_list(file_path)
    size=0
    for f in file_list:
        size += os.path.getsize(file_path+f)
        ff.append(f)
        if(t>1):
            if size>(int(getFileSize(file_path,size=0))/(t-1)):
                ff.remove(f)
                size=os.path.getsize(file_path+f)
                fff.append(ff)
                ff=[]
                ff.append(f)
    fff.append(ff)
    
    for i in range(len(fff)):
        files = fff[i]
        zip_name =save_path+zipname+datetime.datetime.now().strftime('%Y%m%d')+'('+str(i+1)+')'+'.zip'#压缩包名字(%Y%m%d(i))
        zip_files(file_path,files, zip_name)
        


def copyFile(file_path,target_path):
    '''
    功能：复制本地文件
    输入：待复制文件路径，文件粘贴路径
    '''
    if not os.path.isfile(file_path):
        print ("%s 不存在!"%(file_path))
    else:
        if not os.path.exists(target_path):
            os.makedirs(target_path)     
        fpath,fname=os.path.split(file_path)                    
        shutil.copy(file_path, target_path+'//'+ fname)          


def _calcGrowthRate(x,t=3):
    '''
    功能：利用最小二乘法回归计算年成长率
    输入：x为矩阵形式数据，t为窗口年数
    输出：回归得到的成长率值
    '''
    ols=po.PandasRollingOLS(y=pd.DataFrame(x),x=pd.DataFrame(np.arange(1, len(x)+1)),use_const=True,window=t)
    return pd.Series(ols.beta[0]/x.abs().rolling(window=t).mean(),name=x.name)



def calcGrowthRate(data,window=3):
    '''
    功能：计算过去windows年复合变化率（同比）
    计算公式 regress(data,(1,2,...),intercept=True).beta / y.abs().mean()
    输入：data为矩阵形式，window为窗口期（年）（默认为3）
    输出：数据矩阵，每一个值为变化率
    '''
    
    tmp=data.dropna(how='all').reset_index()
    tmp['season']=tmp['time'].apply(lambda x:str(x)[-4:])
    tmp=tmp.set_index('time')
    tqdm.pandas()
    tmp1=tmp.groupby('season')
    ans=[]
    for i in tqdm(tmp.season.unique()):
        ans.append(tmp1.get_group(i).drop(columns='season').apply(_calcGrowthRate))
    tmp=pd.concat(ans).sort_index()
    return tmp

def copyFiles(A,B,cover=True):
    '''
    功能：复制文件夹
    输入：A为源文件路径，B为新文件路径
    '''
    source_path= os.path.abspath(A)
    target_path = os.path.abspath(B)
    if cover:       
        if os.path.exists(target_path):
            # 如果目标路径存在原文件夹的话就先删除
            shutil.rmtree(target_path)  
        shutil.copytree(source_path, target_path)
    else:
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        if os.path.exists(source_path):
            for root, dirs, files in os.walk(source_path):
                for file in files:
                    src_file = os.path.join(root, file)
                    shutil.copy(src_file, target_path)
                    print(src_file)
                for dirss in dirs:
                    dir_file=os.path.join(root,dirss)
                    if os.path.exists(target_path+'//'+dirss):
                        shutil.rmtree(target_path+'//'+dirss)  
                    shutil.copytree(dir_file,target_path+'//'+dirss)

def calcAllFactorAns():
    '''
    功能：计算所有储存在factorInfo文件中的因子的收益、IC等信息
    '''
    import FactorTest.FactorTestMain as FM
    datainfo=pd.read_excel(FactorInfopath)
    for i in tqdm(datainfo.index):
       Test=FM.FactorTest()
       Test.getFactor(read_feather(Factorpath+datainfo.loc[i,'地址']))
       Test.autotest(datainfo.loc[i,'因子名称'],asc=False)
       datainfo.loc[i,'IC']=Test.ICAns[datainfo.loc[i,'因子名称']]['IC:']
       datainfo.loc[i,'ICIR']=Test.ICAns[datainfo.loc[i,'因子名称']]['ICIR:']
       FacDF=Test.portfolioList[datainfo.loc[i,'因子名称']]
       if(datainfo.loc[i,'IC']<0):
           FacDF['多空组合']=FacDF['多空组合']*-1
       datainfo.loc[i,'多空年化收益']=calcPortfolioRet(FacDF['多空组合'])['年化收益率:']
       datainfo.loc[i,'多空信息比率']=calcPortfolioRet(FacDF['多空组合'])['信息比率:']
       FacDF['IC']=Test.ICList[datainfo.loc[i,'因子名称']]
       FacDF.columns=pd.Series(FacDF.columns).apply(lambda x:str(x))
       FacDF.reset_index().to_feather(Factorpath+'plotdata/'+datainfo.loc[i,'因子名称']+'.fth')
    datainfo.to_excel(FactorInfopath, index=False)

def calcPortfolioRet(Rev_seq,t=12):
    """
    功能：计算组合收益率
    输入：
        @param Rev_seq:组合收益序列
        @param t:天数 默认12(月) 252 日
    输出：组合收益序列数据，包含年化收益率，信息比率，胜率，最大回撤
    """
    ret_mean=e**(Rev_seq.apply(lambda x:np.log(x+1)).mean()*t)-1
    ret_sharpe=Rev_seq.mean()*t/Rev_seq.std()/t**0.5
    ret_winrate=Rev_seq[Rev_seq>0].count()/Rev_seq.count()
    Rev_list=(Rev_seq+1).cumprod()
    ret_maxloss=(Rev_list/Rev_list.cummax()-1).min()
    return pd.Series([ret_mean,ret_sharpe,ret_winrate,ret_maxloss],index=['年化收益率:','信息比率:','胜率:','最大回撤:'])

#计算换手率
def calcAnnualTurnover(GroupData,facname,t=12):
    '''
    功能：获取年度换手率信息
    输入：三列标准dataframe，因子名，计算周期
    输出：各个因子下的年度换手率
    '''
    groupdata=GroupData
    groupNum = groupdata[['time', 'code', facname]].groupby(['time', facname]).count().reset_index()
    groupNum['weight'] = 1 / groupNum['code']
    groupdata = groupdata.merge(groupNum[['time', facname, 'weight']], how='outer')
    groupList = groupdata[facname].unique()
    annual_turnover = pd.Series(index=groupList)
    for i in groupList:
        component = groupdata.copy()
        component['weight'] = component['weight'].where(component[facname] == i, 0)
        component['lag_weight'] = component.groupby('code')['weight'].shift(1).fillna(0)
        component = component[component['time'] != component['time'].unique()[0]]
        component['deviate'] = (component['weight'] - component['lag_weight']).abs()
        avg_dev = (component.groupby('time')['deviate'].sum()/2).mean() * t
        annual_turnover[i] = avg_dev
    return annual_turnover

# 更新数据库信息
def DataRenewTime():
    '''
    功能：更新dataInfo.csv中的信息（最新数据时间）
    '''
    datainfo = pd.read_excel(DataInfopath)
    for i in tqdm(datainfo['存储地址'].unique()):
        if (os.path.exists(Datapath + i)):
            datainfo.loc[datainfo['存储地址']==i, '最新时间'] = int(read_feather(Datapath + i)['time'].max())
        else:
            datainfo.loc[datainfo['存储地址']==i, '最新时间'] = np.nan
    datainfo.to_excel(DataInfopath, index=False)

def FactorRenewTime():
    '''
    功能：更新因子库（FactorInfo.xlsx)信息（最新因子数据时间）
    '''
    datainfo = pd.read_excel(FactorInfopath)
    for i in datainfo.index:
        if (os.path.exists(Factorpath + datainfo.loc[i, '地址'])):
            datainfo.loc[i, '最新时间'] = int(read_feather(Factorpath + datainfo.loc[i, '地址'])['time'].max())
        else:
            datainfo.loc[i, '最新时间'] = np.nan
    datainfo.to_excel(FactorInfopath, index=False)

# 更新factor最新信息
def getUpdateStartTime(x, backdays=0):
    '''
    功能：获得开始更新时间
    输入：x为infoDF中的最新时间，backdays是回退天数，日频使用1日，更低频率anndt等使用5日
    输出：开始更新时间
    '''
    if(type(x)!=pd.core.series.Series):
        x=pd.Series(x)
    if (x.count() != len(x)):
        return g_starttime
    else:
        return int((datetime.datetime.strptime(str(int(x.min())), '%Y%m%d') - datetime.timedelta(days=backdays)).strftime('%Y%m%d'))

    

def readLocalData(rawpath,key=''):
    """
    功能：读取本地数据，不需要准备Datapath， 读入自动转化为['time','code',key]
    输入：数据路径，键名
    输出：数据dataframe
    """
    path=Datapath+rawpath
    if(os.path.exists(path)):
        data=read_feather(path)
        data=data.set_index('time')
        data=data.unstack().reset_index()
        if(key==''):
            datainfo=pd.read_excel(DataInfopath)
            key=datainfo[datainfo['存储地址']==rawpath].iloc[0]['数据库键']
        data.columns=['code','time',key]
        data=data[['time','code',key]]
    else:
        data=pd.DataFrame(index=['time','code']).T
    return data

#批量读入
def readLocalDataSet(path_list):
    '''
    功能：批量读入信息
    输入：待读入的文件路径列表['A','B','C']
    输出：读入的信息dataframe
    '''
    data_tot=pd.DataFrame(index=['time','code']).T
    for path in path_list:
       data_tot=data_tot.merge(readLocalData(path),on=['time','code'],how='outer') 
    return data_tot

#合并新旧信息（废弃）
def mergeAB(rawData,newData):
    """
    #检查新信息最新行的空值率
    #    if(newData[newData.time==newData.time.max()].groupby('time').count().iloc[-1,-1]==0):
    #        print( Exception('最后一行为空'))
    #目前采用后值覆盖方法，假定后值正确
    """
    if(rawData.shape[0]>0):
        rawData0=rawData[rawData.time<newData['time'].min()]
        rawData=rawData[rawData.time>=newData['time'].min()]
    else:
        rawData0=rawData
    rawData=rawData.append(newData).drop_duplicates(subset=['time','code'],keep='last')
    rawData=rawData0.append(rawData)
    return rawData

def readLocalFeather(path):
    '''
    功能：读取本地feather文件，无需加入Datapath
    输入：feather文件路径
    输出：包含feather文件信息的dataframe
    '''
    if(os.path.exists(Datapath+path)):
        return read_feather(Datapath+path)
    else:
        return pd.DataFrame(index=['time']).T


#用于存储sql型数据  以object_ID为单位
def saveSqlData(sqlData,infoDF):
    '''
    功能：储存sql型数据到infoDF中
    输入：sql型数据和infoDF
    输出：无
    '''
    path=Datapath+infoDF.loc[infoDF.index[0],'存储地址']
    if(os.path.exists(path)):
        data=read_feather(path)
    else:
        data=pd.DataFrame(index=['id','time']).T  
    sqlData['time']=sqlData['time'].apply(lambda x:int(x))
    data0=data[data.time<sqlData['time'].min()]
    
    data=data[data.time>=sqlData['time'].min()]
    data=data.append(sqlData).drop_duplicates(subset=['id'],keep='last').sort_values(by='time')
    data0=data0.append(data)
    save_feather(Datapath+infoDF.loc[infoDF.index[0],'存储地址'],data0)

#存储日频数据
def saveDailyData(sqlData,infoDF):
    '''
    功能：储存日频数据为feather文件
    输入：sql数据和因子信息dataframe
    输出：无
    '''
    for i in infoDF.index:
        sql_data1=sqlData[['time','code',infoDF.loc[i,'数据库键']]].pivot(index='time',columns='code',values=infoDF.loc[i,'数据库键']).reset_index()
        data0=readLocalFeather(infoDF.loc[i,'存储地址'])
        data0=data0.append(sql_data1)
        data0['time']=data0['time'].apply(lambda x:int(x))
        if('code' in data0):
            del data0['code']
        save_feather(Datapath+infoDF.loc[i,'存储地址'],data0.drop_duplicates(subset='time',keep='last').sort_values(by='time').set_index('time').reset_index())

#用于存储财务数据
def saveFinData(sqlData,infoDF):
    '''
    功能：储存财务数据为feather文件
    输入：sql数据和因子信息dataframe
    输出：无
    '''
    for i in tqdm(infoDF.index):
        sql_data1=sqlData[['time','code',infoDF.loc[i,'数据库键']]].drop_duplicates(subset=['time','code'],keep='last').pivot(index='time',columns='code',values=infoDF.loc[i,'数据库键'])
        data0=readLocalFeather(infoDF.loc[i,'存储地址']).set_index('time',drop=True)
        data0=data0.append(sql_data1).reset_index()
        data0['time']=data0['time'].apply(lambda x:int(x))
        data0=data0.drop_duplicates(subset='time',keep='last')
        if('code' in data0):
            del data0['code']
        save_feather(Datapath+infoDF.loc[i,'存储地址'],data0.set_index('time').reset_index())
#用于存储行业成分股数据
def saveIndData(sqlData,infoDF):
    '''
    功能：储存行业数据为feather文件
    输入：sql数据和因子信息dataframe
    输出：无
    '''
    for i in infoDF.index:
         sqlData1=sqlData[['time','code',infoDF.loc[i,'数据库键']]]
         sqlData2=sqlData[['eddate','code']].dropna()
         sqlData2.columns=['time','code']
         sqlData2['out']=-1
         sqlData=sqlData1.merge(sqlData2,on=['time','code'],how='outer').drop_duplicates(subset=['time','code'],keep='first')     
         sqlData.loc[sqlData[infoDF.loc[i,'数据库键']]!=sqlData[infoDF.loc[i,'数据库键']],infoDF.loc[i,'数据库键']]=sqlData.loc[sqlData[infoDF.loc[i,'数据库键']]!=sqlData[infoDF.loc[i,'数据库键']],'out']
         del sqlData['out']
         
         data0=readLocalFeather(infoDF.loc[i,'存储地址']).set_index('time')
         data0=data0.append(sqlData.pivot(index='time',columns='code',values=infoDF.loc[i,'数据库键'])).reset_index()
         data0['time']=data0['time'].apply(lambda x:int(x))
         data0=data0.drop_duplicates(subset='time',keep='last').set_index('time').reindex(index=getTradeDateList(),columns=getStockList()).fillna(method='ffill').reset_index()
         data0.replace({-1:np.nan},inplace=True)
         if('code' in data0):
             del data0['code']
         save_feather(Datapath+infoDF.loc[i,'存储地址'],data0.set_index('time').reset_index())
         
#存储指数成分股
def saveIndexComponentData(sqlData,infoDF):
    '''
    功能：储存指数成分股为feather文件
    输入：sql数据和因子信息dataframe
    输出：无
    '''
    i=infoDF.index[0]
    sqlData.columns=['time','code','signal']        
    sqlData['time']=sqlData['time'].apply(lambda x:x.strftime('%Y%m%d'))
    data0=readLocalFeather(infoDF.loc[i,'存储地址']).set_index('time')
    data0=data0.append(sqlData.pivot(index='time',columns='code',values='signal')).reset_index()
    data0['time']=data0['time'].apply(lambda x:int(x))
    data0=data0.drop_duplicates(subset='time',keep='last').set_index('time').fillna(method='ffill').reindex(index=getTradeDateList(),columns=getStockList()).fillna(method='ffill').reset_index()  
    data0.replace({'剔除':0,'纳入':1},inplace=True)
    if('code' in data0):
        del data0['code']
    save_feather(Datapath+infoDF.loc[i,'存储地址'],data0.set_index('time').reset_index())


def toShortForm(DF,faclist=''):
    '''
    功能：将dataframe转化为time，code两列的类数据库形式
    输入：数据dataframe
    输出：转化后的数据dataframe
    '''
    if(faclist==''):
        faclist=getfactorname(DF,['code','time'])
    if(type(faclist)=='str'):
        faclist=[faclist]
    DF1=DF.drop_duplicates(subset=['time','code'],keep='last').pivot(index='time',columns='code',values=faclist)
    return DF1

def toLongForm(DF,facname=''):
    '''
    功能：将dataframe进行堆叠，列旋转到行
    输入：数据dataframe
    输出：转化后的数据dataframe
    '''
    DF=DF.stack().reset_index()
    if(facname!=''):
        DF.columns=['time','code',facname]
    return DF


def fillFisicalMonth(DF,faclist):
    '''
    功能：用财务月对数据Dataframe进行索引，用法：DF.pipe(FB.fillFisicalMonth,faclist)
    输入：数据dataframe,因子列表
    输出：用财务月作索引的数据dataframe
    '''
    if(type(faclist)=='str'):
        faclist=[faclist]
    DF1=DF.drop_duplicates(subset=['time','code'],keep='last').pivot(index='time',columns='code',values=faclist).reindex(index=getTradeDateList('month'))
    return DF1.fillna(method='ffill').stack().reset_index().dropna()


def partition(ls, size):
    '''
    功能：将列表切片再合成为固定长度的嵌套列表
    输入：待处理列表和固定长度
    输出：处理后的嵌套列表
        >>> partition([1, 2, 3, 4], 3)
        [[1, 2, 3], [4]]
    '''
    return [ls[i:i+size] for i in range(0, len(ls), size)]
         
         

#装饰器——————————————————————————————————————————————————————
#测时
def testTime(func):
    def test(*args,**kwargs):
        starttime=time.time()
        func(*args,**kwargs)
        endtime=time.time()
        print('用时：',round((endtime-starttime) * 1000),'ms')
    return test

#    @staticmethod  内置函数
#    @property  1. 方法——》属性    2. self._A=1    设置self.A() return self._A  防止_A被修改
#    @classmethod 通过类名直接调用函数


def transData(data,key='factor',ANN_DT='BasicFactor_AShareFinancialIndicator_ANN_DT.txt', output='', startmonth=199912, endmonth=210012):
    '''
    功能：将财务数据转化为时间、股票代码、因子名称三列dataframe或矩阵
        即将更名transFisicalData函数
    输入：
        data为因子矩阵,需预先getFisicalList()处理
        key因子名,output为输出形式（三列或矩阵型）
        ANN_DT为公布日期文件地址，默认'BasicFactor_AShareFinancialIndicator_ANN_DT.txt'；可选三大表ANN_DT
        startmonth和endmonth为int型，表示开始和结束月份
    '''

    ANN_DT=read_feather(Datapath + ANN_DT).set_index('time').reindex(getFisicalList())
    ANN_DT.index.name='time'
    ANN_DT.columns.name='code'
    ANN_DT = ANN_DT.stack().reset_index().set_index('code').astype(int).reset_index()
    if(data.index.name != 'time'):
        data=data.set_index('time')
    data.columns.name='code'
    data=data.stack().reset_index()
    factorDF=pd.merge(ANN_DT,data,on=['time','code'])
    factorDF.columns=['code','time','ANN_DT',key]
    monthlist=getTradeDateList('month')
    factorDF['rtime']=factorDF['ANN_DT'].apply(lambda x:int(str(x)[:6]))
    factorDF=factorDF.drop_duplicates(subset=['code','rtime'],keep='last')
    factorDF_loc=factorDF.pivot(index='rtime',columns='code',values= key)
    factorDF_loc=factorDF_loc.reindex(monthlist[monthlist>=factorDF_loc.index[0]]).ffill()
    factorDF=factorDF_loc.stack().reset_index()
    factorDF.rename(columns={0:key},inplace=True)
    factorDF = factorDF[factorDF.time >= startmonth]
    factorDF = factorDF[factorDF.time <= endmonth]
    if output=='matrix':
        return factorDF.pivot(index='time',columns='code',values=key)
    else:
        return factorDF



def transFisicalData(data,key='factor',ANN_DT='BasicFactor_AShareFinancialIndicator_ANN_DT.txt', output='', startmonth=199912, endmonth=210012):
    '''
    原函数transData
    data为因子矩阵,需预先getFisicalList()处理
    key因子名,output为输出形式（三列或矩阵型）
    ANN_DT为公布日期文件地址，默认'BasicFactor_AShareFinancialIndicator_ANN_DT.txt'；可选三大表ANN_DT
    '''

    return transData(data,key,ANN_DT,output,startmonth,endmonth)


def calc_plot(DF):
    '''
    功能：对dataframe进行plot画图
    输入：用于画图的dataframe
    '''
    # DF=DF.reset_index()
    # DF['time']=DF['time'].apply(toTime)
    # DF.set_index('time').plot()
    applyindex(DF,lambda x:str(x)).plot()

def calcFisicalLYR(DF):
    '''
    功能：将3、6、9月末财务数据时间统一为上年末数据
    输入：三列标准矩阵
    输出：修改后的三列标准矩阵
    '''
    DF=DF.loc[getFisicalList('year')].fillna('empty').reindex(getFisicalList()).ffill()
    DF[DF=='empty']=np.nan
    return DF

def calcFisicalq(DF):
    '''
    功能：将6、9、12月末的当年累积时段财务数据转化为当季度内的数据
    输入：三列标准矩阵
    输出：修改后的三列标准矩阵
    '''
    DFlast=DF.copy()
    DFlast.loc[getFisicalList('year')[getFisicalList('year').isin(DFlast.index)]]=0
    DFlast=DFlast.shift(1)
    return (DF-DFlast)

def calcFisicalttm(DF):    
    '''
    功能：将季度时段财务数据转化为过去四个季度的和（ttm数据）
    输入：三列标准矩阵
    输出：修改后的三列标准矩阵
    '''
    DFq=calcFisicalq(DF)
    return DFq.rolling(window=4).sum()

def applyindex(x,func,*args,**kwargs):
    '''
    功能：直接对DataFrame或Series的index执行func
    输入：x:dataframe或series， func，*args，**kwargs：要执行的function和辅助参数
    输出：修改完成的dataframe或series
    '''
    if(x.index.name==None):
        x.index.name='index'
    xname=x.index.name
    x=x.reset_index()
    x[xname]=x[xname].apply(lambda x:func(x,*args,**kwargs))
    return x.set_index(xname)

# 半衰期序列
def calc_exp_list(window,half_life):
    '''
    功能：根据因子半衰期生成权重序列
    输入：window为int表示窗口期，half_life为半衰期序列
    输出：由半衰期计算的权重序列
    '''
    exp_wt = np.asarray([0.5 ** (1 / half_life)] * window) ** np.arange(window)
    return exp_wt[::-1] / np.sum(exp_wt)

#weighted_std
def calcWeightedStd(series, weights):
    '''
    功能：计算加权标准差
    输入：待计算序列和权重
    输出：加权标准差值
    '''
    weights /= np.sum(weights)
    return np.sqrt(np.sum((series-np.mean(series)) ** 2 * weights))


def calcWeightedMean(series,weight):
    '''
    功能：计算加权平均
    输入：待计算序列和权重
    输出：加权平均值
    '''
    return np.sum(series*weight)/np.sum(weight)

#滚动回归(待完善)
def rollingRegress(y,x,window,const=True):
    try:
        ols=po.PandasRollingOLS(y,x,hasconst=const,window=window)
        return pd.concat([alpha,beta],axis=1)
    except:
        ...
        
def monthToDay(DF,factor_list=''):
    '''
·   功能：将月频数据dataframe转化为日频数据dataframe，空缺数据向下补齐
    输入：DF为原月频数据dataframe（三列标准型），fator_list为需要转换的因子列表
    输出：改造后的数据dataframe（三列标准型）
    '''
    x=DF.copy()
    if(factor_list==''):
        factor_list=getfactorname(x)
    x=x.pivot(index='time',columns=['code'],values=factor_list)
    
    day=getTradeDateList()
    day.name='day'
    month=day.copy()
    month.name='month'
    month=month.apply(lambda x: int(str(x)[:6]))
    time=pd.concat([day, month], axis=1)
    time=time.drop_duplicates(subset=['month'], keep='last')  
    
    x_tmp=pd.DataFrame(index=['time','code']).T
    for fac in factor_list:
        x1=x[fac].reset_index()
        x1=pd.merge(x1, time, left_on='time', right_on='month', how='outer').set_index('day').drop(['time','month'],axis=1)
        x1=x1.reindex(day.values).ffill()
        x1=x1.stack().reset_index()
        x1.columns=['time','code',fac]
        x_tmp=x_tmp.merge(x1,on=['time','code'],how='outer')
    return x_tmp



def plotICList(IClist,RankICList,facname=''):
    '''
    功能：对多空组合的IC和rank IC进行绘图
    输入：IClist月度IC列表
        RankIClist月度IC列表
    '''
    IC_df = pd.DataFrame(IClist,columns=['IC'])
    RankIC_df = pd.DataFrame(RankICList,columns=['RankIC'])
    plt.figure(figsize=(15,10))
    ax1 = plt.subplot(221) 
    ax1.bar([str(i) for i in IC_df.index],IC_df['IC'],bottom = 0, \
            width=0.5 ,label='月度IC') 
    ax1.hlines(IC_df['IC'].mean(),str(IC_df.index[0]),\
                str(IC_df.index[-1]),linestyles='--',color='r',label='月度IC均值')
    ax1.set_xticks(range(0,len(IC_df),20))
    ax1.legend()
    ax1.set_title(facname+'月度IC序列')
    ax2 = plt.subplot(222) 
    ax2.bar([str(i) for i in RankIC_df.index],RankIC_df['RankIC'],\
            bottom = 0, width=0.5 ,label='月度RankIC') 
    ax2.hlines(RankIC_df['RankIC'].mean(),str(RankIC_df.index[0]),\
                str(RankIC_df.index[-1]),linestyles='--',color='r',label='月度RankIC均值')
    ax2.set_title(facname+'月度RankIC序列')
    ax2.set_xticks(range(0,len(IC_df),20))
    ax2.legend()

def plotPortfolioList(portfolioList,facname='',ascloc=False,t='',is_sub_axis=1,lsname='多空组合'):
    '''
    功能：对t分组的每组和多空组合的收益画图
    输入：
        portfolioList : 含有t分组和多空组合收益信息的dataframe
        facname : 表头名称对应因子
        ascloc ：升序或降序，默认为因子值小的为第一组
        is_sub_axis : 是否需要右轴显示
        lsname : 右轴显示的列  list表示多列，str表示单列

    '''  
    data = portfolioList.apply(lambda x:x+1).cumprod().pipe(applyindex,lambda x:str(x))
    def othername(i,n,asc):
        other=str(i)
        if(n!=''):
            if(asc==False and isinstance(i,int)):
                other='第'+other+'组'
                if(int(i)==1):
                    other=other+'(最大组)'
                if(int(i)==n):
                    other=other+'(最小组)'
            else:
                if(asc==True and isinstance(i,int)):
                    other='第'+other+'组'
                    if(int(i)==1):
                        other=other+'(最小组)'
                    if(int(i)==n):
                        other=other+'(最大组)'
        return other
    if is_sub_axis == 0:
        plt.figure(figsize=(16,8))
        for i in data.columns:
            plt.plot(data[i],label = othername(i,t,ascloc))
        plt.title(facname+'原始净值图')
        plt.xticks(range(0,len(data),20))
        plt.legend()
    else:
        if(isinstance(lsname,str)):
            lsname=[lsname]
        fig,ax1 = plt.subplots(figsize=(16,8))
        ax2 = ax1.twinx()
        for i in data.drop(columns=lsname).columns:
            ax1.plot(data[i],label = othername(i,t,ascloc))
        for i in data[lsname].columns:
            if(i==data[lsname].columns[0]):
                ax2.plot(data[i],label = str(i)+'(右轴)',linestyle='--',color='r')
            else:
                ax2.plot(data[i],label = str(i)+'(右轴)',linestyle='--')
        ax1.set_xticks(range(0,len(data),20))
        ax2.set_xticks(range(0,len(data),20))
        plt.title(facname+'原始净值图')
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")



class dataProcess():
    '''
      数据处理部分，嵌套在FM.FactorTest类中执行
      只改变['v']取值——同步改变FactorTest类中的变量
    '''
    def __init__(self,DataBase={'v':pd.DataFrame(columns=['time','code'])}):
        self.dataBase=DataBase
        self.datalist=[]
        self.latestname='all'
        pd.options.mode.use_inf_as_na = True  #剔除inf

    #返回最新值self.latest
    def __call__(self,name='last'):
        if(name=='last'):
            name=self.latestname
        if(name!='all'):
            return self.dataBase['v'][['time','code',name]]
        else:
            return self.dataBase['v']

    #因子处理_等待废弃
    def updateData(self,DataBase):
        self.dataBase=DataBase
    def getData(self,Data,name='factor'):
        '''
           如果输入矩阵，请填写name
           否则请输入sql型
        '''
        if(Data.index.name=='time'):
            Data=toLongForm(Data)
            Data.columns=['time','code',name]
        if('month' in Data):
            Data.rename(columns={'month':'time'},inplace=True)
        if('date' in Data):
            Data.rename(columns={'date':'time'},inplace=True)
        factorList=Data.columns        
        if(len(factorList)<=2):
            print('error')
            return Data
        else:
            dataList=getfactorname(Data,['code','time'])
        for dataname in dataList:
            if(dataname in self.datalist):#如果重复则先删除信息再重新载入
                rest=pd.Series(self.datalist)
                self.datalist=rest[rest!=dataname].tolist()
                del self.dataBase['v'][dataname]
            self.dataBase['v']=self.dataBase['v'].merge(Data[['time','code',dataname]],on=['time','code'],how='outer')
            self.datalist=self.datalist+[dataname]    
            
    #横截面回归取残差，速度慢尽量月频
    def AregBResid(self,yname,xname,rname='resid'):
        '''
          yname 因变量 xname自变量  str /list均可
          返回回归残差，列名为rname
          自动放回dataBase['v']中
        '''
        if(type(yname)==str):
            yname=[yname]
        if(type(xname)==str):
            xname=[xname]
        datagroup=self.dataBase['v'].groupby('time')
        Ans=[]
        for date in tqdm(self.dataBase['v'].time.unique()):
            data_raw=datagroup.get_group(date)
            data_loc=data_raw.dropna(subset=yname+xname)
            data_loc[rname]=calcResid(data_loc[yname], data_loc[xname])
            data_raw[rname]=data_loc[rname]
            Ans.append(data_raw)
        self.tmp=Ans
        self.dataBase['v']=pd.concat(Ans)
        self.latestname=rname

    #因子合成——待完善
    def addFactors(self,addList='',weight_list='',factorname='factorSum'):
        '''
        因子中性化等权合成 
        Parameters
        ----------
        addList : list
            需要相加的因子列表
        weight_list: Series   index addList  columns:权重
            权重，默认等权
        factorname : str
            因子和的名称  默认名称为'factorSum'
        Returns
        -------
        无返回值
        因子和通过getData函数加入self.dataBase

        '''
        if(addList==''):
            addList=getfactorname(self.dataBase['v'])
        if(weight_list==''):
            weight_list=pd.Series(index=addList)
            weight_list=1
        weight=weight/weight.sum()
        facSum=0
        for fac_name in addList:
            tmp=self.dataBase['v'].pivot('time','code',fac_name)
            facSum=facSum+(tmp.sub(tmp.mean(axis=1), axis='index').div(tmp.std(axis=1), axis='index'))
        facSum=factorStack(facSum, factorname)
        self.getData(facSum)
        
    #仍待完善    
    def factorFillNA(self, DF, factor_list='', freq='month', method='median', window=12):
        '''

        Parameters
        ----------
        DF : 
            三列标准型.
        factor_list : str or list, optional
            需要填充缺失值的因子. The default is ''.
        freq : str, optional
            可传入'month'或者'day'. The default is 'month'.
        method : str, optional
            可传入'mean'或者'median'. The default is 'median'.

        Returns
        -------
        三列标准型因子DF，空缺值由行业中性数值填充.

        '''
        if(factor_list==''):
            factor_list=getfactorname(DF)
        if(type(factor_list)==str):
            factor_list=[factor_list]
        x=self.dataBase['v'][['time', 'code']+factor_list].copy()
        x=pd.merge(x, getSWIndustryData(freq=freq), on=['time', 'code'], how='left')
        
        x_tmp = x.groupby(['SWind', 'time']).median().reset_index()
        x_tmp = pd.merge(x, x_tmp, on=['time','SWind'], how='left') #将行业均值/中位数与原因子数据合并
        for i in factor_list:
            x_tmp[i+'_x'].fillna(x_tmp[i+'_y'], inplace=True) #'_x'为原因子列，'_y'为行业均值/中位数列
            x_tmp[i+'_ind'] = x_tmp[i+'_x']
            
        return pd.merge(DF, x_tmp[['time','code']+[i+'_ind' for i in factor_list]], on=['time','code'], how='left')

    
    def AcutB(self,yname,xname):
        pass
    def covAB(self,xnames):
        pass
    