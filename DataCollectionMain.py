import os
from FactorTest.FactorTestPara import *
os.chdir(rootpath)
import FactorTest.UpdateRawData
import FactorTest.UpdateFactorData
import FactorTest.FactorTestBox as FB
import FactorTest.FactorTestMain as FM
from tqdm import tqdm
import pandas as pd
if __name__=='__main__':
    FB.DataRenewTime()
    datainfo=pd.read_excel(DataInfopath)
    for func in datainfo['函数'].unique():
        getattr(FactorTest.UpdateRawData,func)(datainfo[datainfo['函数']==func])
        print(func,'更新完成')
    FB.DataRenewTime()
    print('数据库部分已更新完成')
    #因子库更新
    FB.FactorRenewTime()
    datainfo=pd.read_excel(FactorInfopath)
    for func in tqdm(datainfo['函数'].unique()):
       getattr(FactorTest.UpdateFactorData,func)(datainfo[datainfo['函数']==func])
       print(func,'更新完成')


    #待模块化
    datainfo=pd.read_excel(FactorInfopath)
    for i in tqdm(datainfo.index):
       Test=FM.FactorTest()
       Test.getFactor(FB.read_feather(Factorpath+datainfo.loc[i,'地址']))
       Test.autotest(datainfo.loc[i,'因子名称'],asc=False)
       datainfo.loc[i,'IC']=Test.ICAns[datainfo.loc[i,'因子名称']]['IC:']
       datainfo.loc[i,'ICIR']=Test.ICAns[datainfo.loc[i,'因子名称']]['ICIR:']
       FacDF=Test.portfolioList[datainfo.loc[i,'因子名称']]
       if(datainfo.loc[i,'IC']<0):
           FacDF['多空组合']=FacDF['多空组合']*-1
       datainfo.loc[i,'多空年化收益']=FB.evaluatePortfolioRet(FacDF['多空组合'])['年化收益率:']
       datainfo.loc[i,'多空信息比率']=FB.evaluatePortfolioRet(FacDF['多空组合'])['信息比率:']
       FacDF['IC']=Test.ICList[datainfo.loc[i,'因子名称']]
       FacDF.columns=pd.Series(FacDF.columns).apply(lambda x:str(x))
       FacDF.reset_index().to_feather(Factorpath+'plotdata/'+datainfo.loc[i,'因子名称']+'.fth')
    datainfo.to_excel(FactorInfopath, index=False)


    FB.FactorRenewTime()
    print('因子部分已更新完成')


    
    # compresspath='D:/DataBase/压缩文件'
    # for file in os.listdir(compresspath):
    #     os.remove(compresspath+'/'+file)
    # FB.ZipLocalFiles(Datapath,compresspath,'Data')
    # FB.ZipLocalFiles(Factorpath,compresspath,'Factor')

    