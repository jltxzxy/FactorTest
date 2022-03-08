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
    
    FB.calcAllFactorAns()
    FB.FactorRenewTime()
    print('因子部分已更新完成')

    
    for file in os.listdir(compresspath):
        os.remove(compresspath+file)
    FB.ZipLocalFiles(Datapath,compresspath,'Data')
    FB.ZipLocalFiles(Factorpath,compresspath,'Factor',t=1)

    