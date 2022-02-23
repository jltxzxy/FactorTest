import os
import FactorTest.FactorTestBox as FB
from FactorTest.FactorTestPara import *
import pandas as pd
import numpy as np
from tqdm import tqdm

#反转因子
def calcRet20(infoDF):
    close=FB.read_feather(Datapath+'BasicFactor_Close.txt').set_index('time')
    Adj=FB.read_feather(Datapath+'BasicFactor_AdjFactor.txt').set_index('time')
    close=close*Adj
    ret20=(close/close.shift(20)-1)
    ret20=ret20.reset_index()
    ret20['time']=ret20['time'].apply(lambda x:int(str(x)[:6]))
    ret20=ret20.drop_duplicates(subset=['time'],keep='last').set_index('time')
    ret20=ret20.stack().reset_index()
    ret20.columns=['time','code',infoDF['因子名称'].iloc[0]]
    FB.save_feather(Factorpath+infoDF['地址'].iloc[0],ret20)
