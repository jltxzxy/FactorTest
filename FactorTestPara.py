
#根目录
rootpath='D:\\DataBase\\'

#数据存储目录——因子测试相关
#杂项
filepathtestdata=rootpath+'factortest\\'


#数据更新文件位置
DataInfopath = rootpath+'DataInfo.xlsx'
FactorInfopath = rootpath+'FactorInfo.xlsx'

Datapath=rootpath+'DataBase/'
Factorpath=rootpath+'FactorDB/'
compresspath=rootpath+'压缩文件/'

Temppath=rootpath+'temp/'

g_starttime=19000101
g_endtime=21000101


#存储因子名单

#因子目录文件

import os
import FactorTest.FactorTestBox as FB
import FactorTest.FactorTestMain as FM
import pandas as pd
import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import datetime
import statsmodels.api as sm
import copy

# 用来正常显示中文标签
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif']=['FangSong']
# 用来正常显示负号
plt.rcParams['axes.unicode_minus']=False