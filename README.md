
# 因子测试框架说明文档（8.11更新）

## 1.因子框架组件

### 1.1数据文件夹 DataBase  

每个feather文件存一条数据

### 1.2因子文件夹 FactorDB

每个feather文件存一个因子数据

### 1.3源代码文件夹 FactorTest**

UpdateRawData 放数据更新函数,每一个函数更新一簇数据 

UpdateFactorData  放因子数据，每一个函数更新一簇因子 

FactorTestPara 存放参数

FactorTestMain 存放因子测试的框架

FactorTestBox 存放各类用到的工具函数

DataCollectionMain 是更新的主函数

### 1.4 记录数据字典 DataInfo.xlsx

![Generated](Readme%20pictures/image-1.png)

根据函数列匹配UpdateRawData中对应的更新函数

更新时自动更新最新时间

### 1.5 记录因子字典 FactorInfo.xlsx

![Generated](Readme%20pictures/image-2.png)

每一列存储一个因子，根据函数列匹配UpdateFactorData中对应函数

### 1.6 因子测试结果.ipynb

![Generated](Readme%20pictures/image-3.png)


## 2.因子框架如何安装

目前因子框架部署于www.dwjg.xyz中，可以将1中所有组件复制到本地文件中

其中www.dwjg.xyz/压缩文件中存储有DataBase中的压缩文件（Data1-5.rar)

需要做的是：

### 1.pip第三方库 tqdm 

将pandas库更新至最新 （pip install pandas --upgrade --user)

### 2.为方便期间可以将FactorTest文件复制于python路径site-packages中（与pandas这些放到一起

即可在任意地点调用

![Generated](Readme%20pictures/image-4.png)

然后使用FB.xx FM.xx调用函数

路径可以使用 import sys      sys.path 查看

### 3.修改FactorTestPara.py 中rootpath的位置，默认为与py文件处于同一盘中，如有位置上的修改记得在para文件中修正


## 3.数据库、因子库更新办法

在小电脑上直接打开DataCollectionMain.py  点击运行，等待”数据库部分已更新完成“,"因子部分已更新完成"均出现。


## 4.因子测试FactorTestMain说明文档

![Generated](Readme%20pictures/image-5.png)

可以测试的月度因子：一共三列

time **int （数字！！）  200101**

code 代码 000001.SZ

`      `因子名  因子值  月末因子值

![Generated](Readme%20pictures/image-6.png)

FactorTestMain 中的FactosrTest() 是测试的核心类，在构造函数中已预存了个股收益率等初始信息

通用变量：

**self.FactorDataBase** 因子池 

**self.factorlist** 存储因子名称

**self.filterStockDF** 可以存放指数成分股、行业指数等,会在下面的测试中限定股票池，也可以在底下的测试前修改

**self.retData** 收益数据三列矩阵

**self.ICList** IC数据序列

**self.RankICList** RankIC数据序列   

**self.portfolioList** 投资组合序列

**self.ICAns** IC测试结果

**self.portfolioAns** 投资组合测试结果

**self.portfolioGroup** 投资组合数据

**self.annualTurnover** 储存年化换手率数据

**self.year\_performance** 年化表现

**self.WR** 储存胜率

**self.PL** 储存赔率

方法：

1. **updateFactorList** 更新因子列表self.factorlist
2. **getFactor**可以将因子序列存储加入因子池中
3. **calcIC** 可以计算IC值等信息 calcIC(self,factorlist='',startMonth='',endMonth=''):

    Factorlist 为空则计算全部因子值的相关信息，'ret20'  ['ret20','ROE\_q']均可

    StartMonth 、endMonth 如需填写 请填写数字月份 例如 201001

    得到**self.portfolioList IC结果序列 self.portfolioAns 测试结果（portfolioDF是DF版本）**

4. **calcLongShort** 同上新增两项  t=5 几分组， asc=True 从小到大

    得到**self.ICList IC结果序列 self.ICAns 测试结果(ICDF是DataFrame版本）**

    注意：如果只测试一个因子，会自动画图

5. **autoTest**  涵盖以上两个，自动简易测试
6. **TopK** 每期筛选前k(默认k=30名），计算年化收益率、信息比率、胜率、最大回撤
7. **calcFutureRet** 计算每个月向后未来1到36个月的收益均值，存储在self.FutureRet里
8. **displayWinRate** 显示胜率和赔率（盈亏比）
9. **displayYearPerformance** 展示年度收益，存储在**self.year\_performance**中
10. **calcCorrMatrix** 计算因子值间相关性与因子IC间相关性（注：需要因子数量大于1）
11. **calcCorrBarra** 计算因子值与Barra因子间相关性（目前依赖先前版本Barra)
12. **calcFamaMacBeth** 对因子值和收益进行FamaMacBeth回归，表示因子对股票收益的解释效果
13. **calcPureFactor** 计算纯净因子收益(剔除Barra与行业，得到 原因子+‘\_pure’因子作为纯净因子）
14. **doubleSorting** 对两个因子进行双重排序，得到年化收和信息比率矩阵
15. **calcGroupIC** 计算多个因子的IC、ICIR、Rank IC、Rank ICIR，必须传入因子列表
16. **Indtest为**FactorTest的Inherited Class
    **convertStocktoInd** 将个股数据转化为行业数据


## 5.怎样新增数据集

1. 在datainfo.xlsx中新增行

![Generated](Readme%20pictures/image-7.png)

在函数中填写函数名，名称、存储地址（不需要标注路径）  其他可不填（建议数据库键写英文简称）

2.在UpdateRawData.py中写 与1中函数名一致函数，以infoDF为传入参数，infoDF即datainfo.xlsx

中函数列值与函数名一致的函数，因此必须保证新增的函数名称与之前的不一致

3.可调用FB.getUpdateStartTime(infoDF['最新时间']，backdays=0)，获取目前函数更新到的最新时间点，便于从断点开始更新节约时间，backdays=t代表从最新日期向前推几个自然日

4.目前以写了 FB.saveDailyData(sqlData,infoDF)用于存储数据库中日数据

           FB.saveFinData(sqlData,infoDF)存储财务数据

           FB.saveIndexComponentData(ans, infoDF）存储wind中的指数成分股数据

           FB.saveSqlData 未完成 打算直接存储类似分析师数据这样难以节约空间的类型（直接存储）

存储feather时一定要注意：尽量存储成矩阵形式 

index为 time columns 为code  存储时注意将time独立作为一列（只要.reset\_index()一下就好）

![Generated](Readme%20pictures/image-8.png)

加入后 运行函数 检查数据创立情况

运行DataCollectionMain.py 检查数据更新情况

检查钩稽关系，即不能出现A基于B更新，但A在datainfo.xlsx中排在B前面更新的情况


## 6.怎样新增新的因子

1.在FactorInfo.xlsx中加入新因子信息，填 因子名称、函数名称、地址

![Generated](Readme%20pictures/image-9.png)

2.在UpdateFactorData中写因子更新函数

要求
    1.一定只能从本地获取数据！！ 基础数据使用FB.read\_feather(DataPath+xx )提取,注意提取后的形式，

    2.如果需要用到中间变量，请存放至temp文件夹中，方便重复利用，请注意中间变量的命名细节（形如xxx\_xxx.txt 或xxx.xxx.csv），建议同一类因子放到一个函数里面更新

    3.更新完成后存储至FactorPath+地址的feather中，统一以time**（int）**、code、因子值三列存储

![Generated](Readme%20pictures/image-10.png)

3.在因子测试.ipynb 中加入关于这一因子的测试，如

![Generated](Readme%20pictures/image-11.png)

先加Markdown  因子名称

再加简易测试

## 说明文档：FactorTestBox

        1. read/save  read\_feather/pickle/hdf5  独写各类文件
          read\_feather(地址）  
          save\_feather(地址，文件)
          注：feather 无法存储index信息，注意reset\_index一下

        2. getSql 从数据库获取信息

        4. 时间格式转化

        5. toTime 字符串转datetime

        6. fromTime datetime格式转为int

        7. regress 回归函数 （y,x,con=True)   y是因变量 x是自变量 请确保不存在空值（填充） 且y与x行数一致  con=True 加截距项

        8. getTradeDateList 获取交易日 日历 getStockList 获取所有股票序列 getRetData获取收益率数据 getFisicalList 获得财务报表发布序列

        9. getIndexComponent 获得指数成分股信息 getIndexComponent(300/500/800/1000/'wind')分别相应的指数  

        10. getIndexData获取指数日行情数据

        11. setStockPool 设定股票池 setStockPool(DF,DFfilter) 用DFfilter筛选DF矩阵
            factorDF=FB.setStockPool(factorDF, FB.getIndexComponent(poolname))

        12. 申万行业分类
          a. getSWIndustryData 获取申万行业数据
          b. addSWIndustry 在DF右侧新增一列申万行业分类

        13. 申万板块数据
          a. getSWSector 获取申万板块数据
          b. addSWSector 在DF右侧新增一列申万板块数据
        14. getIndRetData 获取行业收益数据DF三列式

        15. industryAggregate 合并成为大类行业

        16. getBarraData 获取Barra数据（time ,code,十个因子）

        17. factorStack 读取矩阵型，并转为（time,code,factorname)

        18. Kickout 剔除ST、上市60天以内、停牌股

        19. zscore&zscorefac  标准化 dePCT deSTD deMAD 对应三种去极值方法 fillmean fillmedian 对应填充空值
            方法：factor=factor.groupby('time').apply(deMAD)
            也可以直接用factorInit(factor)定义

        20. dayToMonth(DF,factor\_list='',method='last') 日频到月频转换   method:{'last','mean','sum'} 

        21. isinGroupT 判断各股票的指标在多空分中的哪一组

        22. isinTopK 判断各股票的指标知否在前k, 1表示在，0表示不在

        23. calcGroupRet 计算组合收益

        24. calcWRPL 计算胜率赔率

        25. calcResid 计算回归残差

        26. getCMV 获取市值数据

        27. getZXIndustryData 获取中信行业分类数据

        28. 中性化用 ，月频数据  
          a. addXSWindDum 加入SWind列 
          b. addXZXind 加入中信行业分类列
          c. addXSize 加入对数流通市值列
          d. addXBarra 加入barra因子值列

        29. 计算行业市值中性化
          a. Regbysize 计算现有因子和行业市值因子的回归残差
          b. calcNeuIndsize计算纯因子，排除行业市值对因子的影响

        30. 计算市值中性化
          a. RegbySize 计算现有因子和行业市值因子的回归残差，用于中性化
          b. calcNeuSize 计算市值中性，排除市值对因子的影响

        31. 计算Barra因子中性化
          a. RegbyBarra 计算现有因子和barra因子的回归残差，用于中性化
          b. calcNeuBarra 计算Barra因子中性，排除barra因子对待处理因子的影响
          c. 用法示例： factorDF=FB.calcNeuIndSize(factorDF, 'ROE\_q')

        32. calcIC 用pearsonr和spearmanr方法计算相关性（IC、rankIC）

        33. calcIClist 计算组合IC值

        34. ZipLocalFiles(file\_path,save\_path,zipname='',t=5)  将file\_path的文件压缩到save\_path 中，分为t组，组名为zipname(str) 默认为zipname+日期

        35. copyFile 复制本地文件，需输入文件地址和粘贴地址

        36. copyFiles 复制文件夹

        37. 计算年成长率
          a. \_calcGrowthRate 利用最小二乘法回归计算年成长率
          b. calcGrowthRate 利用a中函数计算过去windows年复合变化率（同比）

        38. calcAllFactorAns 计算所有储存在factorInfo文件中的因子的收益、IC等信息

        39. calcPortfolioRet 计算组合收益率

        40. calcAnnualTurnover 获取年度换手率信息

        41. DataRenewTime 更新数据库 FactorRenewTime 更新因子库

        42. getUpdateStartTime 获得开始更新时间

        43. readLocalData 读取本地数据并转为三列式  readLocalDataSet批量读取一系列数据并转为[time,code,xx,xx ]格式 readLocalFeather   读取本地feather文件
            注意：这一步会比较慢，如果不是必要建议还是多用矩阵运算

        45. 存储模块
          a. saveSqlData 存储sql型数据
          b. saveDailyData存储日频数据
          c. saveFinData用于存储财务数据
          d. saveIndData用于存储行业成分股数据
          e. saveIndexComponentData存储指数成分股

        46. toLongForm 矩阵形式转换为Sql形式数据（三列）
            toShortForm Sql形式数据转换为矩阵形式数据

        47. fillFisicalMonth 将财务数据（季）因子化【压缩成矩阵形式、拓展到月频、向上填补后再展开】

        48. partition 将列表切片再合成为每部分固定长度的嵌套列表

        49. transData 将财务数据转化为时间、股票代码、因子名称三列dataframe或矩阵
            将更名为transFisicalData

        50. calc\_plot 对dataframe进行plot画图

        51. 财务数据处理
          a. calcFisicalLYR 将3、6、9月末财务数据时间统一为上年末数据
          b. calcFisicalq 将6、9、12月末的当年累积时段财务数据转化为当季度内的数据
          c. calcFisicalttm 将季度时段财务数据转化为过去四个季度的和（ttm数据）

        52. applyindex 直接对DataFrame或Series的index执行func

        53. calc\_exp\_list 根据因子半衰期生成权重序列

        54. 计算加权数据
          a. calcWeightedStd 计算加权标准差
          b. calcWeightedMean 计算加权平均

        55. rollingRegress 待完善

        56. monthToDay 将月频数据dataframe转化为日频数据dataframe，空缺数据向下补齐

        57. plotICList 对多空组合的IC和rank IC进行绘图

        58. plotPortfolioList 对t分组的每组和多空组合的收益画图



