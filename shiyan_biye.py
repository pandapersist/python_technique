#coding=utf-8
------------------------------------------------------------------------------
常见经典错误
【1】如果对dataframe进行排序或者删除之后，dataframe的索引序列
dataframe.index（可以看成是一个列表）顺序再也不是我们想象中的0,1,2,3,4,5了，谨记！！！！
另外，只要是出现keyerror，不是不含有列，就是行（即索引出现了问题）；
【2】关于编码问题，不要使用sys.setdefaultencoding('utf8')这条语句，编程的时候中文要写成
u'中国'这种格式，另外中国尽量用英文表述；
-------------------------------------------------------------------------------
前期准备工作
os.getcwd()
os.chdir("E:\\Hobbies_Learning\\0000000Beye")  更改python工作目录，便于导入文件

reload(sys)                 
sys.setdefaultencoding('gb18030')

sys.getdefaultencoding()
sys.getdefaultdecoding()

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']#指定默认字体  
mpl.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题 
sns.set_context("talk")

常用包和类函数导入
import numpy as np
from numpy import nan as NA
from numpy.random import randn #导入随机数模块
from pandas import Series
from pandas import DataFrame
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import pylab
import sys
import datetime
from datetime import datetime 
import xgboost as xgb  
import operator  
import time
from numpy import array
from datetime import date
import matplotlib.pyplot as plt
import os
import os,random,cPickle
import sklearn
from sklearn import preprocessing as preprocessing

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
"""一、初探数据，了解数据结构"""
（1）数据集结构
df_Master = pd.read_csv('my_master_train_6.csv',encoding='gb18030')
df_Master.columns.to_csv('data_columns.csv')
df_Master.head(3)
df_Master.info()  在2.7版本中模式没用
df_Master.dtypes
df_Master.count() 统计值个数，进而相减可以得到缺失值的个数
df_Master.describe()

（2）另一方面，多维度了解数据集缺失值情况
一方面按行统计 每个样本的属性缺失值个数
num_missing_row_Master = []
for i in range(len(df_Master.index)):
    num_missing_row_Master.append(sum(pd.isnull(df_Master.ix[i,:])))

df_Master['num_missing_row'] = num_missing_row_Master 将缺失值个数加到df_Master中，此时共有228+7+1个特征

训练集-测试集生成
Master_train = df_Master.ix[0:55000,:] 55000条记录作为训练样本
Master_test = df_Master.ix[55000:60000,:]  5000条记录作为测试样本

Master_train_sorted = Master_train.sort_index(by=['num_missing_row'])
Master_test_sorted = Master_test.sort_index(by=['num_missing_row'])

绘制出散点图
x1 = np.arange(0,55001)
plt.scatter(x1,Master_train_sorted['num_missing_row'],label = '训练集')
x2 = np.arange(0,5000)
plt.scatter(x2,Master_test_sorted['num_missing_row'],label = '测试集')


另一方面按列统计每个属性的样本缺失值个数
为了下面按列统计每个属性的样本缺失值个数
#df_Master[['SanDong','SiChuan','JiLin','TianJing','HuNan',
#'LiaoNing','HuBei']]=df_Master[['SanDong','SiChuan','JiLin','TianJing','HuNan',	
#'LiaoNing','HuBei']].fillna(0) #7个城市特征变量，删除字符型变量

num_missing_col_Master = []
for j in range(len(df_Master.columns)):
    num_missing_col_Master.append(sum(pd.isnull(df_Master.ix[:,j])))
#print num_missing_col_Master

df_col = DataFrame(num_missing_col_Master,columns=['num_missing_col'])
df_col.insert(0,'df_Master_columns',df_Master.columns) 
df_col_sorted = df_col.sort_index(by=['num_missing_col'],ascending=False)

df_col_sorted.to_csv('col_missing_num.csv')
col_num_missing_15 = df_col_sorted['df_Master_columns'][0:15] 
特征缺失值比率前15的特征的图像
fig = plt.figure(figsize=(15,8)); ax = fig.add_subplot(1,1,1)
#%pylab inline #表示启动在线绘图功能
y = array(df_col_sorted['num_missing_col']/ len(df_Master.index)) 
x = range(0,15,1)

ax.plot(y,label=x,linestyle='dashed',marker='o',alpha=0.9)
ax.set_ylabel('缺失率（%）')
ax.set_xlabel('特征编号')

#给图添加text
for i,j in zip(x,y):
    ax.text(i+0.03, j+0.5, '%.2f' % j, ha='center', va= 'bottom')
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
"""二、可视化探索性分析"""

（1）正负样本分布情况
df_Master[df_Master['target']==1]  4431条贷款违约
df_Master[df_Master['target']==0]  55569条正常还款
fig = plt.figure()
labels = ['贷款违约','正常还款']
X = [4431./60000,55569./60000]
plt.pie(X,labels=labels,autopct='%1.2f%%')

(2)贷款个数最多的19个城市(由于数据标签加在直方图上问题，在excel中可以轻松得多效果)
城市与贷款target的关联统计
df_Master['counts'] = [1] * len(df_Master.index)
citys = df_Master.pivot_table('counts',index=['UserInfo_2'],columns='target',aggfunc='sum')
citys.to_csv('333citys_rebell_nums.csv')


city = pd.read_excel('city_rebell_rate.xlsx')
city_loan_19 = province.ix[:,0]

fig = plt.figure();ax = fig.add_subplot(1,1,1)
ax.plot(city_rate['贷款个数'])
ticks = ax.set_xticks([i for i in range(1,20)])
labels = ax.set_xticklabels(array(city_rate['城市名']),rotation=30,fontsize='small')
ax.set_title('贷款个数最多的19个城市')
ax.set_xlabel('贷款个数（个）')

（3）统计训练集中每天借贷的成交量，正负样本分别统计
df_Master_1 = df_Master[df_Master.target==1]
temp = df_Master_1[['ListingInfo','target']].groupby('ListingInfo').agg('sum')
temp = temp.rename(columns={'target':'贷款违约'})
temp['date'] = temp.index
temp.date = temp.date.apply(lambda x:(date(int(x.split('/')[0]),int(x.split('/')[1]),int(x.split('/')[2]))-date(2013,11,1)).days)
temp = temp.sort(columns='date')

ax = temp.plot(x='date',y='贷款违约')

df_Master_0 = df_Master[df_Master.target==0]
df_Master_0.target = [1 for _ in range(len(df_Master_0))]  #生成一列1的三目式
temp_0 = df_Master_0[['ListingInfo','target']].groupby('ListingInfo').agg('sum')
temp_0 = temp_0.rename(columns={'target':'正常还款'})
temp_0['date'] = temp_0.index
temp_0.date = temp_0.date.apply(lambda x:(date(int(x.split('/')[0]),int(x.split('/')[1]),int(x.split('/')[2]))-date(2013,11,1)).days)
temp_0 = temp_0.sort(columns='date')

temp_0.plot(x='date',y='正常还款',ax=ax)

plt.xlabel('天数(20131101~20141109)')
plt.ylabel('个数')

（4）关联统计之每个城市的违约率(这一步由于中文字符乱码问题，在ipython-notebook中完成的，并得到两个excel表)
df_Master['count'] = np.ones(len(df_Master.index))
bdataframe = pd.pivot_table(df_Master[['UserInfo_8','target','count']],
                            index=['UserInfo_8'],columns=['target'],values=['count'],aggfunc=np.sum)
bdataframe_count = bdataframe['count']
bdataframe_count['贷款个数'] = bdataframe_count.sum(axis=1)
bdataframe_count['违约率'] = bdataframe_count[1] / bdataframe_count['贷款个数']
bdataframe_count_sorted = bdataframe_count.sort_index(by=["违约率"],ascending = False)


   绘制相应的图片
city_rate = pd.read_excel('city_rebell_rate.xlsx')
fig = plt.figure(figsize=(20,10)); ax = fig.add_subplot(1,1,1)
#%pylab inline #表示启动在线绘图功能
y = city_rate[u'违约率']
x = range(32)

ax.plot(y,label=x,linestyle='dashed',marker='o',alpha=0.9)
ax.set_ylabel(u'违约率（%）')
ax.set_xlabel(u'城市名')

#给图添加text
for i,j in zip(x,y):
    ax.text(i+0.03, j+0.000005, '%.2f' % j, ha='center', va= 'bottom')



这一步可以得到关于是否为某个省份的特征工程。
province = pd.read_excel(u'城市违约率情况.xlsx')
province_weiyue_6 = province[u'违约率'][:6]

（5）关联统计之UserInfo_9(中国电信、中国移动、中国联通、不详细)，与target
 要先删除空格项（可是这条语句执行速度很慢）
UserInfo_9_array = array(df_Master['UserInfo_9'])
UserInfo_9_strip = []
for each in UserInfo_9_array:
    UserInfo_9_strip.append(each.strip())    numpy科学计算速度就是要快很多！！！不能直接在dataframe上操作，速度会很慢

UserInfo_9_list = []
for i in range(len(df_Master.index)):
    if UserInfo_9_strip[i] == u'中国移动':
        UserInfo_9_list.append(u'中国移动')        
    elif UserInfo_9_strip[i] == u'中国电信':
        UserInfo_9_list.append(u'中国电信')
    elif UserInfo_9_strip[i] == u'中国联通':
        UserInfo_9_list.append(u'中国联通')
    else:
        UserInfo_9_list.append(np.nan)
df_Master['UserInfo_9'] = UserInfo_9_list

fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

target_0 = df_Master.UserInfo_9_strip[df_Master.target == 0].value_counts() 
target_1 = df_Master.UserInfo_9_strip[df_Master.target == 1].value_counts()  

df=pd.DataFrame({u'正常还款':target_0, u'贷款违约':target_1})
df.plot(kind='bar', stacked=True)
plt.title(u"按通讯网络看个人信用情况")
plt.xlabel(u"通讯网络类别") 
plt.ylabel(u"人数") 
plt.show()

#df_Master.drop(['UserInfo_9_strip'],1) 这算是构造了一个特征。

（6）关联统计之UserInfo_22（未婚2614、已婚954、再婚7、初婚3、离婚69、复婚1、丧偶1）与target
UserInfo_22_list = []
UserInfo_22_array = array(df_Master['UserInfo_22'])
for i in range(len(df_Master.index)):
    if UserInfo_22_array[i] == u'已婚':
        UserInfo_22_list.append(u'已婚')
    elif UserInfo_22_array[i] == u'未婚':
        UserInfo_22_list.append(u'未婚')
    elif UserInfo_22_array[i] == u'离婚':
        UserInfo_22_list.append(u'离婚')
    else:
        UserInfo_22_list.append(np.nan)

df_Master['UserInfo_22'] = UserInfo_22_list

fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

target_0 = df_Master.Marriage[df_Master.target == 0].value_counts()
target_1 = df_Master.Marriage[df_Master.target == 1].value_counts()

df=pd.DataFrame({u'贷款违约':target_1, u'正常还款':target_0})
df.plot(kind='bar', stacked=True)
plt.title(u"按婚姻看个人信用情况")
plt.xlabel(u"个人婚姻情况") 
plt.ylabel(u"人数")
plt.show()


（7）关联统计之UserInfo_5性别与target 1女，2男，3小微企业
df_Master.columns
citys = df_Master.pivot_table('counts',index=['UserInfo_5'],columns='target',aggfunc='sum')
citys
citys.to_csv('sex.csv')




fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

UserInfo_5_array = array(df_Master['UserInfo_5'])
UserInfo_5_list = []
for i in range(len(df_Master.index)):
    if UserInfo_5_array[i] == 1:
        UserInfo_5_list.append('女性')             #因为有一些不知道性别的信息，需要先将它们滤除掉
    elif UserInfo_5_array[i] ==2:
        UserInfo_5_list.append('男性')
    else:
        UserInfo_5_list.append(np.nan)
df_Master['UserInfo_5'] =  UserInfo_5_list   

target_0 = df_Master.sex[df_Master.target == 0].value_counts()
target_1 = df_Master.sex[df_Master.target == 1].value_counts()
df=pd.DataFrame({u'正常还款':target_0, u'贷款违约':target_1})
df.plot(kind='bar', stacked=True)
plt.title(u"按性别看个人信用情况")
plt.xlabel(u"性别") 
plt.ylabel(u"人数")
plt.show()

#df_Master.drop(['sex'],1)  其实未必要删除，这也算是特征工程的。
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
"""三、数据预处理"""
（一）清洗样本
（1）观察训练样本和测试样本的一致性，去除不一致性
df_Master = df_Master[df_Master['num_missing_row']<20]  #删除60000-59039个样本点

（2）删除缺失值比较多的列
df_Master = df_Master.drop(col_num_missing_15[:5],1)
去掉5个特征，此时还剩230个特征

（3）剔除常变量
df_Master_std = df_Master.describe().ix[2,:]  
df_Master_std = DataFrame(df_Master_std,columns=['std'])
对这208个数值型的标准差进行排序
df_Master_std_sorted = df_Master_std.sort_index(by=['std'])  
columns_std01 = df_Master_std_sorted[df_Master_std_sorted['std']<0.1]
columns_std01 = array(columns_std01.index)
df_Master = df_Master.drop(columns_std01,1)
删除了15个特征属性，还剩213

（4）删除在重要特征上缺失值比较多的样本点（离群点）
此处选择xgb计算特征的重要性权重。
df_Master2 = df_Master.copy()  为了不影响后面的特征工程，应该复制一个
丢弃字符型变量，只得到数值型变量。
cont_list = []
for i in range(len(df_Master2.columns)):
    if type(df_Master2.ix[0,i]) == np.int64:
        cont_list.append(i)
    elif type(df_Master2.ix[0,i]) == np.float64:   
        cont_list.append(i)
df_Master2 = df_Master2[cont_list]   这一步可以剔除掉34个类别型变量

df_Master2[['SanDong','SiChuan','JiLin','TianJing','HuNan',
'LiaoNing','HuBei']]=df_Master2[['SanDong','SiChuan','JiLin','TianJing','HuNan',	
'LiaoNing','HuBei']].fillna(0) #7个城市特征变量，删除字符型变量

for i in range(len(df_Master2.columns)):
    df_Master2 = df_Master2.dropna(how='all',axis=1) #丢弃全为NaN的列
    df_Master2.ix[:,i] = df_Master2.ix[:,i].fillna(df_Master2.ix[:,i].mean()) 
    #用每一列中值充填相应列的缺失值

xgb进行特征重要性权重的计算，得到各个特征的权重
df_feat_importance

统计样本点在这些重要特征上的缺失值个数，如果缺失值比较多的话，表示为离群点。
NAN_row_feat_importance = []
aarray = array(df_feat_importance['特征'])   df_Master2[aarray]将含有重要特征的数据框拿出来
for i in range(len(df_Master2[aarray].index)):
    NAN_row_feat_importance.append(sum(pd.isnull(df_Master2[aarray].ix[i,:])))

df_Master2['NAN_row_feat_importance'] = NAN_row_feat_importance 将缺失值个数加到df_Master中
df_Master_sorted = df_Master2.sort_index(by=['NAN_row_feat_importance'],ascending=False)
df_Master = df_Master_sorted[df_Master_sorted['NAN_row_feat_importance']<10]

（二）填缺失值
在使用xgb的时候已经使用了平均值进行填充。应该使用平均值进行填充，这样会更合理，
若使用中位值，对于0-1类别型的不好处理。
for i in range(len(df_Master.columns)):
    df_Master = df_Master.dropna(how='all',axis=1) #丢弃全为NaN的列
    df_Master.ix[:,i] = df_Master.ix[:,i].fillna(df_Master.ix[:,i].mean()) 

-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
"""四、特征工程"""
df.dtypes 查看数据框中有哪些数据类型
typelist = []
for i in range(len(df_Master.columns)):
    if type(df_Master.ix[0,i]) not in typelist:
        typelist.append(type(df_Master.ix[0,i]))
    else:
        continue
    
循环遍历，将数值型和类别型特征分别放在两个不同的list，找出有哪些类别型及哪些数值型变量。
cont_list = []
str_list = []
for each in df_Master.columns:
    if type(df_Master[each][0]) == np.int64:
        cont_list.append(each)
    elif type(df_Master[each][0]) == np.float64:   
        cont_list.append(each)
    else:
        str_list.append(each)
#df = df[cont_list]    剩下224个特征,丢弃字符型变量，只得到数值型数据集。

（1）日期型。（将日期转换为数值型特征及离2013年11月1日时间）
df_Master.ListingInfo = df_Master.ListingInfo.apply(lambda x:(date(int(x.split('/')[0]),int(x.split('/')[1]),int(x.split('/')[2]))-date(2013,11,1)).days)

(2) 地理位置型。包括7个字段：
用startswith、endswith来查找字符串
startswith_list = []
for each in df_Master.columns:
    if each.startswith('UserInfo_2'):
        startswith_list.append(each)

城市
citycolums = ['UserInfo_2','UserInfo_4','UserInfo_8','UserInfo_20']
citydict ={u'淮纺':'tanfan',u'九江':'jiujiang',u'三门峡':'sanmeng',u'汕头':'shantou',u'长春':'changchun',u'铁岭':'tieling',u'济南':'jinan',u'成都':'chengdu',u'淄博':'zibo',u'牡丹江':'mudang'}
for each in citycolums:
    alist = []    
    for i in df_Master.index:                             #注意千万不能写成for i in range(len(df_Master.index))
        #print each,df_Master[each]
        if df_Master[each][i] in citydict.keys():
            alist.append(citydict[df_Master[each][i]])        
        else:
            alist.append(np.nan)
    df_Master[each] = alist

dummies_UserInfo_2 = pd.get_dummies(df_Master['UserInfo_2'], prefix= 'UserInfo_2') 
dummies_UserInfo_4 = pd.get_dummies(df_Master['UserInfo_4'], prefix= 'UserInfo_4') 
dummies_UserInfo_8 = pd.get_dummies(df_Master['UserInfo_8'], prefix= 'UserInfo_8') 
dummies_UserInfo_20 = pd.get_dummies(df_Master['UserInfo_20'], prefix= 'UserInfo_20') 
df_Master = pd.concat([df_Master,dummies_UserInfo_2, dummies_UserInfo_4, dummies_UserInfo_8, dummies_UserInfo_20], axis=1)


省份
provincecolums = ['UserInfo_7','UserInfo_19']
provincedict = {u'四川':'sichuang',u'湖南':'hunan',u'湖北':'hubei',u'吉林':'jilin',u'天津':'tianjing',u'山东':'shandong'}

for each in provincecolums:
    alist = []
    for i in df_Master.index:
        if df_Master[each][i] in provincedict.keys():
            alist.append(provincedict[df_Master[each][i]])        
        else:
            alist.append(np.nan)
    df_Master[each] = alist

dummies_UserInfo_7 = pd.get_dummies(df_Master['UserInfo_7'], prefix= 'UserInfo_7') 
dummies_UserInfo_19 = pd.get_dummies(df_Master['UserInfo_19'], prefix= 'UserInfo_19') 
df_Master = pd.concat([df_Master,dummies_UserInfo_7, dummies_UserInfo_19], axis=1)


    
(2)类别型。哑变量，即将类别型变量数值化 get_dummies
使用pandas的“get_dummies”来完成这个工作，并拼接在原来的df_Master上。

三大运营商
UserInfo_9_list_tras = []
for each in df_Master.index:
    if each == u'中国移动':
        UserInfo_9_list_tras.append('ChinaMobile')
    elif each == u'中国联通':
        UserInfo_9_list_tras.append('ChinaUnicom')
    elif each == u'中国电信':
        UserInfo_9_list_tras.append('ChinaTelecom')
    else:
        UserInfo_9_list_tras.append(np.nan)
        
df_Master['UserInfo_9_list_tras'] =UserInfo_9_list_tras
df_Master = pd.concat([df_Master,pd.get_dummies(df_Master['UserInfo_9_list_tras'], prefix= 'UserInfo_9')], axis=1)
#df_Master = df_Master.drop(['UserInfo_9'],1)


婚姻问题
UserInfo_22_list_marriage = []
for each in UserInfo_22_list:
    if each == u'已婚':
        UserInfo_22_list_marriage.append('married')
    elif each == u'未婚':
        UserInfo_22_list_marriage.append('unmarried')
    elif each == u'离婚':
        UserInfo_22_list_marriage.append('divorce')
    else:
        UserInfo_22_list_marriage.append(np.nan)

df_Master['UserInfo_22_list_marriage'] = UserInfo_22_list_marriage
df_Master = pd.concat([df_Master,pd.get_dummies(df_Master['UserInfo_22_list_marriage'], prefix= 'UserInfo_22')], axis=1)
#df_Master = df_Master.drop(['UserInfo_22'],1)


性别
UserInfo_5_list_sex = []
for each in UserInfo_5_list:
    if each == u'男性':
        UserInfo_5_list_sex.append('man')
    elif each == u'女性':
        UserInfo_5_list_sex.append('woman')
    else:
        UserInfo_5_list_sex.append(np.nan)

df_Master = pd.concat([df_Master,pd.get_dummies(df_Master['UserInfo_5_list_sex'], prefix= 'UserInfo_5')], axis=1)
#df_Master = df_Master.drop(['UserInfo_22'],1)


#dummies_Marriage = pd.get_dummies(df_Master['Marriage'], prefix= 'Marriage') 婚姻3
#dummies_UserInfo_9_strip = pd.get_dummies(df_Master['UserInfo_9_strip'], prefix= 'UserInfo_9_strip')  通讯网络3
#df = pd.concat([df_Master,dummies_sex, dummies_Marriage, dummies_UserInfo_9_strip], axis=1)

（3）数值型。对缺失值，使用每列的平均值进行填充。此处还有一个数值型的离散化还没有
for i in range(len(df.columns)):
    df = df.dropna(how='all',axis=1) #丢弃全为NaN的列
    df.ix[:,i] = df.ix[:,i].fillna(df.ix[:,i].mean()) 

（4）特征交叉。（特征合并，将城市合并成一二三级城市）
婚姻问题*性别
UserInfo_22_startswith_list = []
for each in df_Master.columns:
    if each.startswith('UserInfo_22'):
        UserInfo_22_startswith_list.append(each)

for each in UserInfo_5_list:
    if 

UserInfo_5_startswith_list = []
for each in df_Master.columns:
    if each.startswith('UserInfo_5'):
        UserInfo_5_startswith_list.append(each)

k = 1
for i in UserInfo_5_startswith_list:
    for j in UserInfo_22_startswith_list:
        df_Master[k] = df_Master[i]*df_Master[j]        
        k += 1

for each in provincecolums :
'UserInfo_22'

省份+城市
provincecolums = ['UserInfo_7','UserInfo_19']
provincelist = ['_sichuang','_hunan','_hubei','_jilin','_tianjing','_shandong']
province = []
for i in provincecolums:
    for j in provincelist:
        province.append(i+j)
        
citycolums = ['UserInfo_2','UserInfo_4','UserInfo_8','UserInfo_20']
citylist =['_tanfang','_jiujiang','_sanmeng','_shantou','_changchun','_tieling','_jinan','_chengdu','_zibo','_mudang']
city = []
for i in citycolums:
    for j in citylist:
        city.append(i+j)

for i in province:
    for j in city:
        df_Master[i+j] = df_Master[i]+df_Master[j]
sys.getdefaultencoding() # 输出当前编码

(5) 对某些数值特征归一化。scaling，将一些变化幅度较大的特征化到[-1,1]之内
scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df['Age'])
df['Age_scaled'] = scaler.fit_transform(df['Age'], age_scale_param)
fare_scale_param = scaler.fit(df['Fare'])
df['Fare_scaled'] = scaler.fit_transform(df['Fare'], fare_scale_param)
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
"""五、特征选择"""
使用xgb进行选择，注意xgb的运用原理。
选取重要的特征进行下面的模型训练及评估

------------------------------------------------------------------------------

-------------------------------------------------------------------------------
"""六、模型训练及评估"""
（1）单模型
"""LR"""
from sklearn import linear_model

# 用正则取出我们要的属性值
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
df_train_np = df_train.as_matrix()

# y即Survival结果
y = df_train_np[:, 0]

# X即特征属性值
X = df_train_np[:, 1:]

# fit到RandomForestRegressor之中
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(X, y)

test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
result.to_csv("logistic_regression_predictions.csv", index=False)

模型系数关联分析
把得到的model系数和feature关联起来看看。
pd.DataFrame({"columns":list(train_df.columns)[1:], "coef":list(clf.coef_.T)})

这些系数为正的特征，和最后结果是一个正相关，反之为负相关
解释权重系数


"""SVM"""




RF

GBDT

（2）多模型（模型融合）使用Bagging策略融合LR
from sklearn.ensemble import BaggingRegressor

train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
train_np = train_df.as_matrix()

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

# fit到BaggingRegressor之中
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)
bagging_clf.fit(X, y)

test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
predictions = bagging_clf.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
result.to_csv("/Users/HanXiaoyang/Titanic_data/logistic_regression_bagging_predictions.csv", index=False)


------------------------------------------------------------------------------

-------------------------------------------------------------------------------
"""七、工具"""

"""训练xgb模型对特征进行重要性排序，特征选择"""
df_y = df_Master['target'] 
df_X = df_Master.drop(['target', 'Idx'], 1) #1表示列所对应的全部行
y = array(df_y) #将dataframe转换成array，便于后面的计算
X = array(df_X)

import os,random,cPickle
os.mkdir('featurescore')  生成一个文件夹

def ceate_feature_map(features):  
    outfile = open('xgb.fmap', 'w')  
    i = 0  
    for feat in features:  
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))  
        i = i + 1  
    outfile.close()  
  
  
if __name__ == '__main__':  
    #train = pd.read_csv("../input/train.csv")  
    #找出类别型的特征（且特征名含有“cat”字符）
    #cat_sel = [n for n in df_Master.columns if n.startswith('cat')]   
    #将类别型特征数值化 
    #for column in cat_sel:  
        #train[column] = pd.factorize(train[column].values , sort=True)[0] + 1  
  
    params = {  
        'min_child_weight': 0.2,  
        'eta': 0.04,  
        'colsample_bytree': 0.3,  
        'max_depth': 6,  
        'subsample': 0.6,  
        'alpha': 1,  
        'gamma': 0.1,  
        'silent': 1,  
        'verbose_eval': True,  
        'seed': 1024  
    }  
    rounds = 50  
    #y = df_Master[214]  
    #X = df_Master.drop([214, 0], 1)  #1表示列所对应的全部行
  
    xgtrain = xgb.DMatrix(X, label=y)  
    bst = xgb.train(params, xgtrain, num_boost_round=rounds)  
  
    features = [x for x in df_X.columns if x not in [0,214]]  
    ceate_feature_map(features)  
  
    importance = bst.get_fscore(fmap='xgb.fmap')  
    importance = sorted(importance.items(), key=operator.itemgetter(1))  
  
    df = pd.DataFrame(importance, columns=[u'特征', u'权值'])  
    df[u'权重'] = df[u'权值'] / df[u'权值'].sum() 
    df.to_csv("feat_importance.csv", index=False)  
    
    #选择特征重要性前15的特征绘制图像
    plt.figure() 
    df_feat_importance = pd.read_csv('feat_importance.csv')
    df_feat_importance = df_feat_importance.sort_index(by=[u'权重'],ascending=False)
    #df_feat_importance.drop(['index1'], 1)    
    #df_feat_importance['index1'] = [i for i in range(len(df_feat_importance.index))]
    df_feat_importance = df_feat_importance[0:15] #排完序之后索引号也被打乱了。!!
    #df_feat_importance.plot(kind='barh', x='特征', y='权重', legend='权重', figsize=(6, 10))  
    df_feat_importance.plot(kind='barh', x=u'特征', y=u'权重',figsize=(6, 10))    
    plt.title(u'基于XGBoost计算的特征重要性')  
    plt.xlabel(u'特征的权重')  
    plt.show()  



------------------------------------------------------------------------------





-------------------------------------------------------------------------------




------------------------------------------------------------------------------





-------------------------------------------------------------------------------





















