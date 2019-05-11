from __future__ import division
import pandas as pd
import numpy as np
from scipy import stats

#1.read datasets
df = pd.read_csv(r'F:\datasets\慕课数据诊断课程\train.csv\train.csv')
label = df['TARGET']
df = df.drop(['ID', 'TARGET'], axis=1)

#1.Missing Value缺失值
missSet = [np.nan, 9999999999, -999999]

#2.Count distinct不同值的个数
#第1列不同值得个数
len(df.iloc[:, 0].unique())

#多列（前3列）出现不同值得个数,如果只有一列的数据这一句代码就没必要存在，包括后边其他指标的代码。
count_un = df.iloc[:, 0:3].apply(lambda x: len(x.unique()))

#3.zeros values值为0的个数
np.sum(df.iloc[:, 0] == 0)
count_zero = df.iloc[:, 0:3].apply(lambda x: np.sum(x == 0))


'''
平均数和中位数
'''
#4.Mean Values平均数
np.mean(df.iloc[:, 0])#没有去除缺失值之前的均值很低

df.iloc[:, 0][~np.isin(df.iloc[:, 0], missSet)]#去除缺失值，~：取反
np.mean(df.iloc[:, 0][~np.isin(df.iloc[:, 0], missSet)])#去除缺失值后进行均值计算

df_mean = df.iloc[:, 0:3].apply(lambda  x:np.mean(x[~np.isin(x,missSet)]))

#5.Median Value中位数
np.median(df.iloc[:, 0])#没有去除缺失值之前
df.iloc[:, 0][~np.isin(df.iloc[:, 0],missSet)]#去除缺失值
np.median(df.iloc[:, 0][~np.isin(df.iloc[:, 0], missSet)])#去除缺失值后进行计算

df_mean = df.iloc[:,0:3].apply(lambda x:np.median(x[~np.isin(x, missSet)]))

'''
众数，scipy包中的函数
'''
#6.Mode Value
#mode返回两个值，第一个值是众数的本身是什么，第二个是众数有多少个频数
df_mode = df.iloc[:, 0:3].apply(lambda x: stats.mode(x[~np.isin(x, missSet)])[0][0])

#7.Mode Percentage众数的比例
df_mode_count = df.iloc[:, 0:3].apply(lambda x: stats.mode(x[~np.isin(x, missSet)])[1][0])
#频次/数量
df_mode_perct = df_mode_count/df.shape[0]

'''
最大最小值
'''
#8.Min Values
np.min(df.iloc[:,0])

df.iloc[:, 0][~np.isin(df.iloc[:, 0], missSet)]  #去除缺失值
np.min(df.iloc[:, 0][~np.isin(df.iloc[:, 0], missSet)])  #去除缺失值后进行最小值计算

df_min = df.iloc[:, 0:3].apply(lambda x: np.min(x[~np.isin(x, missSet)]))

#9.Max values
np.max(df.iloc[:, 0])

df.iloc[:, 0][~np.isin(df.iloc[:, 0], missSet)]  #去除缺失值
np.max(df.iloc[:, 0][~np.isin(df.iloc[:, 0], missSet)])#去除缺失值后进行最大值计算

df_max = df.iloc[:, 0:3].apply(lambda x:np.max(x[~np.isin(x, missSet)]))

'''
分位点
'''
#10.quantile values
#percentile，主要使用两个入参：1.使用的数据本身。2.要切分或要显示的分位点
np.percentile(df.iloc[:, 0], (1, 5, 25, 50, 75, 95, 99))

df.iloc[:, 0][~np.isin(df.iloc[:, 0], missSet)]#去除缺失值
np.percentile(df.iloc[:, 0][~np.isin(df.iloc[:, 0], missSet)], (1, 5, 25, 50, 75, 95, 99))

#定义一个字典
json_quantile = {}

for i, name in enumerate(df.iloc[:, 0:3].columns):
    print('the {} columns: {} '.format(i, name))
    json_quantile[name] = np.percentile(df[name][~np.isin(df[name], missSet)], (1, 5, 25, 50, 75, 95, 99))
#DataFrame:Python中Pandas库中的一种数据结构，它类似excel，是一种二维表。
#做成DataFrame后需要转置，否则位置对不上
df_quantile = pd.DataFrame(json_quantile)[df.iloc[:, 0:3].columns].T

'''
频数
'''
#11.Frequent Values
df.iloc[:, 0].value_counts().iloc[0:5, ]

df.iloc[:, 0][~np.isin(df.iloc[:, 0], missSet)]  #去除缺失值
df.iloc[:, 0][~np.isin(df.iloc[:, 0], missSet)].value_counts()[0:5]  #去除缺失值后进行频数的统计

json_fre_name = {}
json_fre_count = {}

#如果特征不满5个
def fill_fre_top_5(x):
    if(len(x)) <= 5:
        new_array = np.full(5, np.nan)
        new_array[0:len(x)] = x
        return new_array

df['ind_var1_0'].value_counts()
df['imp_sal_var16_ult1'].value_counts()

#遍历两个特征，columns:取名字
for i, name in enumerate(df[['ind_var1_0','imp_sal_var16_ult1']].columns):
    index_name = df[name][~np.isin(df[name], missSet)].value_counts().iloc[0:5, ].index.values
    #判断是否≤5，如果是，则用刚才定义的填nan的方式
    index_name = fill_fre_top_5(index_name)

    json_fre_name[name] = index_name

    values_count = df[name][~np.isin(df[name], missSet)].value_counts().iloc[0:5, ].values
    values_count = fill_fre_top_5(values_count)

    json_fre_count[name] = values_count

df_fre_name = pd.DataFrame(json_fre_name)[df[['ind_var1_0','imp_sal_var16_ult1']].columns].T
df_fre_count = pd.DataFrame(json_fre_count)[df[['ind_var1_0','imp_sal_var16_ult1']].columns].T
#concat:将数据根据不同的轴做简单的融合,这里利用concat做列合并
df_fre = pd.concat([df_fre_name,df_fre_count], axis=1)

'''
缺失值
'''
np.sum(np.isin(df.iloc[:, 0], missSet))#统计缺失值
df_miss = df.iloc[:, 0:3].apply(lambda x:np.sum(np.isin(x, missSet)))  #遍历每一个遍历的缺失值情况