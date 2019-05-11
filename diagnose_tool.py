'''
将上一个diagnose中的代码功能整合
'''
# -*-coding:utf-8-*-
from __future__ import division
import pandas as pd
import numpy as np
import timeit
from scipy import stats

from datetime import datetime

def fill_fre_top_5(x):
    if(len(x)) <= 5:
        new_array = np.full(5, np.nan)
        new_array[0:len(x)] = x
        return new_array

def eda_analysis(missSet = [np.nan,999999999,-99999999], df = None):
    ##1.Count
    start = timeit.default_timer()
    count_un = df.apply(lambda x: len(x.unique()))
    count_un = count_un.to_frame('count')
    print('不同值的个数：')
    print(count_un)
    print('Count Running Time:{}'.format(timeit.default_timer() - start))

    ##2.Count Zero
    start = timeit.default_timer()
    count_zero = df.apply(lambda x: np.sum(x == 0))
    count_zero = count_zero.to_frame('count_zero')
    print('值为0的个数：')
    print(count_zero)
    print('Count Zero Time:{}'.format(timeit.default_timer() - start))

    ##3.Mean
    start = timeit.default_timer()
    df_mean = df.apply(lambda x: np.mean(x[~np.isin(x, missSet)]))
    df_mean = df_mean.to_frame('mean')
    print('平均数是：')
    print(df_mean)
    print('Mean Time:{}'.format(timeit.default_timer() - start))

    ##4.Median
    start = timeit.default_timer()
    df_median = df.apply(lambda x: np.median(x[~np.isin(x, missSet)]))
    df_median = df_median.to_frame('median')
    print('中位数是：')
    print(df_median)
    print('Median Time:{}'.format(timeit.default_timer() - start))

    ##5.Mode
    start = timeit.default_timer()
    df_mode = df.apply(lambda x: stats.mode(x[~np.isin(x, missSet)])[0][0])
    df_mode = df_mode.to_frame('mode')
    print('众数：')
    print(df_mode)
    print('Mode Time:{}'.format(timeit.default_timer() - start))

    ##6.ModePercentage
    start = timeit.default_timer()
    df_mode_count = df.apply(lambda x: stats.mode(x[~np.isin(x, missSet)])[1][0])
    df_mode_count = df_mode_count.to_frame('mode_count')
    print('众数出现的频次：')
    print(df_mode_count)
    print('ModePercentage Time:{}'.format(timeit.default_timer() - start))

    ##6.1 Mode Percentage
    start = timeit.default_timer()
    df_mode_perct = df_mode_count / df.shape[0]
    df_mode_perct.columns = ['mode_perct']
    print('众数出现的频率：')
    print(df_mode_perct)
    print('Mode Percentage Time:{}'.format(timeit.default_timer() - start))

    ##7.Min
    start = timeit.default_timer()
    df_min = df.apply(lambda x:np.min(x[~np.isin(x, missSet)]))
    df_min = df_min.to_frame('min')
    print('最小值：')
    print(df_min)
    print('Min Time:{}'.format(timeit.default_timer() - start))

    ##8.Max
    start = timeit.default_timer()
    df_max = df.apply(lambda x: np.max(x[~np.isin(x, missSet)]))
    df_max = df_max.to_frame('max')
    print('最大值：')
    print(df_max)
    print('Max Time:{}'.format(timeit.default_timer() - start))

    ##9.Quantile
    start = timeit.default_timer()
    json_quantile = {}

    for i, name in enumerate(df.iloc[:, 0:3].columns):
        json_quantile[name] = np.percentile(df[name][~np.isin(df[name], missSet)], (1, 5, 25, 50, 75, 95, 99))

    df_quantile = pd.DataFrame(json_quantile)[df.iloc[:, 0:3].columns].T
    df_quantile.columns = ['quan01', 'quan05', 'quan25', 'quan50', 'quan75', 'quan95', 'quan99']
    print('分位点为：')
    print(df_quantile)
    print('Quantile Time:{}'.format(timeit.default_timer() - start))

    ##10.Frequence
    start = timeit.default_timer()
    json_fre_name = {}
    json_fre_count = {}

    for i, name in enumerate(df.columns):
        ##1.Index Name
        index_name = df[name][~np.isin(df[name], missSet)].value_counts().iloc[0:5, ].index.values
        ##1.1 If the length of array is less than 5
        index_name = fill_fre_top_5(index_name)

        json_fre_name[name] = index_name

        ##2.Value Count
        values_count = df[name][~np.isin(df[name], missSet)].value_counts().iloc[0:5, ].values
        ##2.If the length of array is less than 5
        values_count = fill_fre_top_5(values_count)

        json_fre_count[name] = values_count

    df_fre_name = pd.DataFrame(json_fre_name)[df.columns].T
    df_fre_count = pd.DataFrame(json_fre_count)[df.columns].T

    df_fre = pd.concat([df_fre_name, df_fre_count], axis=1)
    df_fre.columns = ['value1', 'value2', 'value3', 'value4', 'value5',
                      'freq1', 'freq2', 'freq3', 'freq4', 'freq5']
    print('数据出现的频数：')
    print(df_fre)
    print('Frequence Time:{}'.format(timeit.default_timer() - start))

    ##11.Miss Value Count
    start = timeit.default_timer()
    df_miss = df.apply(lambda x: np.sum(np.isin(x, missSet)))
    df_miss = df_miss.to_frame('freq_miss')
    print('缺失值：')
    print('df_miss')
    print('Miss Value Count Time:{}'.format(timeit.default_timer() - start))

    ##12.Combine All Informations
    start = timeit.default_timer()
    df_eda_summary = pd.concat(
        [count_un, count_zero, df_mean, df_median, df_mode,
         df_mode_count, df_mode_perct, df_min, df_max, df_fre,
         df_miss, ], axis=1
    )
    print('Combine All Informations:{}'.format(timeit.default_timer() - start))

    return df_eda_summary