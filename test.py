import pandas as pd
from diagnose_tool import *
from diagnose_tool import eda_analysis

#导入数据
df = pd.read_csv(r'F:\datasets\慕课数据诊断课程数据集\train.csv\test.csv')
#删掉不需要的ID和标签列，看数据集的情况是否需要使用
# label = df['TARGET']
# df = df.drop(['ID', 'TARGET'], axis=1)

start = timeit.default_timer()
df_eda_summary = eda_analysis(missSet=[np.nan, 9999999999, -999999], df=df.iloc[:, 0:3])
print('EDA Running Time:{0:.2f} seconds'.format(timeit.default_timer() - start))
