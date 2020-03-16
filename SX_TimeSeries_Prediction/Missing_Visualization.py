#!usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author:jinxin
@time: 2020/03/12
@Project : PycharmProjects
"""
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
import sx_data_preprocess as sdp
import pandas as pd
from operate_influxdb import operate_influxdb

"""
数据完整性/缺失情况可视化，两种方法都可以，建议用第二种
"""


# 数据缺失heatmap
def missing_heatmap(df):
    colors = ['#000099', '#ffff00']
    sns.heatmap(df.isnull(), cmap=sns.color_palette(colors))
    plt.show()


# missingno包画数据缺失图
def missing_msno(df):
    msno.matrix(df, labels=True, fontsize=8, figsize=(12, 8))
    plt.show()


# 时间戳连续性检测
# def timestamp_continuity_detection(df):
#     ts = pd.DataFrame()
#     ts['time'] = df['Time'].apply(lambda x: pd.Timestamp(x))
#     ts['diff'] = ts['time'].diff(1)
#     plt.plot(ts['time'], ts['diff'])
#     plt.show()
#     print(ts)


# 时间戳连续性检测
def timestamp_continuity_detection(df):
    ts = pd.DataFrame()
    ts['time'] = df['Time'].apply(lambda x: pd.Timestamp(x))
    timestamp_list = [ts['time'][0]]
    ts['diff'] = ts['time'].diff(1)
    print(ts)
    for i in range(1, len(ts)):
        if ts['diff'][i] < pd.Timedelta('00:00:05'):
            timestamp_list.append(ts['time'][i])
        else:
            n = ts['diff'][i] / pd.Timedelta('00:00:03')
            timestamp_list.extend([np.NAN] * int(n))
    ts = pd.DataFrame(data=timestamp_list, columns=['Timestamp'])
    missing_msno(ts)
    # return ts
    # print()


table = '0拉丝机'
oi = operate_influxdb('myDB')
print(oi.show_all_tables())
# codenameNum = oi.query_all_df_limit(table, 1000).CodeName.nunique()
# rows = 50000 * codenameNum
# df = oi.query_all_df_limit(tablename=table, rows=rows)
sql = 'select * from ' + '\"' + table + '\"' + \
      ' where time > ' + '\'' + '2020-02-01T00:00:00Z' + '\''  + ';'

# sql = 'select Date, Time from ' + '\"' + table + '\"' + ';'
# print(sql)
df = oi.query_df(sql)
df = sdp.codename_to_columns(df, table)
df.to_csv(r'C:\Users\X1\Desktop\data_last_50000\\' + table + '.csv')
missing_msno(df)
timestamp_continuity_detection(df)

# sql = 'select * from ' + '\"' + table + '\"' + \
#       ' where time > ' + '\'' + '2020-01-13T00:00:00Z' + '\'' + \
#       ' and time < ' + '\'' + '2020-01-15T00:00:00Z' + '\'' + ';'
# print(sql)
# df = oi.query_df(sql)
# print(df)
# print(df.shape)
# df = sdp.codename_to_columns(df, table)
# missing_msno(df)
# timestamp_continuity_detection(df)
