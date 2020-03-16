#!usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author:jinxin
@time: 2020/01/14
@Project : PycharmProjects
"""
import pandas as pd


def codename_to_columns(dataframe):
    df = dataframe
    df['Time'] = df['Date'] + 'T' + df['Time']
    df_pivot = df.pivot(values='Value', index='Time', columns='CodeName')
    df_pivot['Time'] = df_pivot.index
    df_pivot = pd.DataFrame(data=df_pivot.values, columns=list(df_pivot.columns))
    df = df.drop(columns=['CodeName', 'Date', 'Value']).drop_duplicates(subset=['Time'])
    dfm = df.merge(df_pivot, left_on='Time', right_on='Time')
    return dfm


