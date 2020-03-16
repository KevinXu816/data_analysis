#!usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author:jinxin
@time: 2020/01/19
@Project : PycharmProjects
"""
import json
import numpy as np
import pandas as pd


def full_process(df, table):
    df = codename_to_columns(df, table)
    df = dtype_convert(df, table)
    df = missing_value_processing(df)
    return df


def codename_to_columns(df, table):
    df['Time'] = df['Date'] + 'T' + df['Time']
    df_pivot = df.pivot(values='Value', index='Time', columns='CodeName')
    df_pivot['Time'] = df_pivot.index
    df_pivot = pd.DataFrame(data=df_pivot.values, columns=list(df_pivot.columns))
    df = df.drop(columns=['CodeName', 'time', 'Date', 'Value', 'OrderID', 'WorkID']).drop_duplicates(subset=['Time'])
    df = df.merge(df_pivot, left_on='Time', right_on='Time')
    df = dtype_convert(df, table)
    return df


def dtype_convert(df, table):
    json_data = open(r"C:\Users\X1\PycharmProjects\Project01\DataProcess\dtypes.json", encoding='utf-8').read()
    dtypes = json.loads(json_data).get(table)
    columns = list(df.columns)
    df = df.replace('', np.NAN)
    for column in columns:
        df[column] = df[column].astype(dtypes.get(column), errors='ignore')
    return df


def missing_value_processing(df):
    columns = list(df.columns)
    ratio = df.isnull().sum(axis=0) / df.shape[0]
    for column in columns:
        if ratio[column] > 0.5:
            df.drop(columns=column, inplace=True)
            continue
        if str(df[column].dtypes).find('float') == 0:
            df[column].fillna(df[column].mean(), inplace=True)
        if str(df[column].dtypes).find('int') == 0:
            df[column].fillna(df[column].mean(), inplace=True)
        if df[column].dtype == bool:
            df[column].fillna(df[column].mode(), inplace=True)  # 众数替换
            df[column] = df[column].astype('int')  # True/False转换为0，1
        df.dropna()
    return df
