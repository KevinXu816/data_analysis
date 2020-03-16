#!usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author:jinxin
@time: 2020/03/08
@Project : PycharmProjects
"""
import json
from main_LSTM import neural_network
import Evaluation
import sx_data_preprocess as sdp
from operate_influxdb import operate_influxdb
from class_moving_series import moving_series
from create_time_feat_one_hot import create_time_feature
import pandas as pd


def run_LSTM(df, table, target_col, lead_time):
    """
    Feature Engineering
    """
    # 生成时间特征
    df = create_time_feature(df, ['hour', 'periodOfDay', 'dayOfWeek'])
    print(df.columns)
    train_size = df.shape[0] * 0.7
    test_size = df.shape[0] * 0.3
    df.drop(columns=['State', 'Time'], inplace=True)

    # 生成统计特征
    columns = list(df.columns)
    use_col = []
    json_data = open(r"C:\Users\X1\PycharmProjects\Project01\DataProcess\dtypes.json", encoding='utf-8').read()
    dtypes = json.loads(json_data).get(table)
    for column in columns:
        if dtypes.get(column) != 'bool':  # ?是否还有其他类型是Int的列是有限离散值
            use_col.append(column)
    cms = moving_series(df, use_col, windows=[20], y_col=target_col)
    select_list = ['corr', 'mean', 'min', 'max', 'medium', 'var', 'ewm_mean', 'double_ewm']
    df = cms.select_create(select_list)
    df = sdp.missing_value_processing(df)
    print(df.columns)

    """
    Train and Test the Model.
    """
    obj_NN = neural_network(df, target_col=target_col, lead_time=15)
    obj_NN.split_dataset()
    inv_yhat_test, inv_y_test, time_cost = obj_NN.NN_model()

    """
    Save the result.
    """
    save_path = r'C:\Users\X1\Desktop\data_50000_result\\' + table + '.txt'
    Evaluation.score_print_write(inv_y_test, inv_yhat_test, ['LSTM', str(lead_time)],
                                 [table, target_col], [int(train_size), int(test_size), time_cost], save_path)


table = '4偏心1'
target_col = 'PX1V5'

"""
Get data and pre process.
"""
oi = operate_influxdb('myDB')
print(oi.show_all_tables())
codenameNum = oi.query_all_df_limit(table, 1000).CodeName.nunique()
rows = 50000 * codenameNum
df = oi.query_all_df_limit(tablename=table, rows=rows)
df = sdp.full_process(df, table)
# df = pd.read_csv(r'C:\Users\X1\Desktop\data_50000\\' + table + '.csv')
# df = sdp.dtype_convert(df, table)
# df = sdp.missing_value_processing(df)
print(df.dtypes)
print(df.columns)

for lead_time in [30, 60, 150, 300, 600]:
    run_LSTM(df, table, target_col, lead_time)
