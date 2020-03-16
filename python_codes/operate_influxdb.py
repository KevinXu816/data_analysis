#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:jinxin
@time: 2019/12/26 14:45
@Project : Project01
"""
from __future__ import unicode_literals

import pandas as pd
from influxdb import InfluxDBClient


class operate_influxdb:

    def __init__(self, database, ip='localhost', port=8086, username='admin', password='admin'):
        self.ip = ip
        self.port = port
        self.username = username
        self.password = password
        self.database = database

    def get_connect(self):
        conn = InfluxDBClient(self.ip, self.port, self.username, self.password, self.database)
        return conn

    def query(self, sql):
        conn = self.get_connect()
        result = conn.query(sql)
        return result

    def query_all(self, tablename):
        sql = 'select * from ' + '\"' + tablename + '\"' + ';'
        result = self.query(sql)
        return result

    def query_df(self, sql, tablename):
        result = self.query(sql)
        df = pd.DataFrame(list(result.get_points(measurement=tablename)))
        return df

    def query_all_df(self, tablename):
        sql = 'select * from ' + '\"' + tablename + '\"' + ';'
        result = self.query(sql)
        df = pd.DataFrame(list(result.get_points(measurement=tablename)))
        return df

    def insert_points(self, point_dict_list, tablename):
        conn = self.get_connect()
        json_body = []
        for i in range(len(point_dict_list)):
            a = {
                "measurement": tablename,
                "tags": {"id": i},
                "fields": point_dict_list[i]
            }
            json_body.append(a)
        conn.write_points(json_body)

    def insert_df(self, df, tablename):
        conn = self.get_connect()
        length = len(df)
        batch = 10000
        n = int(length / batch + 1)
        rem = length % batch

        for j in range(n):
            if j != n - 1:
                json_body = []
                for i in range(batch):
                    index = j * batch + i
                    a = {
                        "measurement": tablename,
                        "tags": {"id": index},
                        "fields": df.iloc[index, :].to_dict()
                    }
                    json_body.append(a)
                conn.write_points(json_body)
            else:
                json_body = []
                for i in range(rem):
                    index = j * batch + i
                    a = {
                        "measurement": tablename,
                        "tags": {"id": index},
                        "fields": df.iloc[index, :].to_dict()
                    }
                    json_body.append(a)
                conn.write_points(json_body)

    def show_all_tables(self):
        sql = 'show measurements;'
        result = self.query(sql)
        table_list = [a.get('name') for a in list(result.get_points())]
        return table_list

    def drop_table(self, tablename):
        sql = 'drop measurement ' + '\"' + tablename + '\"' + ';'
        self.query(sql)

    def delete_table_data(self, tablename):
        sql = 'delete from ' + '\"' + tablename + '\"' + ';'
        self.query(sql)

# if __name__ == "__main__":
# df = pd.DataFrame({'Col1': [10, 20, 15, 30, 45], 'Col2': [13, 23, 18, 33, 48], 'Col3': [17, 27, 22, 37, 52]})
# oi = operate_influxdb('testdatadb')
# oi.insert_df(df, 'test02')
# # oi.insert_points([{'Col1': 20, 'Col2': 23, 'Col3': 27}], 'test05')
#
# data = oi.query('select Col1, Col2 from test02;')
# print(data)
# data = oi.query_df('select Col1, Col2 from test02;', 'test02')
# print(data)
