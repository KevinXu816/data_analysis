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
import time
import numpy as np


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
        conn.close()
        return result

    def query_all(self, tablename):
        sql = 'select * from ' + '\"' + tablename + '\"' + ';'
        result = self.query(sql)
        return result

    def query_df(self, sql):
        result = self.query(sql)
        df = pd.DataFrame(list(result.get_points()))
        df.replace('', np.NAN, inplace=True)
        return df

    def query_all_df(self, tablename):
        sql = 'select * from ' + '\"' + tablename + '\"' + ';'
        result = self.query(sql)
        df = pd.DataFrame(list(result.get_points()))
        df.replace('', np.NAN, inplace=True)
        return df

    def query_all_df_limit(self, tablename, rows):
        sql = 'select * from ' + '\"' + tablename + '\"' + 'limit ' + str(rows) + ';'
        result = self.query(sql)
        df = pd.DataFrame(list(result.get_points()))
        df.replace('', np.NAN, inplace=True)
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
        conn.close()

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
        conn.close()

    def show_all_tables(self):
        sql = 'show measurements;'
        result = self.query(sql)
        table_list = [a.get('name') for a in list(result.get_points())]
        return table_list

    def count_rows(self, tablename):
        sql = 'select count(time) from ' + '\"' + tablename + '\"' + ';'
        count = self.query(sql)
        return count

    def drop_table(self, tablename):
        sql = 'drop measurement ' + '\"' + tablename + '\"' + ';'
        self.query(sql)

    def delete_table_data(self, tablename):
        sql = 'delete from ' + '\"' + tablename + '\"' + ';'
        self.query(sql)

    def time_series_simulation(self, df, tablename):
        conn = self.get_connect()
        for i in range(len(df)):
            json_body = []
            a = {
                "measurement": tablename,
                "tags": {"id": i},
                "fields": df.iloc[i, :].to_dict()
            }
            json_body.append(a)
            conn.write_points(json_body)
            print(i)
            time.sleep(1)
        conn.close()
