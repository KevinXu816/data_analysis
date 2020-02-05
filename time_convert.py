#!/usr/bin/python
# encoding:utf-8
import time
from datetime import datetime
import pandas as pd


def format_to_timestamp(formatTime):
    timeArray = time.strptime(formatTime, "%Y-%m-%dT%H:%M:%S")
    timestamp = time.mktime(timeArray)
    return timestamp


def timestamp_to_format(timestamp):
    time_local = time.localtime(timestamp)
    formatTime = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
    return formatTime


def change_format(formatTime):
    timeArray = time.strptime(formatTime, "%d.%m.%Y %H:%M:%S")
    timestamp = time.mktime(timeArray)
    time_local = time.localtime(timestamp)
    formatTime = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
    return formatTime


# print(format_to_timestamp('2020-01-03 18:16:59'))
# print(timestamp_to_format(time.time()))
# print(format_to_timestamp('2019-12-29T16:18:11'))
print(format_to_timestamp('2019-12-30T00:11:15'))
# print(timestamp_to_format(1577607073))
#
# file = r"C:\Users\X1\Downloads\Dr_Hartmann_Project\52692_MWS_output.csv"
# dd_MWS = pd.read_csv(file)
#
# dd_MWS['Datum'] = dd_MWS.iloc[:, 0].apply(lambda x: change_format(x))
# print(dd_MWS)
# dd_MWS.to_csv(file)

# begin = datetime.now().timestamp()
# end = datetime.now().timestamp()
# print(begin)
# print(end)
# k = end - begin
# print(k)
