#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:jinxin
@time: 2019/12/17 15:40
@Project : Project01
"""

import pandas as pd


def create_time_feature(df, fea_list):
    # 把字符串格式的时间转换成Timestamp格式
    df['time'] = df.iloc[:, 0].apply(lambda x: pd.Timestamp(x))

    # 年份
    if 'year' in fea_list:
        df['year'] = df['time'].apply(lambda x: x.year)

    # 月份
    if 'month' in fea_list:
        temp = pd.get_dummies(df['time'].apply(lambda x: x.month), prefix='month')
        df = pd.concat([df, temp], axis=1)

    # 日
    if 'day' in fea_list:
        df['day'] = df['time'].apply(lambda x: x.day)

    # 小时
    if 'hour' in fea_list:
        df['hour'] = df['time'].apply(lambda x: x.hour)

    # 分钟
    if 'minute' in fea_list:
        df['minute'] = df['time'].apply(lambda x: x.minute)

    # 秒数
    if 'second' in fea_list:
        df['second'] = df['time'].apply(lambda x: x.second)

    # 一天中的第几分钟
    if 'minuteOfDay' in fea_list:
        df['minuteOfDay'] = df['time'].apply(lambda x: x.minute + x.hour * 60)

    # 星期几；
    if 'dayOfWeek' in fea_list:
        df['dayOfWeek'] = df['time'].apply(lambda x: x.dayofweek)

    # 一年中的第几天
    if 'dayOfYear' in fea_list:
        df['dayOfYear'] = df['time'].apply(lambda x: x.dayofyear)

    # 一年中的第几周
    if 'weekOfYear' in fea_list:
        df['weekOfYear'] = df['time'].apply(lambda x: x.week)

    # 一天中哪个时间段：凌晨:1、早晨:2、上午:3、中午:4、下午:5、傍晚:6、晚上:7、深夜:0；
    period_dict = {
        23: 0, 0: 0, 1: 0,
        2: 1, 3: 1, 4: 1,
        5: 2, 6: 2, 7: 2,
        8: 3, 9: 3, 10: 3, 11: 3,
        12: 4, 13: 4,
        14: 5, 15: 5, 16: 5, 17: 5,
        18: 6,
        19: 7, 20: 7, 21: 7, 22: 7,
    }
    if 'periodOfDay' in fea_list:
        df['temp'] = df['time'].apply(lambda x: x.hour)
        df['periodOfDay'] = df['temp'].map(period_dict)

    # 季节，春季：0，夏季：1，秋季：2，冬季：3
    season_dict = {
        1: 0, 2: 0, 3: 0,
        4: 1, 5: 1, 6: 1,
        7: 2, 8: 2, 9: 2,
        10: 3, 11: 3, 12: 3,
    }
    if 'season' in fea_list:
        df['temp'] = df['time'].apply(lambda x: x.month)
        df['season'] = df['temp'].map(season_dict)

    # 是否闰年
    if 'isLeapYear' in fea_list:
        df['isLeapYear'] = df['time'].apply(lambda x: 1 if x.is_leap_year else 0)

    # 是否月初
    if 'isMonthStart' in fea_list:
        df['isMonthStart'] = df['time'].apply(lambda x: 1 if x.is_month_start else 0)

    # 是否月末
    if 'isMonthEnd' in fea_list:
        df['isMonthEnd'] = df['time'].apply(lambda x: 1 if x.is_month_end else 0)

    # 是否季节初
    if 'isQuarterStart' in fea_list:
        df['isQuarterStart'] = df['time'].apply(lambda x: 1 if x.is_quarter_start else 0)

    # 是否季节末
    if 'isQuarterEnd' in fea_list:
        df['isQuarterEnd'] = df['time'].apply(lambda x: 1 if x.is_quarter_end else 0)

    # 是否年初
    if 'isYearStart' in fea_list:
        df['isYearStart'] = df['time'].apply(lambda x: 1 if x.is_year_start else 0)

    # 是否年尾
    if 'isYearEnd' in fea_list:
        df['isYearEnd'] = df['time'].apply(lambda x: 1 if x.is_year_end else 0)

    # 是否周末
    if 'isWeekend' in fea_list:
        df['isWeekend'] = df['time'].apply(lambda x: 1 if x.dayofweek in [5, 6] else 0)

    # 是否公共假期
    public_vacation_list = [
        '20190101', '20190102', '20190204', '20190205', '20190206',
        '20190207', '20190208', '20190209', '20190210', '20190405',
        '20190406', '20190407', '20190501', '20190502', '20190503',
        '20190504', '20190607', '20190608', '20190609', '20190913',
        '20190914', '20190915', '20191001', '20191002', '20191003',
        '20191004', '20191005', '20191006', '20191007',
    ]  # 此处未罗列所有公共假期
    df['format_date'] = df['time'].apply(lambda x: x.strftime('%Y%m%d'))
    df['isHoliday'] = df['format_date'].apply(lambda x: 1 if x in public_vacation_list else 0)

    # 是否工作时间
    df['isWorkingTime'] = 0
    df['temp'] = df['time'].apply(lambda x: x.hour)
    df.loc[((df['temp'] >= 8) & (df['temp'] < 22)), 'isWorkingTime'] = 1

    return df


if __name__ == "__main__":
    # 构造时间数据
    date_time_str_list = [
        '2019-01-01 01:22:26', '2019-02-02 04:34:52', '2019-03-03 06:16:40',
        '2019-04-04 08:11:38', '2019-05-05 10:52:39', '2019-06-06 12:06:25',
        '2019-07-07 14:05:25', '2019-08-08 16:51:33', '2019-09-09 18:28:28',
        '2019-10-10 20:55:12', '2019-11-11 22:55:12', '2019-12-12 00:55:12',
    ]
    df = pd.DataFrame({'时间': date_time_str_list})
    ds = create_time_feature(df, ['month', 'periodOfDay', 'isYearStart'])
    print(ds.columns)
    print(ds[['time', 'month_1', 'month_2', 'month_3', 'month_4', 'month_5',
       'month_6', 'month_7', 'month_8', 'month_9', 'month_10', 'month_11',
       'month_12',]])
