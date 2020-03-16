#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:X1
@time: 2019/12/04
"""
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

import numpy as np


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))


def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)


def rmse(y_true, y_pred):
    return math.sqrt(mse(y_true, y_pred))


def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)


def score_print_write(y_true, y_pred, algorithm_and_leadtime, table_and_target, samples_timecost, pathwrite):
    print('-------------------Evaluation--------------------------')
    print('Table and Target_col: ' + table_and_target[0] + ', ' + table_and_target[1])
    print('Algorithm and lead_time: ' + algorithm_and_leadtime[0] + ', ' + algorithm_and_leadtime[1])
    print('Mean Absolute Error:  %.3f' % mae(y_true, y_pred))
    # print('Mean Absolute Percentage Error:  %.3f' % mape(y_true, y_pred))
    print('Mean Squared Error:  %.3f' % mse(y_true, y_pred))
    print('Root Mean Squared Error:  %.3f' % rmse(y_true, y_pred))
    print('r2 score:  %.3f' % r2(y_true, y_pred))
    print('Training size: %d,  Time cost:  %.3f' % (samples_timecost[0], samples_timecost[1]))

    f = open(pathwrite, 'a+')
    f.write("-----------------Evaluation----------------------------\n")
    f.write('Table and Target_col:   ' + table_and_target[0] + ', ' + table_and_target[1] + '\n')
    f.write('Algorithm and lead_time:   ' + algorithm_and_leadtime[0] + ', ' + algorithm_and_leadtime[1] + '\n')
    f.write('Training size:  %d, Training size:  %d,  Time cost:   %.3f' % (samples_timecost[0], samples_timecost[1], samples_timecost[2]) + '\n')
    f.write('Mean Absolute Error:   %.3f' % mae(y_true, y_pred) + '\n')
    # f.write('Mean Absolute Percentage Error:   %.3f' % mape(y_true, y_pred) + '\n')
    f.write('Mean Squared Error:   %.3f' % mse(y_true, y_pred) + '\n')
    f.write('Root Mean Squared Error:   %.3f' % rmse(y_true, y_pred) + '\n')
    f.write('r2 score:   %.3f' % r2(y_true, y_pred) + '\n')
    f.write('\n')
    f.write('\n')
    f.write('\n')

    f.close()
