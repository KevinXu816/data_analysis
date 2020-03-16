#!usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author:jinxin
@time: 2020/03/06
@Project : PycharmProjects
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')


class feature_selection():
    def __init__(self, data, k, target_col, method='pca', pca_radio=0.99):
        self.data = data
        self.method = method
        self.target_col = target_col
        self.k = k
        self.pca_ratio = pca_radio

    def feature_selection_correlation(self, df):
        k = self.k

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # select features with high correlation
        ch2 = SelectKBest(chi2, k)
        X_1 = ch2.fit_transform(X, y.astype('int'))
        list_ = ch2.get_support(indices=True).tolist()
        df1 = pd.DataFrame(X_1, columns=[df.columns[i] for i in list_])
        df1 = pd.concat([df1, y], axis=1)
        df1 = df1.rename(columns={0: self.target_col})

        print('Columns after selections are', df1.columns)

        return df1

    def rf_importance(self, df):
        k = self.k

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        rf = RandomForestRegressor()
        rf.fit(X, y)

        ranking = sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), df.columns.to_list()), reverse=True)[
                  :k]
        print(ranking)

        co_list = []
        for i in range(k):
            co_list.append(ranking[i][1])

        print('Columns after selections are', co_list)

        return df[co_list]

    def rfe_function(self, df):
        k = self.k

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        lr = Ridge(alpha=100000, fit_intercept=True, normalize=True, copy_X=True, max_iter=1500, tol=1e-4,
                   solver='auto')
        rfe = RFE(estimator=lr, n_features_to_select=k)
        rfe.fit_transform(X, y)
        ranking = sorted(zip(rfe.ranking_, X.columns.to_list()), reverse=True)[:k]

        co_list = []
        for i in range(k):
            co_list.append(ranking[i][1])

        print('Columns after selections are', co_list)

        return df[co_list]

    def preprocess(self, pca_ratio):
        data = self.data.dropna()

        print('number of columns before feat selection is', self.data.shape[1])
        method = self.method

        # define X&y
        y = data[self.target_col]
        X = data[data.columns.difference([self.target_col])]

        # normalize data
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = self.scaler.fit_transform(np.c_[X, y])

        y_scaled = pd.DataFrame(self.data[:, -1])
        X_scaled = self.data[:, :-1]

        # convert array to df to fit in feature_selection_correlation and rf_importance function
        df2 = pd.DataFrame(X_scaled, columns=[data.columns[i] for i in range(data.shape[1] - 1)])
        df2 = pd.concat([df2, y_scaled], axis=1)
        df2 = df2.rename(columns={0: self.target_col})

        global df

        if method == 'corr':
            df = self.feature_selection_correlation(df2)
        if method == 'rf':
            df = self.rf_importance(df2)
        if method == 'rfe':
            df = self.rfe_function(df2)
        if method == 'pca':
            pca = PCA(n_components=pca_ratio)
            X_pca = pca.fit_transform(X_scaled)
            print(pca.explained_variance_ratio_)
            print(pca.explained_variance_)
            df = pd.DataFrame(np.c_[X_pca, y_scaled])

        print('number of columns after feat selection is', df.shape[1])
        return df.values
