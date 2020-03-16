#!usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author:jinxin
@time: 2019/12/31
@Project : PycharmProjects
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

np.random.seed(1)  # 设定随机种子，保证实验可复现


class lstm():

    def __init__(self, dataset, hyper_params):
        self.scaler = MinMaxScaler(feature_range=(0, 1))  # 归一化
        self.dataset = self.scaler.fit_transform(dataset)  # 数据集初始化归一化
        self.num_neur = hyper_params.get('num_neur')  # 初始化隐层数和各层神经元个数
        self.look_back = hyper_params.get('look_back')  # 初始化窗口长度
        self.forward = hyper_params.get('forward')  # 预测后多少时间
        self.epochs = hyper_params.get('epochs')  # 初始化训练次数
        self.batch_size = hyper_params.get('batch_size')  # 初始化批数
        self.train_ratio = hyper_params.get('train_ratio')  # 初始化训练集分割比例
        self.feature_num = hyper_params.get('feature_num')  # 初始化特征数量
        self.y_true = []
        self.x_train = []  # 初始化训练集x部分-训练特征
        self.y_train = []  # 初始化训练集y部分-监督信号
        self.x_test = []  # 初始化测试集x部分-测试特征
        self.y_test = []  # 初始化测试集y部分-监督信号
        self.trainPredict = []
        self.testPredict = []

    # 分割训练集与测试集
    def split_dataset(self):
        # 转换数据结构，准备训练集与测试集
        def create_dataset(dataset, look_back, forward):
            dataX, dataY = [], []
            for i in range(len(dataset) - look_back - forward):
                a = dataset[i:(i + look_back), 0:dataset.shape[1]]
                dataX.append(a)
                dataY.append(dataset[i + look_back + forward, 0])
            return np.array(dataX), np.array(dataY)

        train_size = int(len(self.dataset) * self.train_ratio)
        train_data = self.dataset[0:train_size + self.look_back + self.forward, :]
        test_data = self.dataset[train_size:len(self.dataset), :]

        # 具体分割后数据集
        x_all, self.y_true = create_dataset(self.dataset, self.look_back, self.forward)
        x_train, self.y_train = create_dataset(train_data, self.look_back, self.forward)
        x_test, self.y_test = create_dataset(test_data, self.look_back, self.forward)
        print(self.y_true.shape)
        print(self.y_train.shape)
        print(self.y_test.shape)
        print('sssss')

        # Reshape input to be [samples, feature_num, features]
        self.x_train = np.reshape(x_train, (x_train.shape[0], self.feature_num, x_train.shape[1]))
        self.x_test = np.reshape(x_test, (x_test.shape[0], self.feature_num, x_test.shape[1]))

    def inverse_scaler(self, trainPredict, testPredict):
        datasety_like = np.zeros(shape=(self.dataset.shape[0], self.dataset.shape[1]))
        datasety_like[:, 0] = self.dataset[:, 0]
        self.y_true = self.scaler.inverse_transform(datasety_like)[:, 0]

        # 将预测值反标准化到正常值
        trainPredict_dataset_like = np.zeros(shape=(len(trainPredict), self.dataset.shape[1]))
        trainPredict_dataset_like[:, 0] = trainPredict[:, 0]  # 将预测值填充进新建数组
        trainPredict = self.scaler.inverse_transform(trainPredict_dataset_like)[:, 0]  # 数据转换

        y_train_dataset_like = np.zeros(shape=(len(self.y_train), self.dataset.shape[1]))
        y_train_dataset_like[:, 0] = self.y_train
        self.y_train = self.scaler.inverse_transform(y_train_dataset_like)[:, 0]

        testPredict_dataset_like = np.zeros(shape=(len(testPredict), self.dataset.shape[1]))
        testPredict_dataset_like[:, 0] = testPredict[:, 0]
        testPredict = self.scaler.inverse_transform(testPredict_dataset_like)[:, 0]

        y_test_dataset_like = np.zeros(shape=(len(self.y_test), self.dataset.shape[1]))
        y_test_dataset_like[:, 0] = self.y_test
        self.y_test = self.scaler.inverse_transform(y_test_dataset_like)[:, 0]

        return trainPredict, testPredict

    # 创建并拟合LSTM网络
    def lstm(self):
        start_cr_a_fit_net = time.time()  # 记录网络创建与训练时间
        self.split_dataset()  # 数据分割

        # 创建并拟合LSTM网络
        LSTM_model = Sequential()
        for i in range(len(self.num_neur)):  # 构建多层网络
            if len(self.num_neur) == 1:
                LSTM_model.add(LSTM(self.num_neur[i], input_shape=(None, self.look_back)))
            else:
                if i < len(self.num_neur) - 1:
                    LSTM_model.add(LSTM(self.num_neur[i], input_shape=(None, self.look_back), return_sequences=True))
                else:
                    LSTM_model.add(LSTM(self.num_neur[i], input_shape=(None, self.look_back)))

        LSTM_model.add(Dense(1))
        LSTM_model.summary()  # Summary the structure of neural network/网络结构总结
        LSTM_model.compile(loss='mean_squared_error', optimizer='adam')  # Compile the neural network/编译网络
        LSTM_model.fit(self.x_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size
                       , verbose=0)  # Fit the LSTM network/拟合LSTM网络
        end_cr_a_fit_net = time.time() - start_cr_a_fit_net
        print('Running time of creating and fitting the LSTM network: %.2f Seconds' % (end_cr_a_fit_net))

        # LSTM prediction/LSTM进行预测
        trainPredict = LSTM_model.predict(self.x_train)  # Predict by training data set
        testPredict = LSTM_model.predict(self.x_test)  # Predict by Temp data set

        # 将预测值反标准化到正常值
        self.trainPredict, self.testPredict = self.inverse_scaler(trainPredict, testPredict)

        return self.trainPredict, self.testPredict, self.y_train, self.y_test, self.y_true

    # 可视化结果
    def plot(self, ylabel, title, start_time, end_time):
        # 转换数据结构用于作图-训练预测结果
        shape_set = [[data] for data in self.dataset[:, 0]]
        print(shape_set)
        trainPredictPlot = np.empty_like(shape_set)
        trainPredictPlot[:, 0] = np.nan
        trainPredictPlot[self.look_back + self.forward:len(self.trainPredict) + self.look_back + self.forward,
        0] = self.trainPredict

        # 转换数据结构用于作图-测试预测结果
        testPredictPlot = np.empty_like(shape_set)
        testPredictPlot[:] = np.nan
        testPredictPlot[len(self.trainPredict) + self.look_back + self.forward:len(self.dataset), 0] = self.testPredict

        # 作图
        datasety_like = np.zeros(shape=(self.dataset.shape[0], self.dataset.shape[1]))
        datasety_like[:, 0] = self.dataset[:, 0]
        y = self.scaler.inverse_transform(datasety_like)[:, 0]

        xs = pd.date_range(start=start_time, end=end_time, periods=len(y))

        A, = plt.plot(xs, y[0:len(y)], linewidth='2', color='r')  # 真实值
        B, = plt.plot(xs, trainPredictPlot, linewidth='1.5', color='g')  # LSTM训练集结果
        C, = plt.plot(xs, testPredictPlot, linewidth='1.5', color='c')  # LSTM测试集结果

        plt.legend((A, B, C), ('real_value', 'LSTM_train', 'LSTM_test'), loc='best')
        plt.gcf().autofmt_xdate()  # 自动旋转日期标记

        plt.xlabel('Times', family='Times New Roman', fontsize=16)  # X轴
        plt.ylabel(ylabel, family='Times New Roman', fontsize=16)  # Y轴

        plt.title(title, family='Times New Roman', fontsize=16)  # 添加标题

        # plt.savefig(r'C:\Users\10321\Desktop\result.png', dpi=900)  # 保存图片

        plt.show()
        del trainPredictPlot, testPredictPlot
