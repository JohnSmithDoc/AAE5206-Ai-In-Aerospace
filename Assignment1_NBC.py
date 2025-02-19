#   Student Number: 24066688g
#   Student   Name: Yang Guodong

import warnings
import numpy as np
from collections import defaultdict
from sklearn.datasets import load_iris


class NBC:
    def __init__(self, feature_types, num_classes):
        self.feature_types = feature_types
        self.num_classes = num_classes
        # 用于存放使用训练数据求取的每个类的先验概率
        self.class_priors = None
        # 用于存放3个类条件下的4个特征（连续量）的概率分布，所以一共有12个分布，每一个类条件下有4个独立的分布，均假设为高斯分布
        self.pdf_params = defaultdict(lambda: defaultdict(dict))

    # 计算类的先验概率
    def calculate_classes_priors(self, y):
        self.class_priors = np.zeros(self.num_classes)
        np_y = np.array(y)
        N = len(np_y)

        # 计算三个分类的先验概率
        for c in range(self.num_classes):
            # y是个一维数组，过滤出数组中分别为0,1,2的元素的个数，即3个类的个数
            mask = (np_y == c)  # 此处得到一个bool数组，通过sum可以得到个数
            self.class_priors[c] = np.sum(mask) / N

        # debug 查看下类的先验概率计算结果
        # print('classes priors probabilities are:', self.class_priors)

    # 计算3个条件下4个特征(相互独立)的概率密度函数的参数，一共12个
    def calculate_pdfs_params(self, X, y):
        shape = X.shape
        # 获取特征的列数
        columns = shape[1]
        # 如果比初始化时得到的feature_type的特征多，则告警，这要求初始化时feature_type一定要初始化合理
        if columns > len(self.feature_types):
            warnings.warn("features are out of limit!!!", RuntimeWarning)
            columns = len(self.feature_types)

        for c_index in range(self.num_classes):
            # 找到 类分别为0 1 2 的对应的X的行，临时构成一个多维数组
            matching_rows = X[y == c_index]

            # debug
            # print('matching rows:', matching_rows)

            for f_index in range(columns):
                # 再提取指定列的数据
                column_data = matching_rows[:, f_index]
                # 计算均值和方差
                mean = np.mean(column_data)
                var = np.var(column_data)

                # 方差不能为0
                if var == 0:
                    var = 1e-6

                # 存到重要参数中，行代表类的索引，列代表特征的索引，例如params[0][1]表示的是在类为0的条件下特征1的概率分布（高斯分布的两个重要参数）
                self.pdf_params[c_index][f_index] = {
                    'mean': mean,
                    'var': var
                }

        # debug
        # print('pdfs are ', self.pdf_params)

    # 计算此新样本特征已观测到的条件下的类别的后验概率
    def calculate_posterior(self, x):
        # 存放计算出的后验概率
        posterior_log = np.zeros(self.num_classes)
        # 这里获取特征的列数
        columns = len(x)
        # print('input x col is:', columns)

        for c_i in range(self.num_classes):
            # 类的先验
            prior_log = np.log(self.class_priors[c_i])

            con_prob_log = 0
            # 特征的条件概率
            for f_i in range(columns):
                mean = self.pdf_params[c_i][f_i]['mean']
                var = self.pdf_params[c_i][f_i]['var']
                # 计算某个样本在特定类条件下的条件概率密度P(X=X_test|y = Ck) = 连乘 P(x_1 = x_test_1|y = Ck)，对数空间变为加法
                con_prob_log += -0.5 * (np.log(2 * np.pi * var)) + (-0.5 * ((x[f_i] - mean) ** 2) / var)

            # 存放了每一个类的后验概率，一共三个，P(y=Ck|X=X_test) = P(y=Ck)*(X=X_test|y = Ck)，转换为对数空间即变为加法
            posterior_log[c_i] = prior_log + con_prob_log

        return posterior_log

    # 训练
    def fit(self, X, y):
        # 1、计算类的先验概率
        self.calculate_classes_priors(y)
        # 2、计算条件概率密度函数的参数
        self.calculate_pdfs_params(X, y)

    # 分类预测
    def predict(self, X):
        class_indexes_arr = []

        for x in X:
            # 计算每一类的后验概率，然后取出后验概率最大的类的index
            posterior_log = self.calculate_posterior(x)
            class_indexes_arr.append(np.argmax(posterior_log))

        return np.array(class_indexes_arr)


################################
if __name__ == "__main__":
    # 加载数据,ref:https://blog.csdn.net/qq_46626684/article/details/123820726
    iris = load_iris()
    X, y = iris['data'], iris['target']

    # 数据洗牌并准备训练数据和测试数据
    N, D = X.shape
    Ntrain = int(0.75 * N)
    shuffler = np.random.permutation(N)
    Xtrain = X[shuffler[:Ntrain]]
    ytrain = y[shuffler[:Ntrain]]
    Xtest = X[shuffler[Ntrain:]]
    ytest = y[shuffler[Ntrain:]]

    # 初始化朴素贝叶斯分类器
    nbc = NBC(feature_types=['r', 'r', 'r', 'r'], num_classes=3)

    # 训练
    nbc.fit(Xtrain, ytrain)

    # 训练精度
    train_yhat = nbc.predict(Xtrain)
    train_accuracy = np.mean(train_yhat == ytrain)
    print('ytrain is', ytrain)
    print('train_yhat is', train_yhat)
    print(f"Training accuracy is: {train_accuracy:.6f}, Training error is: {(1 - train_accuracy):.6f}")

    # 测试精度
    yhat = nbc.predict(Xtest)
    test_accuracy = np.mean(yhat == ytest)
    print('yhat is', yhat)
    print('ytest is', ytest)
    print(f"Testing accuracy: {test_accuracy:.6f}, Testing error is: {(1 - test_accuracy):.6f}")
