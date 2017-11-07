# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

from feature_supermarket import *


# 评分函数
def score(y_test,y_pred):
    return 1.0 / (1.0 + np.sqrt(mean_squared_error(y_test, y_pred)))

def cross_valid(regressor, train_data,valid_data,train_feature,l_roll_feats):
    # print valid_data['SaleDate']
    valid_dates = valid_data['SaleDate'].unique()
    valid_salecount = valid_data['saleCount']
    feats = [f for f in train_data.columns if f not in l_roll_feats]
    # valid_data = valid_data[feats]
    batch_0 = valid_data[valid_data['SaleDate']==pd.to_datetime(valid_dates[0])].copy()
    batch_0['saleCount'] = 0
    batch_0['saleCount'] = regressor.predict(batch_0[train_feature].values)
    y_pre = batch_0['saleCount']
    train_data = pd.concat([train_data,batch_0],axis=0)
    for index in range(1,len(valid_dates)):
        train_data = train_data[feats]
        batch = valid_data[valid_data['SaleDate']==pd.to_datetime(valid_dates[index])].copy()
        batch = batch[feats]
        train_data = pd.concat([train_data,batch],axis=0)
        train_data,l_roll_feats = get_roll_feats(train_data)
        batch = train_data[train_data['SaleDate']==pd.to_datetime(valid_dates[index])].copy()
        y_pre_i = regressor.predict(batch[train_feature].values)
        train_data['saleCount'][train_data['SaleDate'].isin([valid_dates[index]])] = y_pre_i
        y_pre = np.concatenate((y_pre, y_pre_i))
    return score(valid_salecount,y_pre)


# def cross_valid(regressor, bucket, lagging):
#     valid_loss = []
#     last = [[] for i in range(len(bucket[bucket.keys()[0]]))]
#     for time_series in sorted(bucket.keys(), key=float):
#         if time_series >= 120:
#             if int(time_series) in range(120, 120 + lagging * 2, 2):
#                 last = np.concatenate((last, np.array(bucket[time_series], dtype=float)[:, -1].reshape(-1, 1)), axis=1)
#             else:
#                 batch = np.array(bucket[time_series], dtype=float)
#                 y = batch[:, -1]
#                 batch = np.delete(batch, -1, axis=1)
#                 batch = np.concatenate((batch, last), axis=1)
#                 y_pre = regressor.predict(batch)
#                 last = np.delete(last, 0, axis=1)
#                 last = np.concatenate((last, y_pre.reshape(-1, 1)), axis=1)
#                 loss = np.mean(abs(np.expm1(y) - np.expm1(y_pre)) / np.expm1(y))
#                 valid_loss.append(loss)
#     # print 'day: %d loss: %f' % (int(day), day_loss)
#     return np.mean(valid_loss)


def feature_vis(regressor, train_feature):
    importances = regressor.feature_importances
    indices = np.argsort(importances)[::-1]
    selected_features = [train_feature[e] for e in indices]
    plt.figure(figsize=(20, 10))
    plt.title("train_feature importances")
    plt.bar(range(len(train_feature)), importances[indices],
            color="r", align="center")
    plt.xticks(range(len(selected_features)), selected_features, rotation=70)
    plt.show()



