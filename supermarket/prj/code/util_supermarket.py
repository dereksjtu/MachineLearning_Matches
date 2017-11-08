# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

from feature_process import *


# 评分函数
def score(y_test,y_pred):
    return 1.0 / (1.0 + np.sqrt(mean_squared_error(y_test, y_pred)))
import xgboost as xgb

def cross_valid(regressor, train_data,valid_data,train_feature,l_roll_feats):
    valid_dates = valid_data['SaleDate'].unique()
    # valid_salecount = valid_data[[['Class','SaleDate','saleCount']]
    valid_template = valid_data[['Class','SaleDate']].copy()
    feats = [f for f in train_data.columns if f not in l_roll_feats]
    batch_0 = valid_data[valid_data['SaleDate']==pd.to_datetime(valid_dates[0])].copy()
    batch_0['saleCount'] = 0
    xgbvalid = xgb.DMatrix(batch_0[train_feature])
    batch_0['saleCount'] = regressor.predict(xgbvalid)
    result = batch_0[['Class','SaleDate','saleCount']]
    train_data = pd.concat([train_data,batch_0],axis=0)
    for index in range(1,len(valid_dates)):
        train_data = train_data[feats]
        batch = valid_data[valid_data['SaleDate'] == pd.to_datetime(valid_dates[index])].copy()
        batch = batch[feats]
        train_data = pd.concat([train_data,batch],axis=0)
        train_data,l_roll_feats = get_roll_feats(train_data)
        train_data.fillna(0,inplace=True)
        batch = train_data[train_data['SaleDate'] == pd.to_datetime(valid_dates[index])].copy()
        batch['saleCount'] = 0
        xgbvalid = xgb.DMatrix(batch[train_feature])
        batch['saleCount'] = regressor.predict(xgbvalid)
        result_i = batch[['Class','SaleDate','saleCount']]
        result = pd.concat([result,result_i],axis=0)
        # xgbvalid = xgb.DMatrix(batch[train_feature])
        # batch = regressor.predict(xgbvalid)
        train_data['saleCount'][train_data['SaleDate'].isin([valid_dates[index]])] = batch['saleCount']
    result = pd.merge(valid_template[['Class','SaleDate']], result,on = ['Class','SaleDate'],how='left')
    result.fillna(0,inplace=True)
    score_pre = score(valid_data['saleCount'],result['saleCount'])
    print score_pre
    return score_pre


def feature_vis(importance, train_feature):
    importances = importance
    indices = np.argsort(importances)[::-1]
    selected_features = [train_feature[e] for e in indices]
    plt.figure(figsize=(20, 10))
    plt.title("train_feature importances")
    plt.bar(range(len(train_feature)), importances[indices],
            color="r", align="center")
    plt.xticks(range(len(selected_features)), selected_features, rotation=70)
    plt.show()