# -*- coding:utf-8 -*-
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

import time
# Handle table like and matrices
import pandas as pd
import numpy as np

# Support functions
from feature_supermarket import *

# Machine tool kits
import xgboost as xgb
from sklearn.metrics import mean_squared_error

train_path = '../input/train.csv'
test_path = '../input/test.csv'
hol_path = '../input/holiday.csv'
train_date_path = '../input/train_date.csv'
cache_path = '../input/cache/'
output_path = '../output/'

# 评分函数
def score(y_test,y_pred):
    return 1.0 / (1.0 + np.sqrt(mean_squared_error(y_test, y_pred)))

# train = pd.read_csv(train_path,encoding='gbk',engine='python')
# test = pd.read_csv(test_path)

def train_test_split(train_o,train_new_o):
    train_o,train_new_o = reshape_train(train_o)
    train = train_o[train_o['SaleDate'] >= 20150201]
    train = train[train['SaleDate'] <= 20150331]
    train_new = train_new_o[train_new_o['SaleDate'] >= 20150201]
    train_new = train_new[train_new['SaleDate'] <= 20150331]
    test = train_new_o[train_new_o['SaleDate'] >= 20150401]
    del test['parClass']
    return train ,train_new, test

if __name__ == "__main__":

    week_4 = [
                ['2015-04-01','2015-04-02','2015-04-03','2015-04-04','2015-04-05','2015-04-06','2015-04-07'],
                ['2015-04-08','2015-04-09','2015-04-10','2015-04-11','2015-04-12','2015-04-13','2015-04-14'],
                ['2015-04-15','2015-04-16','2015-04-17','2015-04-18','2015-04-19','2015-04-20','2015-04-21'],
                ['2015-04-22','2015-04-23','2015-04-24','2015-04-25','2015-04-26','2015-04-27','2015-04-28'],
                ['2015-04-29','2015-04-30']
            ]
    month_4 = [
                '2015-04-01','2015-04-02','2015-04-03','2015-04-04','2015-04-05','2015-04-06','2015-04-07',
                '2015-04-08','2015-04-09','2015-04-10','2015-04-11','2015-04-12','2015-04-13','2015-04-14',
                '2015-04-15','2015-04-16','2015-04-17','2015-04-18','2015-04-19','2015-04-20','2015-04-21',
                '2015-04-22','2015-04-23','2015-04-24','2015-04-25','2015-04-26','2015-04-27','2015-04-28',
                '2015-04-29','2015-04-30'
            ]
    week_5 = [
                ['2015-05-01','2015-05-02','2015-05-03','2015-05-04','2015-05-05','2015-05-06','2015-05-07'],
                ['2015-05-08','2015-05-09','2015-05-10','2015-05-11','2015-05-12','2015-05-13','2015-05-14'],
                ['2015-05-15','2015-05-16','2015-05-17','2015-05-18','2015-05-19','2015-05-20','2015-05-21'],
                ['2015-05-22','2015-05-23','2015-05-24','2015-05-25','2015-05-26','2015-05-27','2015-05-28'],
                ['2015-05-29','2015-05-30','2015-05-31']
            ]

    t0 = time.time()
    train_o = pd.read_csv(train_path,encoding='gbk',engine='python')
    test_o = pd.read_csv(test_path)
    train_o,train_new_o = reshape_train(train_o)

    train_new_o.SaleDate = train_new_o.SaleDate.map(lambda x: timeHandle(x))
    train_new_o.SaleDate = pd.to_datetime(train_new_o.SaleDate)

    ### 验证 train为2,3月份， test为4月份数据
    train ,train_new, test = train_test_split(train_o,train_new_o)
    test.loc[:,'saleCount'] = 0

    # 特征1： 提取固定特征
    train, train_new, test = get_origin_feats(train, train_new, test)

    # 分离测试集
    test_1, test_2, test_3, test_4, test_5 ,test = test_split(train_new, test, week_4)
    # 分离验证集
    test_valid_1,test_valid_2,test_valid_3,test_valid_4,test_valid_5, test_valid = valid_split(train_new_o, week_4, month_4)
    # 特征2： 提取滚动特征
    train_test = merge_train_test(train_new, test_1)
    train_test = get_roll_feats(train_test)

    train_feat = train_test[train_test['SaleDate'] >= '2015-02-01']    #使用二月份以后的数据
    train_feat_1 = train_feat[train_feat['SaleDate'] < '2015-04-01']   #训练集为2-3月份

    test_feat = train_test[train_test['SaleDate'] >= '2015-04-01']
    test_feat = test_feat[test_feat['SaleDate'] <= '2015-04-30'] #验证集为四月份
    test_feat.loc[:,'saleCount'] = 0

    train_feat_1.fillna(0,inplace=True)
    test_feat.fillna(0,inplace=True)

    test_feat_1 = test_feat[test_feat['SaleDate'].isin(week_4[0])]
    test_feat_1.fillna(0,inplace=True)
    test_feat_1['saleCount'] = 0
    feature_names = list(train_feat_1.columns)
    do_not_use_for_training = ['SaleDate','saleCount','dayOfYear','dayOn7DayDiff','parWeekOn1WeekRatio','holDaySaleCount_mean']
    predictors = [f for f in feature_names if f not in do_not_use_for_training]
    # print predictors, len(predictors)
    xgbtrain = xgb.DMatrix(train_feat_1[predictors], train_feat_1['saleCount'])
    # xgbvalid = xgb.DMatrix(test_feat_1[predictors], test_feat_1['saleCount'])
    xgbvalid = xgb.DMatrix(test_feat_1[predictors])
    # watchlist = [(xgbtrain, 'train'), (xgbvalid, 'valid')]

    # params = {'min_child_weight': 10, 'eta': 0.1, 'colsample_bytree': 0.3, 'max_depth': 8,
    #             'subsample': 0.8, 'lambda': 1., 'nthread': 4, 'booster' : 'gbtree', 'silent': 1,
    #             'eval_metric': 'rmse', 'objective': 'reg:linear'}
    params = {'min_child_weight': 100, 'eta': 0.09, 'colsample_bytree': 0.3, 'max_depth': 11,
                'subsample': 0.6, 'lambda': 1., 'nthread': 4, 'booster' : 'gbtree', 'silent': 1,
                'eval_metric': 'rmse', 'objective': 'reg:linear'}
    xgbtrain = xgb.DMatrix(train_feat_1[predictors], train_feat_1['saleCount'])
    xgbvalid = xgb.DMatrix(test_feat_1[predictors])
    model = xgb.train(params, xgbtrain, num_boost_round=100)
    param_score = pd.Series(model.get_fscore()).sort_values(ascending=False)
    print "Parameter score: "
    print param_score
    test_feat_1.loc[:,'saleCount'] = model.predict(xgbvalid)
    test_feat_1['saleCount'] = test_feat_1['saleCount'].astype('int')
    result = test_feat_1[['Class','SaleDate','saleCount']]
    result = pd.merge(test_valid_1[['Class','SaleDate']], result, on=['Class','SaleDate'], how='left')
    score_1 = score(result['saleCount'],test_valid_1['saleCount'])

    print "first 7 days predictive score:{}".format(score_1)


    ### 验证 4月份数据
