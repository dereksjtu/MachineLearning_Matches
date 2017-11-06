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
from dataClean_supermarket import *

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
    # train_o,train_new_o = reshape_train(train_o)
    train = train_o[train_o['SaleDate'] >= 20150101]
    # train = train[train['SaleDate'] <= 20150331]
    train = train[train['SaleDate'] <= 20150331]
    train_new = train_new_o[train_new_o['SaleDate'] >= 20150101]
    # train_new = train_new[train_new['SaleDate'] <= 20150331]
    train_new = train_new[train_new['SaleDate'] <= 20150331]
    test = train_new_o[train_new_o['SaleDate'] >= 20150401]
    del test['parClass']
    return train ,train_new, test

if __name__ == "__main__":

    week_4 = [
                ['2015-04-01','2015-04-02','2015-04-03','2015-04-04','2015-04-05','2015-04-06','2015-04-07'],
                ['2015-04-08','2015-04-10','2015-04-11','2015-04-12','2015-04-13','2015-04-14'],
                ['2015-04-15','2015-04-17','2015-04-18','2015-04-19','2015-04-20','2015-04-21'],
                ['2015-04-22','2015-04-23','2015-04-24','2015-04-25','2015-04-26','2015-04-27','2015-04-28'],
                ['2015-04-29','2015-04-30']

                # ['2015-04-01','2015-04-02','2015-04-03','2015-04-04','2015-04-05','2015-04-06','2015-04-07',
                # '2015-04-08','2015-04-09','2015-04-10','2015-04-11','2015-04-12','2015-04-13','2015-04-14'],
                # ['2015-04-15','2015-04-16','2015-04-17','2015-04-18','2015-04-19','2015-04-20','2015-04-21',
                # '2015-04-22','2015-04-23','2015-04-24','2015-04-25','2015-04-26','2015-04-27','2015-04-28'],
                # ['2015-04-29','2015-04-30']

                # ['2015-04-01'],['2015-04-02'],['2015-04-03'],['2015-04-04'],['2015-04-05'],['2015-04-06'],['2015-04-07'],
                # ['2015-04-08'],['2015-04-09'],['2015-04-10'],['2015-04-11'],['2015-04-12'],['2015-04-13'],['2015-04-14'],
                # ['2015-04-15'],['2015-04-16'],['2015-04-17'],['2015-04-18'],['2015-04-19'],['2015-04-20'],['2015-04-21'],
                # ['2015-04-22'],['2015-04-23'],['2015-04-24'],['2015-04-25'],['2015-04-26'],['2015-04-27'],['2015-04-28'],
                # ['2015-04-29'],['2015-04-30']

                # ['2015-04-01','2015-04-02','2015-04-03','2015-04-04'],
                # ['2015-04-05','2015-04-06','2015-04-07','2015-04-08'],
                # ['2015-04-09','2015-04-10','2015-04-11','2015-04-12'],
                # ['2015-04-13','2015-04-14','2015-04-15','2015-04-16'],
                # ['2015-04-17','2015-04-18','2015-04-19','2015-04-20'],
                # ['2015-04-21','2015-04-22','2015-04-23','2015-04-24'],
                # ['2015-04-25','2015-04-26','2015-04-27','2015-04-28'],
                # ['2015-04-29','2015-04-30']

                # ['2015-04-01'],['2015-04-02'],['2015-04-03'],['2015-04-04'],['2015-04-05'],['2015-04-06'],['2015-04-07'],
                # ['2015-04-08'],['2015-04-09'],['2015-04-10'],['2015-04-11'],['2015-04-12'],['2015-04-13'],['2015-04-14'],
                # ['2015-04-15'],['2015-04-16'],['2015-04-17'],['2015-04-18'],['2015-04-19'],['2015-04-20'],['2015-04-21'],
                # ['2015-04-22'],['2015-04-23'],['2015-04-24'],['2015-04-25'],['2015-04-26'],['2015-04-27'],['2015-04-28'],
                # ['2015-04-29'],['2015-04-30']

                # ['2015-04-01','2015-04-02','2015-04-03','2015-04-04'],
                # ['2015-04-06','2015-04-07','2015-04-08','2015-04-09','2015-04-10'],
                # ['2015-04-11','2015-04-12','2015-04-13','2015-04-14','2015-04-15'],
                # ['2015-04-16','2015-04-17','2015-04-18','2015-04-19','2015-04-20'],
                # ['2015-04-21','2015-04-22','2015-04-23','2015-04-24','2015-04-25'],
                # ['2015-04-26','2015-04-27']
        # # ,'2015-04-28''2015-04-29','2015-04-30']

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
                ['2015-05-29','2015-05-30']

                # ['2015-05-01'],['2015-05-02'],['2015-05-03'],['2015-05-04'],['2015-05-05'],['2015-05-06'],['2015-05-07'],
                # ['2015-05-08'],['2015-05-09'],['2015-05-10'],['2015-05-11'],['2015-05-12'],['2015-05-13'],['2015-05-14'],
                # ['2015-05-15'],['2015-05-16'],['2015-05-17'],['2015-05-18'],['2015-05-19'],['2015-05-20'],['2015-05-21'],
                # ['2015-05-22'],['2015-05-23'],['2015-05-24'],['2015-05-25'],['2015-05-26'],['2015-05-27'],['2015-05-28'],
                # ['2015-05-29'],['2015-05-30']
            ]


    # do_not_use_class = [1516,3005,3424]
    # do_not_use_class = [1507]
    # perform extreamly bad
    do_not_use_class = [
# 11,
# 3013,
# 1308,
# 2207,
# 3018,
# 2013,
# 31,
# 1202,
# 2210,
# 1521,
# 2008,
# 2205,
# 2014,
# 2204,
# 1001,
# 2206,
# 13,
# 10,
# 3016,
# 1518,
# 2011,
# 2202,
# 1505,
# 2203,
# 2201,
# 23,
# 30,
# 1203,
15,
20,
# 1201,
22,
12
]

#     t0 = time.time()
#     train_o = pd.read_csv(train_path,encoding='gbk',engine='python')
#     test_o = pd.read_csv(test_path)
#     train_o,train_new_o = reshape_train(train_o)
#     # 验证集中不预测的类
#     # train_o, train_new_o = exclude_class(train_new_o, train_o, do_not_use_class)
#
#
#
#     # 验证 train为2,3月份， test为4月份数据
#     train ,train_new, test = train_test_split(train_o,train_new_o)
#     test.loc[:,'saleCount'] = 0
#
#     # 特征1： 提取固定特征
#     train_new = exclude_abnormal_value(train_new)
#     print 'Filter abnormal value.'
#     train_new_o.SaleDate = train_new_o.SaleDate.map(lambda x: timeHandle(x))
#     train_new_o.SaleDate = pd.to_datetime(train_new_o.SaleDate)
#     train, train_new, test = get_origin_feats(train, train_new, test)
#
#
#     # # 分离测试集
#     start_date = '2015-01-01'
#     # test_1, test_2, test_3, test_4, test_5 ,test = test_split(train_new, test, week_4)
#     test_1 = test[test['SaleDate'].isin(week_4[0])]
#     # 分离验证集
#     # test_valid_1,test_valid_2,test_valid_3,test_valid_4,test_valid_5, test_valid = valid_split(train_new_o, week_4, month_4)
#     test_valid = valid_split(train_new_o, week_4, month_4)
#     #验证集取4月1号到4月28号
#     # 特征2： 提取滚动特征
#     train_test = merge_train_test(train_new, test_1)
#     train_test,l_roll_feats = get_roll_feats(train_test)
#
#
#     train_feat = train_test[train_test['SaleDate'] >= start_date]    #使用二月份以后的数据
#     train_feat_1 = train_feat[train_feat['SaleDate'] < '2015-04-01']   #训练集为2-3月份
#
#     #排除过年的异常值
#     # exclude_date = ['2015-02-16','2015-02-17','2015-02-18']
#     # train_feat_1 = train_feat_1[~train_feat_1['SaleDate'].isin(exclude_date)]   #训练集为2-3月份
#
#     test_feat = train_test[train_test['SaleDate'] >= '2015-04-01']
#     test_feat = test_feat[test_feat['SaleDate'] <= '2015-04-30'] #验证集为四月份
#     test_feat.loc[:,'saleCount'] = 0
#
#
#     train_feat_1.fillna(0,inplace=True)
#     test_feat.fillna(0,inplace=True)
#
#
#     test_feat_1 = test_feat[test_feat['SaleDate'].isin(week_4[0])]
#     test_feat_1.fillna(0,inplace=True)
#     test_feat_1.loc[:,'saleCount'] = 0
#     feature_names = list(train_feat_1.columns)
#     do_not_use_for_training = ['SaleDate','saleCount','Coupon',
#                                'dayOfYear','price_mean','price_median',
#                                # 'parCumtype','parClass',
#                                # 'parHotPast1MonthIndex',
#                                # 'dayOn21DayDiff',
#                                'lastWeekSaleCount_mean',
#                                # 'expweighted_14_avg',
#                                'trend_7','expweighted_7_avg',
#                                'classWeekdayRatio',
#                                'moving_30_avg','expweighted_30_avg',
#                                # 'disholDaySaleCount_mean',
#                                'disholDaySaleCount_max',
#                                # 'holDaySaleCount_min',
#                                # 'holDaySaleCount_median',
#                                # 'holDaySaleCount_mean','month','parHotIndex','dayOn5DayDiff'
#                                ]
#     predictors = [f for f in feature_names if f not in do_not_use_for_training]
#
#     params = {'min_child_weight': 100, 'eta': 0.09, 'colsample_bytree': 0.3, 'max_depth': 7,
#                 'subsample': 0.85, 'lambda': 1, 'nthread': 4, 'booster' : 'gbtree', 'silent': 1,
#                 'eval_metric': 'rmse', 'objective': 'reg:linear'}
#     boostRound = 100
#
#     xgbtrain = xgb.DMatrix(train_feat_1[predictors], train_feat_1['saleCount'])
#     xgbvalid = xgb.DMatrix(test_feat_1[predictors])
#     model = xgb.train(params, xgbtrain, num_boost_round=boostRound)
#     param_score = pd.Series(model.get_fscore()).sort_values(ascending=False)
#     print "Parameter score: "
#     print param_score, len(predictors)
#     test_feat_1.loc[:,'saleCount'] = model.predict(xgbvalid)
#     result = test_feat_1[['Class','SaleDate','saleCount']]
#     # result['saleCount'] = 1.3 *  result['saleCount']
#     test_valid_1 = test_valid[test_valid['SaleDate'].isin(week_4[0])]
#     test_valid_1.fillna(0,inplace=True)
#     result = pd.merge(test_valid_1[['Class','SaleDate']], result, on=['Class','SaleDate'], how='left')
#     # result['saleCount'][result['saleCount'] < 0] = 0
#     result.fillna(0,inplace=True)
#
#     score_1 = score(result['saleCount'],test_valid_1['saleCount'])
#     print "the 1th day predictive score:{}".format(score_1)
#
#
#     # 第二轮
#     # 特征2： 提取滚动特征
#     # 4月2号 - 4月30号
#     for i in range(1,5):
#         # l_roll_feats = []
#         train_test = train_test[train_test['SaleDate'] < '2015-04-01']
#         train_test = merge_train_test(train_test, test_feat_1)
#         feats = [f for f in train_test.columns if f not in l_roll_feats]
#         train_test = train_test[feats]
#
#         test_i = test[test['SaleDate'].isin(week_4[i])]
#         train_test = merge_train_test(train_test, test_i)
#         train_test,l_roll_feats = get_roll_feats(train_test)
#
#         train_feat = train_test[train_test['SaleDate'] >= start_date]    #使用二月份以后的数据
#         # train_feat_1 = train_feat[train_feat['SaleDate'] < '2015-04-08']   #训练集为2-3月份
#         # 排除掉这轮要预测的日期
#         train_feat = train_feat[~train_feat['SaleDate'].isin(week_4[i])]
#         # print week_4[i]
#         # print train_feat['SaleDate']
#
#         test_feat = train_test[train_test['SaleDate'] >= '2015-04-01']
#         test_feat = test_feat[test_feat['SaleDate'] <= '2015-04-30'] #验证集为四月份
#         test_feat.loc[:,'saleCount'] = 0
#
#
#         train_feat_1.fillna(0,inplace=True)
#         test_feat.fillna(0,inplace=True)
#
#         test_feat_1 = test_feat[test_feat['SaleDate'].isin(week_4[i])]
#         # print test_feat_1['SaleDate']
#         test_feat_1.fillna(0,inplace=True)
#         test_feat_1.loc[:,'saleCount'] = 0
#         feature_names = list(train_feat_1.columns)
#         # do_not_use_for_training = ['SaleDate','saleCount','Coupon',
#         #                            'dayOfYear','price_mean','price_median'
#         #                            ]
#         predictors = [f for f in feature_names if f not in do_not_use_for_training]
#
#         # params = {'min_child_weight': 100, 'eta': 0.09, 'colsample_bytree': 0.3, 'max_depth': 7,
#         #         'subsample': 0.85, 'lambda': 1, 'nthread': 4, 'booster' : 'gbtree', 'silent': 1,
#         #         'eval_metric': 'rmse', 'objective': 'reg:linear'}
#
#         xgbtrain = xgb.DMatrix(train_feat_1[predictors], train_feat_1['saleCount'])
#         xgbvalid = xgb.DMatrix(test_feat_1[predictors])
#         model = xgb.train(params, xgbtrain, num_boost_round=boostRound)
#
#         test_feat_1.loc[:,'saleCount'] = model.predict(xgbvalid)
#         result_i = test_feat_1[['Class','SaleDate','saleCount']]
#         # result_i['saleCount'] = 1.2 *  result_i['saleCount']
#         print len(result_i['saleCount'])
#         test_valid_i = test_valid[test_valid['SaleDate'].isin(week_4[i])]
#         test_valid_i.fillna(0,inplace=True)
#         result_i = pd.merge(test_valid_i[['Class','SaleDate']], result_i, on=['Class','SaleDate'], how='left')
#         print week_4[i]
#         # result['saleCount'][result['saleCount'] < 0] = 0
#         result_i.fillna(0,inplace=True)
#         score_i = score(test_valid_i['saleCount'],result_i['saleCount'])
#         result = pd.concat([result, result_i], axis=0)
#         print "the {}th day predictive score:{}".format(i + 1,score_i)
#
# result = pd.merge(test_valid[['Class','SaleDate']], result, on=['Class','SaleDate'], how='left')
# result['saleCount'][result['saleCount'] < 0] = 0
#
#
# result.to_csv('result.csv',index=False)
# test_valid.to_csv('test_valid.csv',index=False)
# score_f = score(test_valid['saleCount'],result['saleCount'])
#
# print "Elapse time is {} minutes".format((time.time() - t0) / (1.0 * 60))
# print "Total predictive score:{}".format(score_f)
#
# # ser_score = pd.Series([score(test_valid['saleCount'][test_valid['Class'] == i],result['saleCount'][result['Class'] == i]) for i in result.columns],index=result.columns)
#
# # for further analyse
# l_index = []
# l_score = []
# for i in test_valid['Class'].unique():
#     score_1 = score(test_valid['saleCount'][test_valid['Class'] == i] ,result['saleCount'][result['Class'] == i])
#     l_index.append(i)
#     l_score.append(score_1)
# ser_score = pd.Series(l_score, index=l_index)
# ser_score.to_csv('ser_score.csv')



    ### 验证 4月份数据

#     ########################## 提交训练结果 #####################################
#
    t0 = time.time()
    # do_not_use_class = [1507,3208,3311,3413]
    train_o = pd.read_csv(train_path,encoding='gbk',engine='python')
    test_o = pd.read_csv(test_path)
    train_o,train_new_o = reshape_train(train_o)
    # test_o, train_new_o = exclude_class(train_new_o, test_o, do_not_use_class)

    # train_new_o.SaleDate = train_new_o.SaleDate.map(lambda x: timeHandle(x))
    # train_new_o.SaleDate = pd.to_datetime(train_new_o.SaleDate)
    # test_o.SaleDate = test_o.SaleDate.map(lambda x: timeHandle(x))
    # test_o.SaleDate = pd.to_datetime(train_new_o.SaleDate)

    test_o.loc[:,'saleCount'] = 0

    # 特征1： 提取固定特征
    train, train_new, test = get_origin_feats(train_o, train_new_o, test_o)
    train_new = exclude_abnormal_value(train_new)
    print 'Filter abnormal value.'


    test_5 = test[test['SaleDate'].isin(week_5[0])]
    # 特征2： 提取滚动特征
    train_test = merge_train_test(train_new, test_5)
    train_test,l_roll_feats = get_roll_feats(train_test)

    train_feat = train_test[train_test['SaleDate'] >= '2015-01-01']    #使用二月份以后的数据
    train_feat_1 = train_feat[train_feat['SaleDate'] < '2015-05-01']   #训练集为2-3月份

    test_feat = train_test[train_test['SaleDate'] >= '2015-05-01']
    test_feat = test_feat[test_feat['SaleDate'] <= '2015-05-31'] #验证集为四月份
    # test_valid = test_valid[test_valid['SaleDate'] < '2015-04-27']
    test_feat.loc[:,'saleCount'] = 0


    train_feat_1.fillna(0,inplace=True)
    test_feat.fillna(0,inplace=True)

    test_feat_1 = test_feat[test_feat['SaleDate'].isin(week_5[0])]
    test_feat_1.fillna(0,inplace=True)
    test_feat_1.loc[:,'saleCount'] = 0
    feature_names = list(train_feat_1.columns)
    do_not_use_for_training = ['SaleDate','saleCount','Coupon',
                               'dayOfYear','price_mean','price_median',
                               # 'parCumtype','parClass',
                               # 'parHotPast1MonthIndex',
                               # 'dayOn21DayDiff',
                               'lastWeekSaleCount_mean',
                               # 'expweighted_14_avg',
                               'trend_7','expweighted_7_avg',
                               'classWeekdayRatio',
                               'moving_30_avg','expweighted_30_avg',
                               # 'disholDaySaleCount_mean',
                               'disholDaySaleCount_max',
                               # 'holDaySaleCount_min',
                               # 'holDaySaleCount_median',
                               # 'holDaySaleCount_mean','month','parHotIndex','dayOn5DayDiff'
                               ]
    predictors = [f for f in feature_names if f not in do_not_use_for_training]

    params = {'min_child_weight': 100, 'eta': 0.09, 'colsample_bytree': 0.3, 'max_depth': 8,
                'subsample': 0.85, 'lambda': 1, 'nthread': 4, 'booster' : 'gbtree', 'silent': 1,
                'eval_metric': 'rmse', 'objective': 'reg:linear'}
    boostRound = 100

    xgbtrain = xgb.DMatrix(train_feat_1[predictors], train_feat_1['saleCount'])
    xgbvalid = xgb.DMatrix(test_feat_1[predictors])
    model = xgb.train(params, xgbtrain, num_boost_round=boostRound)
    param_score = pd.Series(model.get_fscore()).sort_values(ascending=False)
    print "Parameter score: "
    print param_score, len(predictors)
    test_feat_1.loc[:,'saleCount'] = model.predict(xgbvalid)
    result = test_feat_1[['Class','SaleDate','saleCount']]
    test_valid_1 = test[test['SaleDate'].isin(week_5[0])]
    # test_valid_1.fillna(0,inplace=True)
    result = pd.merge(test_valid_1[['Class','SaleDate']], result, on=['Class','SaleDate'], how='left')
    result.fillna(0,inplace=True)
    # print result
    # result['saleCount'][result['saleCount'] < 0] = 0
    # score_1 = score(result['saleCount'],test_valid_1['saleCount'])
    # print "the 1th day predictive score:{}".format(score_1)


    # 第二轮
    # 特征2： 提取滚动特征
    # 4月2号 - 4月28号
    for i in range(1,5):
        # l_roll_feats = []
        train_test = train_test[train_test['SaleDate'] < '2015-05-01']
        train_test = merge_train_test(train_test, test_feat_1)
        feats = [f for f in train_test.columns if f not in l_roll_feats]
        train_test = train_test[feats]

        test_i = test[test['SaleDate'].isin(week_5[i])]
        train_test = merge_train_test(train_test, test_i)
        train_test,l_roll_feats = get_roll_feats(train_test)

        train_feat = train_test[train_test['SaleDate'] >= '2015-01-01']    #使用二月份以后的数据
        # train_feat_1 = train_feat[train_feat['SaleDate'] < '2015-04-08']   #训练集为2-3月份
        # 排除掉这轮要预测的日期
        train_feat = train_feat[~train_feat['SaleDate'].isin(week_5[i])]
        # print week_4[i]
        # print train_feat['SaleDate']

        test_feat = train_test[train_test['SaleDate'] >= '2015-05-01']
        test_feat = test_feat[test_feat['SaleDate'] <= '2015-05-30'] #验证集为四月份
        test_feat.loc[:,'saleCount'] = 0


        train_feat_1.fillna(0,inplace=True)
        test_feat.fillna(0,inplace=True)

        test_feat_1 = test_feat[test_feat['SaleDate'].isin(week_5[i])]
        # print test_feat_1['SaleDate']
        test_feat_1.fillna(0,inplace=True)
        test_feat_1.loc[:,'saleCount'] = 0
        feature_names = list(train_feat_1.columns)
        # do_not_use_for_training = ['SaleDate','saleCount','Coupon',
        #                            'dayOfYear','price_mean','price_median'
        #                            ]
        predictors = [f for f in feature_names if f not in do_not_use_for_training]

        # params = {'min_child_weight': 100, 'eta': 0.09, 'colsample_bytree': 0.3, 'max_depth': 7,
        #             'subsample': 0.6, 'lambda': 1, 'nthread': 4, 'booster' : 'gbtree', 'silent': 1,
        #             'eval_metric': 'rmse', 'objective': 'reg:linear'}

        xgbtrain = xgb.DMatrix(train_feat_1[predictors], train_feat_1['saleCount'])
        xgbvalid = xgb.DMatrix(test_feat_1[predictors])
        model = xgb.train(params, xgbtrain, num_boost_round=boostRound)
        param_score = pd.Series(model.get_fscore()).sort_values(ascending=False)
        # print "Parameter score: "
        # print param_score
        test_feat_1.loc[:,'saleCount'] = model.predict(xgbvalid)
        result_i = test_feat_1[['Class','SaleDate','saleCount']]
        print len(result_i['saleCount'])
        test_valid_i = test[test['SaleDate'].isin(week_5[i])]
        test_valid_i.fillna(0,inplace=True)
        result_i = pd.merge(test_valid_i[['Class','SaleDate']], result_i, on=['Class','SaleDate'], how='left')
        result_i.fillna(0,inplace=True)
        print week_5[i]
        # result['saleCount'][result['saleCount'] < 0] = 0
        # print result
        # score_i = score(test_valid_i['saleCount'],result_i['saleCount'])
        result = pd.concat([result, result_i], axis=0)
        # print result
        # print "the {}th day predictive score:{}".format(i + 1,score_i)

del test['saleCount']
result = pd.merge(test[['Class','SaleDate']], result, on=['Class','SaleDate'], how='left')
result['saleCount'][result['saleCount'] < 0] = 0
result['saleCount'] = result['saleCount'].astype('int')
result['Class'] = result['Class'].astype('int')
result['SaleDate'] = test_o['SaleDate']
result.to_csv('result.csv',index=False)
result.rename(columns={'Class':u'编码','saleCount':u'销量','SaleDate':u'日期'},inplace=True)
result.to_csv('result.csv',index=False,encoding='gbk')
# test_valid.to_csv('test_valid.csv',index=False)
# score = score(test_valid['saleCount'],result['saleCount'])
print "Elapse time is {} minutes".format((time.time() - t0) / (1.0 * 60))
# print "Total predictive score:{}".format(score)

    ########################## 提交训练结果 ####################################