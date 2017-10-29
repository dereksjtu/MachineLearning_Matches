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
    train_o,train_new_o = reshape_train(train_o)
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
                ['2015-04-08','2015-04-09','2015-04-10','2015-04-11','2015-04-12','2015-04-13','2015-04-14'],
                ['2015-04-15','2015-04-16','2015-04-17','2015-04-18','2015-04-19','2015-04-20','2015-04-21'],
                ['2015-04-22','2015-04-23','2015-04-24','2015-04-25','2015-04-26','2015-04-27','2015-04-28'],
                ['2015-04-29','2015-04-30']

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
                ['2015-05-29','2015-05-30','2015-05-31']
            ]

    do_not_use_class = [1516,3005,3424]

    t0 = time.time()
    train_o = pd.read_csv(train_path,encoding='gbk',engine='python')
    test_o = pd.read_csv(test_path)
    train_o,train_new_o = reshape_train(train_o)
    # 验证集中不预测的类
    # train_o, train_new_o = exclude_class(train_new_o, train_o, do_not_use_class)

    train_new_o.SaleDate = train_new_o.SaleDate.map(lambda x: timeHandle(x))
    train_new_o.SaleDate = pd.to_datetime(train_new_o.SaleDate)

    # 验证 train为2,3月份， test为4月份数据
    train ,train_new, test = train_test_split(train_o,train_new_o)
    test.loc[:,'saleCount'] = 0

    # 特征1： 提取固定特征
    train, train_new, test = get_origin_feats(train, train_new, test)
    # train_new = exclude_abnormal_value(train_new)
    print 'Filter abnormal value.'


    # # 分离测试集
    # test_1, test_2, test_3, test_4, test_5 ,test = test_split(train_new, test, week_4)
    test_1 = test[test['SaleDate'].isin(week_4[0])]
    # 分离验证集
    # test_valid_1,test_valid_2,test_valid_3,test_valid_4,test_valid_5, test_valid = valid_split(train_new_o, week_4, month_4)
    test_valid = valid_split(train_new_o, week_4, month_4)
    #验证集取4月1号到4月28号
    # test_valid = test_valid[test_valid['SaleDate'] < '2015-04-29']
    # 特征2： 提取滚动特征
    train_test = merge_train_test(train_new, test_1)
    train_test,l_roll_feats = get_roll_feats(train_test)

    train_feat = train_test[train_test['SaleDate'] >= '2015-01-01']    #使用二月份以后的数据
    train_feat_1 = train_feat[train_feat['SaleDate'] < '2015-04-01']   #训练集为2-3月份

    test_feat = train_test[train_test['SaleDate'] >= '2015-04-01']
    test_feat = test_feat[test_feat['SaleDate'] <= '2015-04-30'] #验证集为四月份
    test_feat.loc[:,'saleCount'] = 0


    train_feat_1.fillna(0,inplace=True)
    test_feat.fillna(0,inplace=True)

    test_feat_1 = test_feat[test_feat['SaleDate'].isin(week_4[0])]
    test_feat_1.fillna(0,inplace=True)
    test_feat_1.loc[:,'saleCount'] = 0
    feature_names = list(train_feat_1.columns)
    do_not_use_for_training = ['SaleDate','saleCount','Coupon',
                               'dayOfYear','price_mean','price_median',
                               'parHotPast1MonthIndex',
                               'dayOn21DayDiff',
                               'lastWeekSaleCount_mean'
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
    result.fillna(0,inplace=True)
    test_valid_1 = test_valid[test_valid['SaleDate'].isin(week_4[0])]
    test_valid_1.fillna(0,inplace=True)
    result = pd.merge(test_valid_1[['Class','SaleDate']], result, on=['Class','SaleDate'], how='left')
    # result['saleCount'][result['saleCount'] < 0] = 0
    score_1 = score(result['saleCount'],test_valid_1['saleCount'])
    # print "first 7 days predictive score:{}".format(score_1)
    print "the 1th day predictive score:{}".format(score_1)


    # 第二轮
    # 特征2： 提取滚动特征
    # 4月2号 - 4月28号
    for i in range(1,5):
        # l_roll_feats = []
        train_test = train_test[train_test['SaleDate'] < '2015-04-01']
        train_test = merge_train_test(train_test, test_feat_1)
        feats = [f for f in train_test.columns if f not in l_roll_feats]
        train_test = train_test[feats]

        test_i = test[test['SaleDate'].isin(week_4[i])]
        train_test = merge_train_test(train_test, test_i)
        train_test,l_roll_feats = get_roll_feats(train_test)

        train_feat = train_test[train_test['SaleDate'] >= '2015-01-01']    #使用二月份以后的数据
        # train_feat_1 = train_feat[train_feat['SaleDate'] < '2015-04-08']   #训练集为2-3月份
        # 排除掉这轮要预测的日期
        train_feat = train_feat[~train_feat['SaleDate'].isin(week_4[i])]
        # print week_4[i]
        # print train_feat['SaleDate']

        test_feat = train_test[train_test['SaleDate'] >= '2015-04-01']
        test_feat = test_feat[test_feat['SaleDate'] <= '2015-04-30'] #验证集为四月份
        test_feat.loc[:,'saleCount'] = 0


        train_feat_1.fillna(0,inplace=True)
        test_feat.fillna(0,inplace=True)

        test_feat_1 = test_feat[test_feat['SaleDate'].isin(week_4[i])]
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
        result_i.fillna(0,inplace=True)
        test_valid_i = test_valid[test_valid['SaleDate'].isin(week_4[i])]
        test_valid_i.fillna(0,inplace=True)
        result_i = pd.merge(test_valid_i[['Class','SaleDate']], result_i, on=['Class','SaleDate'], how='left')
        print week_4[i]
        # result['saleCount'][result['saleCount'] < 0] = 0
        # print result
        score_i = score(test_valid_i['saleCount'],result_i['saleCount'])
        result = pd.concat([result, result_i], axis=0)
        print "the {}th day predictive score:{}".format(i + 1,score_i)

result = pd.merge(test_valid[['Class','SaleDate']], result, on=['Class','SaleDate'], how='left')
result['saleCount'][result['saleCount'] < 0] = 0
result.to_csv('result.csv',index=False)
test_valid.to_csv('test_valid.csv',index=False)
score = score(test_valid['saleCount'],result['saleCount'])
print "Total predictive score:{}".format(score)



    ### 验证 4月份数据


    # ### 提交训练结果
    # train_o = pd.read_csv(train_path,encoding='gbk',engine='python')
    # test_o = pd.read_csv(test_path)
    # train_o,train_new_o = reshape_train(train_o)
    #
    # # train_new_o.SaleDate = train_new_o.SaleDate.map(lambda x: timeHandle(x))
    # # train_new_o.SaleDate = pd.to_datetime(train_new_o.SaleDate)
    #
    # # 特征1：提取固定特征
    # print train_o['SaleDate']
    # train, train_new, test = get_origin_feats(train_o, train_new_o, test_o)
    #
    # test_1, test_2, test_3, test_4, test_5 ,test = test_split(train_new, test, week_5)
    #
    # # 特征2： 提取滚动特征
    # train_test = merge_train_test(train_new, test_1)
    # train_test = get_roll_feats(train_test)
    #
    # train_feat = train_test[train_test['SaleDate'] >= '2015-01-01']
    # train_feat = train_feat[train_feat['SaleDate'] <= '2015-04-30']   #训练集为1-4月份
    #
    # test_feat = train_test[train_test['SaleDate'].isin(week_5[0])]
    # test_feat.loc[:,'saleCount'] = 0
    # train_test = train_test[~train_test['SaleDate'].isin(test_1['SaleDate'])]
    #
    # train_feat.fillna(0,inplace=True)
    # test_feat.fillna(0,inplace=True)
    #
    #
    # feature_names = list(train_feat.columns)
    # do_not_use_for_training = ['SaleDate','saleCount','dayOfYear','price_mean','price_median','dayOn6DayDiff','dayOn5DayDiff','dayOn7DayDiff','dayOn8DayDiff','dayOn21DayDiff',
    #                            ]
    # predictors = [f for f in feature_names if f not in do_not_use_for_training]
    #
    # params = {'min_child_weight': 100, 'eta': 0.09, 'colsample_bytree': 0.3, 'max_depth': 8,
    #             'subsample': 0.6, 'lambda': 1., 'nthread': 4, 'booster' : 'gbtree', 'silent': 1,
    #             'eval_metric': 'rmse', 'objective': 'reg:linear'}
    # xgbtrain = xgb.DMatrix(train_feat[predictors], train_feat['saleCount'])
    # xgbvalid = xgb.DMatrix(test_feat[predictors])
    # model = xgb.train(params, xgbtrain, num_boost_round=100)
    # param_score = pd.Series(model.get_fscore()).sort_values(ascending=False)
    # print "Parameter score: "
    # print param_score
    # test_feat.loc[:,'saleCount'] = model.predict(xgbvalid)
    # test_feat['saleCount'] = test_feat['saleCount'].astype('int')
    # result_1 = test_feat[['Class','SaleDate','saleCount']]
    # result_1.fillna(0,inplace=True)
    # test_feat.fillna(0,inplace=True)
    # result_1 = pd.merge(test_1[['Class','SaleDate']], result_1, on=['Class','SaleDate'], how='left')
    # result_1.to_csv('result_1.csv',index=False)
    #
    #
    #
    #
    # # 第二轮
    # rolling_features = ['hotPast1MonthIndex','hotPast1WeekIndex', 'hotPast2WeekIndex','parHotPast1MonthIndex' ,'parHotPast1WeekIndex' ,'parHotPast2WeekIndex',
    #                     'day3OoverWeek3TotRatio', 'parWeekDayRatio', 'parWeekOn1WeekRatio','parWeekOn2WeekRatio' ,'weekDayRatio' ,'weekOn1WeekRatio' ,'weekOn2WeekRatio',
    #                     'dayOn14DayDiff','dayOn1DayDiff','dayOn2DayDiff','dayOn3DayDiff','dayOn4DayDiff','dayOn5DayDiff','dayOn6DayDiff','dayOn7DayDiff'
    #                     ]
    # feats = [f for f in train_test.columns if f not in rolling_features]
    # train_test = train_test[feats]
    # test_feat = test_feat[feats]
    # train_test = pd.concat([train_test,test_feat],axis=0)
    # train_test = merge_train_test(train_test, test_2)
    #
    # # 特征2： 提取滚动特征
    # train_test = get_roll_feats(train_test)
    #
    # train_feat = train_test[train_test['SaleDate'] >= '2015-01-01']    #使用二月份以后的数据
    #
    # test_feat = train_test[train_test['SaleDate'].isin(week_5[1])]
    # test_feat.loc[:,'saleCount'] = 0
    # train_test = train_test[~train_test['SaleDate'].isin(test_2['SaleDate'])]
    #
    #
    # train_feat.fillna(0,inplace=True)
    # test_feat.fillna(0,inplace=True)
    #
    # feature_names = list(train_feat.columns)
    # do_not_use_for_training = ['SaleDate','saleCount','dayOfYear','price_mean','price_median','dayOn6DayDiff','dayOn5DayDiff','dayOn7DayDiff','dayOn8DayDiff','dayOn21DayDiff',
    #                            ]
    # predictors = [f for f in feature_names if f not in do_not_use_for_training]
    #
    # params = {'min_child_weight': 100, 'eta': 0.09, 'colsample_bytree': 0.3, 'max_depth': 8,
    #             'subsample': 0.6, 'lambda': 1., 'nthread': 4, 'booster' : 'gbtree', 'silent': 1,
    #             'eval_metric': 'rmse', 'objective': 'reg:linear'}
    # xgbtrain = xgb.DMatrix(train_feat[predictors], train_feat['saleCount'])
    # xgbvalid = xgb.DMatrix(test_feat[predictors])
    # model = xgb.train(params, xgbtrain, num_boost_round=100)
    # param_score = pd.Series(model.get_fscore()).sort_values(ascending=False)
    # print "Parameter score: "
    # print param_score
    # test_feat['saleCount'] = model.predict(xgbvalid)
    # test_feat['saleCount'] = test_feat['saleCount'].astype('int')
    # result_2 = test_feat[['Class','SaleDate','saleCount']]
    # result_2 = pd.merge(test_2[['Class','SaleDate']], result_2, on=['Class','SaleDate'], how='left')
    # # print result_2['Class'].value_counts()
    # result_2.fillna(0,inplace=True)
    # # result_2['saleCount'][result_2['saleCount'] < 0 ] = 0
    # result_2.to_csv('result_2.csv',index=False)
    #
    #
    # # 第三轮
    # rolling_features = ['hotPast1MonthIndex','hotPast1WeekIndex', 'hotPast2WeekIndex','parHotPast1MonthIndex' ,'parHotPast1WeekIndex' ,'parHotPast2WeekIndex',
    #                     'day3OoverWeek3TotRatio', 'parWeekDayRatio', 'parWeekOn1WeekRatio','parWeekOn2WeekRatio' ,'weekDayRatio' ,'weekOn1WeekRatio' ,'weekOn2WeekRatio',
    #                     'dayOn14DayDiff','dayOn1DayDiff','dayOn2DayDiff','dayOn3DayDiff','dayOn4DayDiff','dayOn5DayDiff','dayOn6DayDiff','dayOn7DayDiff'
    #                     ]
    # feats = [f for f in train_test.columns if f not in rolling_features]
    # train_test = train_test[feats]
    # test_feat = test_feat[feats]
    # train_test = pd.concat([train_test,test_feat],axis=0)
    # train_test = merge_train_test(train_test, test_3)
    #
    # # 特征2： 提取滚动特征
    # train_test = get_roll_feats(train_test)
    #
    # train_feat = train_test[train_test['SaleDate'] >= '2015-01-01']    #使用二月份以后的数据
    #
    # test_feat = train_test[train_test['SaleDate'].isin(week_5[2])]
    # test_feat.loc[:,'saleCount'] = 0
    # train_test = train_test[~train_test['SaleDate'].isin(test_3['SaleDate'])]
    #
    #
    # train_feat.fillna(0,inplace=True)
    # test_feat.fillna(0,inplace=True)
    #
    # feature_names = list(train_feat.columns)
    # do_not_use_for_training = ['SaleDate','saleCount','dayOfYear','price_mean','price_median','dayOn6DayDiff','dayOn5DayDiff','dayOn7DayDiff','dayOn8DayDiff','dayOn21DayDiff',
    #                            ]
    # predictors = [f for f in feature_names if f not in do_not_use_for_training]
    #
    # params = {'min_child_weight': 100, 'eta': 0.09, 'colsample_bytree': 0.3, 'max_depth': 8,
    #             'subsample': 0.6, 'lambda': 1., 'nthread': 4, 'booster' : 'gbtree', 'silent': 1,
    #             'eval_metric': 'rmse', 'objective': 'reg:linear'}
    # xgbtrain = xgb.DMatrix(train_feat[predictors], train_feat['saleCount'])
    # xgbvalid = xgb.DMatrix(test_feat[predictors])
    # model = xgb.train(params, xgbtrain, num_boost_round=100)
    # param_score = pd.Series(model.get_fscore()).sort_values(ascending=False)
    # print "Parameter score: "
    # print param_score
    # test_feat['saleCount'] = model.predict(xgbvalid)
    # test_feat['saleCount'] = test_feat['saleCount'].astype('int')
    # result_3 = test_feat[['Class','SaleDate','saleCount']]
    # result_3 = pd.merge(test_3[['Class','SaleDate']], result_3, on=['Class','SaleDate'], how='left')
    # # print result_2['Class'].value_counts()
    # result_3.fillna(0,inplace=True)
    # # result_2['saleCount'][result_2['saleCount'] < 0 ] = 0
    # result_3.to_csv('result_3.csv',index=False)

    ### 提交训练结果
    # ### 提交训练结果
    # train_o = pd.read_csv(train_path,encoding='gbk',engine='python')
    # test_o = pd.read_csv(test_path)
    # train_o,train_new_o = reshape_train(train_o)
    #
    # # train_new_o.SaleDate = train_new_o.SaleDate.map(lambda x: timeHandle(x))
    # # train_new_o.SaleDate = pd.to_datetime(train_new_o.SaleDate)
    #
    # # 特征1：提取固定特征
    # print train_o['SaleDate']
    # train, train_new, test = get_origin_feats(train_o, train_new_o, test_o)
    # train_new.to_csv('train_new.csv',index=False)
    #
    # # test_1, test_2, test_3, test_4, test_5 ,test = test_split(train_new, test, week_5)
    # #
    # # # 特征2： 提取滚动特征
    # # train_test = merge_train_test(train_new, test_1)
    # # train_test = get_roll_feats(train_test)
    # #
    # # train_feat = train_test[train_test['SaleDate'] >= '2015-01-01']
    # # train_feat = train_feat[train_feat['SaleDate'] <= '2015-04-30']   #训练集为1-4月份
    # #
    # # test_feat = train_test[train_test['SaleDate'].isin(week_5[0])]
    # # test_feat.loc[:,'saleCount'] = 0
    # # train_test = train_test[~train_test['SaleDate'].isin(test_1['SaleDate'])]
    # #
    # # train_feat.fillna(0,inplace=True)
    # # test_feat.fillna(0,inplace=True)
    # #
    # #
    # # feature_names = list(train_feat.columns)
    # # do_not_use_for_training = ['SaleDate','saleCount','dayOfYear','price_mean','price_median','dayOn6DayDiff','dayOn5DayDiff','dayOn7DayDiff','dayOn8DayDiff','dayOn21DayDiff',
    # #                            ]
    # # predictors = [f for f in feature_names if f not in do_not_use_for_training]
    # #
    # # params = {'min_child_weight': 100, 'eta': 0.09, 'colsample_bytree': 0.3, 'max_depth': 8,
    # #             'subsample': 0.6, 'lambda': 1., 'nthread': 4, 'booster' : 'gbtree', 'silent': 1,
    # #             'eval_metric': 'rmse', 'objective': 'reg:linear'}
    # # xgbtrain = xgb.DMatrix(train_feat[predictors], train_feat['saleCount'])
    # # xgbvalid = xgb.DMatrix(test_feat[predictors])
    # # model = xgb.train(params, xgbtrain, num_boost_round=100)
    # # param_score = pd.Series(model.get_fscore()).sort_values(ascending=False)
    # # print "Parameter score: "
    # # print param_score
    # # test_feat.loc[:,'saleCount'] = model.predict(xgbvalid)
    # # test_feat['saleCount'] = test_feat['saleCount'].astype('int')
    # # result_1 = test_feat[['Class','SaleDate','saleCount']]
    # # result_1.fillna(0,inplace=True)
    # # test_feat.fillna(0,inplace=True)
    # # result_1 = pd.merge(test_1[['Class','SaleDate']], result_1, on=['Class','SaleDate'], how='left')
    # # result_1.to_csv('result_1.csv',index=False)
    # #
    # #
    # # # 第二轮
    # # rolling_features = ['hotPast1MonthIndex','hotPast1WeekIndex', 'hotPast2WeekIndex','parHotPast1MonthIndex' ,'parHotPast1WeekIndex' ,'parHotPast2WeekIndex',
    # #                     'day3OoverWeek3TotRatio', 'parWeekDayRatio', 'parWeekOn1WeekRatio','parWeekOn2WeekRatio' ,'weekDayRatio' ,'weekOn1WeekRatio' ,'weekOn2WeekRatio',
    # #                     'dayOn14DayDiff','dayOn1DayDiff','dayOn2DayDiff','dayOn3DayDiff','dayOn4DayDiff','dayOn5DayDiff','dayOn6DayDiff','dayOn7DayDiff'
    # #                     ]
    # # feats = [f for f in train_test.columns if f not in rolling_features]
    # # train_test = train_test[feats]
    # # test_feat = test_feat[feats]
    # # train_test = pd.concat([train_test,test_feat],axis=0)
    # # train_test = merge_train_test(train_test, test_2)
    # #
    # # # 特征2： 提取滚动特征
    # # train_test = get_roll_feats(train_test)
    # #
    # # train_feat = train_test[train_test['SaleDate'] >= '2015-01-01']    #使用二月份以后的数据
    # #
    # # test_feat = train_test[train_test['SaleDate'].isin(week_5[1])]
    # # test_feat.loc[:,'saleCount'] = 0
    # # train_test = train_test[~train_test['SaleDate'].isin(test_2['SaleDate'])]
    # #
    # #
    # # train_feat.fillna(0,inplace=True)
    # # test_feat.fillna(0,inplace=True)
    # #
    # # feature_names = list(train_feat.columns)
    # # do_not_use_for_training = ['SaleDate','saleCount','dayOfYear','price_mean','price_median','dayOn6DayDiff','dayOn5DayDiff','dayOn7DayDiff','dayOn8DayDiff','dayOn21DayDiff',
    # #                            ]
    # # predictors = [f for f in feature_names if f not in do_not_use_for_training]
    # #
    # # params = {'min_child_weight': 100, 'eta': 0.09, 'colsample_bytree': 0.3, 'max_depth': 8,
    # #             'subsample': 0.6, 'lambda': 1., 'nthread': 4, 'booster' : 'gbtree', 'silent': 1,
    # #             'eval_metric': 'rmse', 'objective': 'reg:linear'}
    # # xgbtrain = xgb.DMatrix(train_feat[predictors], train_feat['saleCount'])
    # # xgbvalid = xgb.DMatrix(test_feat[predictors])
    # # model = xgb.train(params, xgbtrain, num_boost_round=100)
    # # param_score = pd.Series(model.get_fscore()).sort_values(ascending=False)
    # # print "Parameter score: "
    # # print param_score
    # # test_feat['saleCount'] = model.predict(xgbvalid)
    # # test_feat['saleCount'] = test_feat['saleCount'].astype('int')
    # # result_2 = test_feat[['Class','SaleDate','saleCount']]
    # # result_2 = pd.merge(test_2[['Class','SaleDate']], result_2, on=['Class','SaleDate'], how='left')
    # # # print result_2['Class'].value_counts()
    # # result_2.fillna(0,inplace=True)
    # # # result_2['saleCount'][result_2['saleCount'] < 0 ] = 0
    # # result_2.to_csv('result_2.csv',index=False)
    #
    # # # 第三轮
    # # rolling_features = ['hotPast1MonthIndex','hotPast1WeekIndex', 'hotPast2WeekIndex','parHotPast1MonthIndex' ,'parHotPast1WeekIndex' ,'parHotPast2WeekIndex',
    # #                     'day3OoverWeek3TotRatio', 'parWeekDayRatio', 'parWeekOn1WeekRatio','parWeekOn2WeekRatio' ,'weekDayRatio' ,'weekOn1WeekRatio' ,'weekOn2WeekRatio',
    # #                     'dayOn14DayDiff','dayOn1DayDiff','dayOn2DayDiff','dayOn3DayDiff','dayOn4DayDiff','dayOn5DayDiff','dayOn6DayDiff','dayOn7DayDiff'
    # #                     ]
    # # feats = [f for f in train_test.columns if f not in rolling_features]
    # # train_test = train_test[feats]
    # # test_feat_2 = test_feat_2[feats]
    # # train_test = pd.concat([train_test,test_feat_2],axis=0)
    # # train_test = merge_train_test(train_test, test_3)
    # #
    # # # 特征2： 提取滚动特征
    # # train_test = get_roll_feats(train_test)
    # #
    # # train_feat = train_test[train_test['SaleDate'] >= '2015-01-01']    #使用二月份以后的数据
    # #
    # # test_feat = train_test[train_test['SaleDate'] >= '2015-05-15']
    # # test_feat.loc[:,'saleCount'] = 0
    # # train_test = train_test[~train_test['SaleDate'].isin(test_3['SaleDate'])]
    # #
    # #
    # # train_feat.fillna(0,inplace=True)
    # # test_feat.fillna(0,inplace=True)
    # #
    # # test_feat_3 = test_feat[test_feat['SaleDate'].isin(week_5[2])]
    # # test_feat_3.fillna(0,inplace=True)
    # # test_feat_3['saleCount'] = 0
    # #
    # # feature_names = list(train_feat.columns)
    # # do_not_use_for_training = ['SaleDate','saleCount','dayOfYear','price_mean','price_median','dayOn6DayDiff','dayOn5DayDiff','dayOn7DayDiff','dayOn8DayDiff','dayOn21DayDiff',
    # #                            ]
    # # predictors = [f for f in feature_names if f not in do_not_use_for_training]
    # #
    # # params = {'min_child_weight': 100, 'eta': 0.09, 'colsample_bytree': 0.3, 'max_depth': 8,
    # #             'subsample': 0.6, 'lambda': 1., 'nthread': 4, 'booster' : 'gbtree', 'silent': 1,
    # #             'eval_metric': 'rmse', 'objective': 'reg:linear'}
    # # xgbtrain = xgb.DMatrix(train_feat[predictors], train_feat['saleCount'])
    # # xgbvalid = xgb.DMatrix(test_feat_3[predictors])
    # # model = xgb.train(params, xgbtrain, num_boost_round=100)
    # # param_score = pd.Series(model.get_fscore()).sort_values(ascending=False)
    # # print "Parameter score: "
    # # print param_score
    # # test_feat_3['saleCount'] = model.predict(xgbvalid)
    # # test_feat_3['saleCount'] = test_feat_3['saleCount'].astype('int')
    # # result_3 = test_feat_2[['Class','SaleDate','saleCount']]
    # # result_3 = pd.merge(test_3[['Class','SaleDate']], result_3, on=['Class','SaleDate'], how='left')
    # # # print result_2['Class'].value_counts()
    # # result_3.fillna(0,inplace=True)
    # # result_3['saleCount'][result_3['saleCount'] < 0 ] = 0
    # # result_3.to_csv('result_3.csv',index=False)
    # ### 提交训练结果
