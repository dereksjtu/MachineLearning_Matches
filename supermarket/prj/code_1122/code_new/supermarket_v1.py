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
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

train_path = '../input_new/train.csv'
test_path = '../input_new/test.csv'
hol_path = '../input_new/holiday.csv'
train_date_path = '../input_new/train_date.csv'
# train_coupon_path = '../input/train_new_coupon.csv'
# coupon_May_pre_path = '../input/train_coupon_reshape_May_pre.csv'
# coupon_Apri_pre_path = '../input/train_coupon_reshape_Apri_pre.csv'
# cache_path = '../input/cache/'
# output_path = '../output/'

# 评分函数
def score(y_test,y_pred):
    return 1.0 / (1.0 + np.sqrt(mean_squared_error(y_test, y_pred)))

# train = pd.read_csv(train_path,encoding='gbk',engine='python')
# test = pd.read_csv(test_path)

def train_test_split_1(train_o,train_new_o):
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


#### trianing for ligjhtGBM ####
def fit_gbm(x_train, x_test, y_train, y_test):

        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)

        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 8,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0,
            'lambda_l2': 1
        }

        def rmse_d(y, d):
            c = d.get_label()
            result = 1.0 / (1.0 + np.sqrt(mean_squared_error(y, c)))
            return "self define cost functinos-", -result, False

        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=90000,
                        feval=rmse_d,
                        valid_sets=lgb_eval,
                        early_stopping_rounds=1000)

        # gbm.save_model('./output/model_gbm.txt')

        print('Feature importances:', list(gbm.feature_importance()))

        return gbm

def fit_gbm_cv(x_train, x_test, y_train, y_test):
        import lightgbm as lgb
        import pandas as pd
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import GridSearchCV

        estimator = lgb.LGBMRegressor(objective='regression',
                                      colsample_bytree=0.8,
                                      subsample=0.9,
                                      subsample_freq=5,
                                      metric= 'rmse')

        # params = {
        #     'boosting_type': 'gbdt',
        #     'objective': 'regression',
        #     'metric': 'rmse',
        #     'num_leaves': 64,
        #     'learning_rate': 0.1,
        #     'feature_fraction': 0.8,
        #     'bagging_fraction': 0.8,
        #     'bagging_freq': 5,
        #     'verbose': 0,
        #     # 'lambda_l2': 1
        # }

        param_grid = {
            'learning_rate': [0.01,0.02,0.05,0.1,0.3],
            # 'n_estimators': [1000, 2000],
            'num_leaves': [8, 32, 64 , 128]
        }

        fit_param = {'categorical_feature': [0, 1]}
        gbm = GridSearchCV(estimator, param_grid, fit_params=fit_param, n_jobs=5, refit=True)
        gbm.fit(x_train, y_train)

        # gbm = lgb.train(params,
        #                 lgb_train,
        #                 num_boost_round=90000,
        #                 feval=rmse_d,
        #                 valid_sets=lgb_eval,
        #                 early_stopping_rounds=10000)

        print('.....................................cv results.......................')
        print(gbm.cv_results_)

if __name__ == "__main__":

    week_4 = [
                # ['2015-04-01','2015-04-02','2015-04-03','2015-04-04','2015-04-05','2015-04-06','2015-04-07',
                # '2015-04-08','2015-04-10','2015-04-11','2015-04-12','2015-04-13','2015-04-14',
                # '2015-04-15','2015-04-17','2015-04-18','2015-04-19','2015-04-20','2015-04-21',
                # '2015-04-22','2015-04-23','2015-04-24','2015-04-25','2015-04-26','2015-04-27','2015-04-28',
                # '2015-04-29','2015-04-30']

                # ['2015-04-01','2015-04-02','2015-04-03','2015-04-04','2015-04-05','2015-04-06','2015-04-07'],
                # ['2015-04-08','2015-04-10','2015-04-11','2015-04-12','2015-04-13','2015-04-14'],
                # ['2015-04-15','2015-04-17','2015-04-18','2015-04-19','2015-04-20','2015-04-21'],
                # ['2015-04-22','2015-04-23','2015-04-24','2015-04-25','2015-04-26','2015-04-27','2015-04-28'],
                # ['2015-04-29','2015-04-30']

                # ['2015-04-01','2015-04-02','2015-04-03','2015-04-04','2015-04-05','2015-04-06','2015-04-07',
                # '2015-04-08','2015-04-09','2015-04-10','2015-04-11','2015-04-12','2015-04-13','2015-04-14'],
                # ['2015-04-15','2015-04-16','2015-04-17','2015-04-18','2015-04-19','2015-04-20','2015-04-21',
                # '2015-04-22','2015-04-23','2015-04-24','2015-04-25','2015-04-26','2015-04-27','2015-04-28'],
                # ['2015-04-29','2015-04-30']

                # #正常整4月预测
                ['2015-04-01'],['2015-04-02'],['2015-04-03'],['2015-04-04'],['2015-04-05'],['2015-04-06'],['2015-04-07'],
                ['2015-04-08'],['2015-04-10'],['2015-04-11'],['2015-04-12'],['2015-04-13'],['2015-04-14'],
                ['2015-04-15'],['2015-04-17'],['2015-04-18'],['2015-04-19'],['2015-04-20'],['2015-04-21'],
                ['2015-04-22'],['2015-04-23'],['2015-04-24'],['2015-04-25'],['2015-04-26'],['2015-04-27'],['2015-04-28'],
                ['2015-04-29'],['2015-04-30']

                #排除节前3天，即预测25天
                # ['2015-04-04'],['2015-04-05'],['2015-04-06'],['2015-04-07'],
                # ['2015-04-08'],['2015-04-10'],['2015-04-11'],['2015-04-12'],['2015-04-13'],['2015-04-14'],
                # ['2015-04-15'],['2015-04-17'],['2015-04-18'],['2015-04-19'],['2015-04-20'],['2015-04-21'],
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
                #正常整4月预测
                '2015-04-01','2015-04-02','2015-04-03','2015-04-04','2015-04-05','2015-04-06','2015-04-07',
                '2015-04-08','2015-04-09','2015-04-10','2015-04-11','2015-04-12','2015-04-13','2015-04-14',
                '2015-04-15','2015-04-16','2015-04-17','2015-04-18','2015-04-19','2015-04-20','2015-04-21',
                '2015-04-22','2015-04-23','2015-04-24','2015-04-25','2015-04-26','2015-04-27','2015-04-28',
                '2015-04-29','2015-04-30'

                #排除节前3天，即预测25天
                # '2015-04-04','2015-04-05','2015-04-06','2015-04-07',
                # '2015-04-08','2015-04-09','2015-04-10','2015-04-11','2015-04-12','2015-04-13','2015-04-14',
                # '2015-04-15','2015-04-16','2015-04-17','2015-04-18','2015-04-19','2015-04-20','2015-04-21',
                # '2015-04-22','2015-04-23','2015-04-24','2015-04-25','2015-04-26','2015-04-27','2015-04-28',
                # '2015-04-29','2015-04-30'

                #单独预测假日3天
                # '2015-04-04','2015-04-05','2015-04-06'
            ]
    week_5 = [
                # ['2015-05-01','2015-05-02','2015-05-03','2015-05-04','2015-05-05','2015-05-06','2015-05-07'],
                # ['2015-05-08','2015-05-09','2015-05-10','2015-05-11','2015-05-12','2015-05-13','2015-05-14'],
                # ['2015-05-15','2015-05-16','2015-05-17','2015-05-18','2015-05-19','2015-05-20','2015-05-21'],
                # ['2015-05-22','2015-05-23','2015-05-24','2015-05-25','2015-05-26','2015-05-27','2015-05-28'],
                # ['2015-05-29','2015-05-30']

                ['2015-05-01'],['2015-05-02'],['2015-05-03'],['2015-05-04'],['2015-05-05'],['2015-05-06'],['2015-05-07'],
                ['2015-05-08'],['2015-05-09'],['2015-05-10'],['2015-05-11'],['2015-05-12'],['2015-05-13'],['2015-05-14'],
                ['2015-05-15'],['2015-05-16'],['2015-05-17'],['2015-05-18'],['2015-05-19'],['2015-05-20'],['2015-05-21'],
                ['2015-05-22'],['2015-05-23'],['2015-05-24'],['2015-05-25'],['2015-05-26'],['2015-05-27'],['2015-05-28'],
                ['2015-05-29'],['2015-05-30']
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

    coupon_class = [15,
20,
21,
22,
23,
30,
31,
32,
33,
34,
1501,
1502,
1503,
1505,
1508,
1510,
1512,
1513,
1515,
1517,
1518,
1519,
1521,
2001,
2002,
2003,
2005,
2006,
2007,
2008,
2009,
2010,
2011,
2012,
2013,
2014,
2015,
2101,
2103,
2104,
2105,
2106,
2107,
2201,
2202,
2203,
2204,
2205,
2206,
2207,
2208,
2209,
2210,
2211,
2212,
2301,
2302,
2303,
2304,
2305,
2306,
2307,
2309,
2310,
2311,
2314,
2317,
3001,
3002,
3003,
3004,
3006,
3007,
3008,
3010,
3011,
3013,
3016,
3018,
3107,
3109,
3110,
3112,
3113,
3114,
3116,
3117,
3118,
3119,
3126,
3319,
3320,
3402,
3403,
3407,
3408,
3415,
3423,
3424,
3426,
3431]

    # t0 = time.time()
    # ## 提取公共类
    # train_new = pd.read_csv('train_new.csv')
    # test = pd.read_csv('../input/test.csv')
    # l_inter = np.intersect1d(train_new['Class'].unique(),test['Class'].unique())
    # print "inter class: ",len(l_inter)
    # del train_new,test
    # ## 提取公共类
    #
    #
    # train_o = pd.read_csv(train_path,encoding='gbk',engine='python')
    # test_o = pd.read_csv(test_path)
    # train_o,train_new_o = reshape_train(train_o)
    # # 验证集中不预测的类
    # # train_o, train_new_o = exclude_class(train_new_o, train_o, coupon_class)
    #
    # #取公共类
    # train_new_o = train_new_o[train_new_o['Class'].isin(l_inter)]
    # train_o = train_o[train_o['Class'].isin(l_inter)]
    # #取公共类
    #
    #
    # # 验证 train为2,3月份， test为4月份数据
    # train ,train_new, test = train_test_split_1(train_o,train_new_o)
    # test.loc[:,'saleCount'] = 0
    #
    # # 特征1： 提取固定特征
    # train_new = exclude_abnormal_value(train_new)
    # # train_new = exclude_abnormal_value_coupon(train_new)
    # print 'Filter abnormal value.'
    # train_new_o.SaleDate = train_new_o.SaleDate.map(lambda x: timeHandle(x))
    # train_new_o.SaleDate = pd.to_datetime(train_new_o.SaleDate)
    # train, train_new, test = get_origin_feats(train, train_new, test)
    #
    # #取公共类
    # test = test[test['Class'].isin(l_inter)]
    # #取公共类
    #
    # print len(train_new['Class'].unique())
    # print len(test['Class'].unique())
    # # 提取四月前的Coupon特征
    # del train_new['Coupon']
    # train_coupon = pd.read_csv(train_coupon_path)
    # train_coupon['SaleDate'] = pd.to_datetime(train_coupon['SaleDate'])
    # train_new = pd.merge(train_new,train_coupon[['Class','SaleDate','Coupon']],on=['Class','SaleDate'],how='left')
    # train_new['Coupon'].fillna(0,inplace=True)
    # print train_new['Coupon'].unique()
    # # 提取四月前的Coupon特征
    # #将预测的4月的Coupon特征合并到4月验证集中
    # del test['Coupon']
    # test_coupon = pd.read_csv(coupon_Apri_pre_path)
    # test_coupon['SaleDate'] = pd.to_datetime(test_coupon['SaleDate'])
    # test = pd.merge(test,test_coupon[['Class','SaleDate','Coupon']],on=['Class','SaleDate'],how='left')
    # test['Coupon'].fillna(0,inplace=True)
    # print test['Coupon'].unique()
    # #将预测的4月的Coupon特征合并到4月验证集中
    #
    #
    #
    # #训练日期
    # start_date = '2015-03-01'
    # # start_date = '2015-01-01'
    # end_date = '2015-04-01'
    # # train_hol_dates = ['2015-01-01','2015-01-02','2015-01-03','2015-02-18','2015-02-19','2015-02-20','2015-02-21','2015-02-22','2015-02-23','2015-02-24']
    # # train_hol_dates = ['2015-01-01','2015-01-02','2015-01-03','2015-02-18','2015-02-19','2015-02-20']
    # # train_hol_dates = ['2015-01-01','2015-01-02','2015-01-03','2015-02-16','2015-02-17','2015-02-18','2015-02-19','2015-02-20','2015-02-21','2015-02-22','2015-02-23','2015-02-24','2015-04-29','2015-04-30']
    #
    # # # 分离测试集
    # test_1 = test[test['SaleDate'].isin(week_4[0])]
    # # 分离验证集
    # # test_valid_1,test_valid_2,test_valid_3,test_valid_4,test_valid_5, test_valid = valid_split(train_new_o, week_4, month_4)
    # test_valid = valid_split(train_new_o, week_4, month_4)
    #
    # #取公共类
    # test_valid_o = test_valid[test_valid['Class'].isin(l_inter)]
    # #取公共类
    #
    # #验证集取4月1号到4月28号
    # # 特征2： 提取滚动特征
    # train_test = merge_train_test(train_new, test_1)
    # # train_test,l_roll_feats = get_roll_feats(train_test)
    # l_roll_feats = []
    #
    # # train_test = pd.get_dummies(train_test,columns=['month','Class','parClass','day'])
    #
    #
    # train_feat = train_new[train_new['SaleDate'] >= start_date]    #使用二月份以后的数据
    # train_feat = train_feat[train_feat['SaleDate'] < end_date]   #训练集为2-3月份
    # #单独用假日预测
    # # train_feat_1 = train_test[train_test['SaleDate'].isin(train_hol_dates)]   #训练集为2-3月份
    #
    # #排除过年的异常值
    # # exclude_date = ['2015-02-17','2015-02-18']
    # # train_feat_1 = train_feat_1[~train_feat_1['SaleDate'].isin(exclude_date)]   #训练集为2-3月份
    #
    # test_feat = test[test['SaleDate'] >= end_date]
    # test_feat = test_feat[test_feat['SaleDate'] <= '2015-04-30'] #验证集为四月份
    # test_feat.loc[:,'saleCount'] = 0
    #
    #
    #
    # train_feat.fillna(0,inplace=True)
    # test_feat.fillna(0,inplace=True)
    #
    #
    # # test_feat_1 = test_feat[test_feat['SaleDate'].isin(week_4[0])]
    # # test_feat_1.fillna(0,inplace=True)
    # # test_feat_1.loc[:,'saleCount'] = 0
    # feature_names = list(train_feat.columns)
    # do_not_use_for_training = ['SaleDate','saleCount',
    #                            'Coupon',
    #                            'dayOfYear','price_mean','price_median',
    #                            # 'parCumtype','parClass',
    #                            # 'parHotPast1MonthIndex',
    #                            # 'dayOn21DayDiff',
    #                            'lastWeekSaleCount_mean',
    #                            # 'expweighted_14_avg',
    #                            'trend_7',
    #                            'trend_14',
    #                            # 'expweighted_7_avg',
    #                            'classWeekdayRatio',
    #                            # 'parClassWeekdayRatio',
    #                            'moving_30_avg',
    #                            'expweighted_30_avg',
    #                            'moving_21_avg',
    #                            'expweighted_21_avg',
    #                            'moving_14_avg',
    #                            'expweighted_14_avg',
    #                            'moving_7_avg',
    #                            # 'expweighted_7_avg',
    #                            # 'disholDaySaleCount_mean',
    #                            'disholDaySaleCount_max',
    #                            'last2WeekSaleCount_max','last3WeekSaleCount_max',
    #                            'Class',
    #                            'parClass',
    #                            'last3wTot','last4wTot','last2wTot','last1wTot',
    #                            'last21d','last28d','last7d','last14d',
    #                            'last7d_mean','last21d_mean','last14d_mean','last30d_mean',
    #                            # 'last1d',
    #                            'last1wMean','last2wMean','last3wMean','last4wMean',
    #                            # 'diff_2',
    #                            # 'weekly_fb',
    #                            'saleCount_fb',
    #                            'trend_fb',
    #                            'hotIndex',
    #                            # 'wkHolRatio',
    #                            # 'diswkDaySaleCount_mean',
    #                            # 'holDaySaleCount_min',
    #                            # 'holDaySaleCount_min',
    #                            # 'holDaySaleCount_median',
    #                            # 'holDaySaleCount_mean','month','parHotIndex','dayOn5DayDiff'
    #                            # 'last2d','last4d','last',
    #                            'cumType','parClassWeekdayRatio','parHotIndex','parCumType',
    #                            'classWeekdayRatio_std','classWeekdayRatio_median','classWeekdayRatio_max',
    #                            'parclassWeekdayRatio_std','parclassWeekdayRatio_median','parclassWeekdayRatio_max',#反效果
    #                            'diswkDaySaleCount_std','diswkDaySaleCount_median','diswkDaySaleCount_max',
    #                            'wkDaySaleCount_std','wkDaySaleCount_median','wkDaySaleCount_max',
    #                            'holDaySaleCount_std','holDaySaleCount_median','holDaySaleCount_max',
    #                            'disholDaySaleCount_std','disholDaySaleCount_median','disholDaySaleCount_max',#反效果
    #                            'holDaySaleCount_std','holDaySaleCount_median','holDaySaleCount_max',
    #                            'parclassWeekdayRatio_mean',
    #                            ]
    #
    #
    # predictors = [f for f in feature_names if f not in do_not_use_for_training]
    #
    # params = {'min_child_weight': 10, 'eta': 0.02, 'colsample_bytree': 0.3, 'max_depth': 6,
    #             'subsample': 0.8, 'lambda': 2, 'nthread': 4, 'booster' : 'gbtree', 'silent': 1,
    #             'eval_metric': 'rmse', 'objective': 'reg:linear'}
    # boostRound = 1000
    #
    # # print train_feat_1[predictors]
    #
    # # ---------------------------  lgb ---------------------------------
    # # del test_feat['saleCount']
    # # X = train_feat[predictors]
    # # y = train_feat['saleCount']
    # # x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)
    # # lgb_model = fit_gbm(x_train, x_test, y_train, y_test)
    # # test_feat.loc[:,'saleCount'] = lgb_model.predict(test_feat[predictors])
    # # result = test_feat[['Class','SaleDate','saleCount']]
    # # # test_valid_1 = test_valid[test_valid['SaleDate'].isin(week_4[0])]
    # # # test_valid_1.fillna(0,inplace=True)
    # # test_valid = test_valid_o.copy()
    # # result = pd.merge(test_valid[['Class','SaleDate']], result, on=['Class','SaleDate'], how='left')
    # # result['saleCount'][result['saleCount'] < 0] = 0
    # # result.fillna(0,inplace=True)
    # # score_1 = score(result['saleCount'],test_valid_o['saleCount'])
    # # print "the 1th day predictive score:{}".format(score_1)
    # # ---------------------------  lgb ---------------------------------
    #
    # # ---------------------------  xgbs ---------------------------------
    # xgbtrain = xgb.DMatrix(train_feat[predictors], train_feat['saleCount'])
    # xgbvalid = xgb.DMatrix(test_feat[predictors])
    # model = xgb.train(params, xgbtrain, num_boost_round=boostRound)
    # param_score = pd.Series(model.get_fscore()).sort_values(ascending=False)
    # print "Parameter score: "
    # param_score.to_csv('param_score.csv')
    # print param_score, len(predictors)
    # test_feat.loc[:,'saleCount'] = model.predict(xgbvalid)
    # result = test_feat[['Class','SaleDate','saleCount']]
    # # test_valid_1 = test_valid[test_valid['SaleDate'].isin(week_4[0])]
    # # test_valid_1.fillna(0,inplace=True)
    # test_valid = test_valid_o.copy()
    # result = pd.merge(test_valid[['Class','SaleDate']], result, on=['Class','SaleDate'], how='left')
    # result['saleCount'][result['saleCount'] < 0] = 0
    # result.fillna(0,inplace=True)
    # score_1 = score(result['saleCount'],test_valid_o['saleCount'])
    # print "the 1th day predictive score:{}".format(score_1)
    # # ---------------------------  xgb ---------------------------------


    # 第二轮
    # 特征2： 提取滚动特征
    # 4月2号 - 4月30号

    #四月整月预测28天
    #排除节前3天，即预测25天
    # for i in range(1,25):
    # 单独用假日预测 4.4,4.5,4.6
    # for i in range(1,5):
#     for i in range(1,28):
#         # l_roll_feats = []
#         train_test = train_test[train_test['SaleDate'] < end_date]
#         # 单独用假日预测
#         # train_test = train_test[train_test['SaleDate'].isin(train_hol_dates)]
#         train_test = merge_train_test(train_test, test_feat_1)
#         feats = [f for f in train_test.columns if f not in l_roll_feats]
#         train_test = train_test[feats]
#
#         test_i = test[test['SaleDate'].isin(week_4[i])]
#         train_test = merge_train_test(train_test, test_i)
#         # train_test,l_roll_feats = get_roll_feats(train_test)
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
#         # feature_names = list(train_feat_1.columns)
#         # predictors = [f for f in feature_names if f not in do_not_use_for_training]
#
#         xgbtrain = xgb.DMatrix(train_feat_1[predictors], train_feat_1['saleCount'])
#         xgbvalid = xgb.DMatrix(test_feat_1[predictors])
#         if i % 7 == 0:
#             model = xgb.train(params, xgbtrain, num_boost_round=boostRound)
#
#         test_feat_1.loc[:,'saleCount'] = model.predict(xgbvalid)
#         result_i = test_feat_1[['Class','SaleDate','saleCount']]
#         # result_i['saleCount'] = 1.2 *  result_i['saleCount']
#         print len(result_i['saleCount'])
#         test_valid_i = test_valid[test_valid['SaleDate'].isin(week_4[i])]
#         test_valid_i.fillna(0,inplace=True)
#         result_i = pd.merge(test_valid_i[['Class','SaleDate']], result_i, on=['Class','SaleDate'], how='left')
#         print week_4[i]
#         result['saleCount'][result['saleCount'] < 0] = 0
#         result_i.fillna(0,inplace=True)
#         # result['saleCount'][result['saleCount'] < 0] = 1
#         # result_i.fillna(1,inplace=True)
#         score_i = score(test_valid_i['saleCount'],result_i['saleCount'])
#         result = pd.concat([result, result_i], axis=0)
#         print "the {}th day predictive score:{}".format(i + 1,score_i)
#
# result = pd.merge(test_valid[['Class','SaleDate']], result, on=['Class','SaleDate'], how='left')
# result['saleCount'][result['saleCount'] < 0] = 0
# result['saleCount'].fillna(0,inplace=True)
# result['saleCount'] = result['saleCount'].astype('int')
# result.to_csv('result_valid_coupon.csv',index=False)
# test_valid.to_csv('test_valid_valid_coupon.csv',index=False)
# score_f = score(test_valid['saleCount'],result['saleCount'])

# test_valid_valid_coupon = pd.read_csv('test_valid_valid_coupon.csv')
# test_valid_valid_nocoupon = pd.read_csv('test_valid_valid_nocoupon.csv')
# result_valid_coupon = pd.read_csv('result_valid_coupon.csv')
# result_valid_nocoupon = pd.read_csv('result_valid_nocoupon.csv')
# test_valid = pd.concat([test_valid_valid_coupon,test_valid_valid_nocoupon],axis=0)
# result = pd.concat([result_valid_coupon,result_valid_nocoupon],axis=0)

# score_f = score(test_valid['saleCount'],result['saleCount'])
# print "Elapse time is {} minutes".format((time.time() - t0) / (1.0 * 60))
# print "Total predictive score:{}".format(score_f)


# # for further analyse
# l_index = []
# l_score = []
# for i in test_valid['Class'].unique():
#     score_1 = score(test_valid['saleCount'][test_valid['Class'] == i] ,result['saleCount'][result['Class'] == i])
#     l_index.append(i)
#     l_score.append(score_1)
# ser_score = pd.Series(l_score, index=l_index)
# ser_score.to_csv('ser_score.csv')


#  全部预测
        ########################## 提交训练结果 #####################################

    t0 = time.time()
    # do_not_use_class = [1507,3208,3311,3413]
    train_o = pd.read_csv(train_path,encoding='gbk',engine='python')
    test_o = pd.read_csv(test_path)
    train_o,train_new_o = reshape_train(train_o)
    # test_o, train_new_o = exclude_class(train_new_o, test_o, coupon_class)

    # train_new_o.SaleDate = train_new_o.SaleDate.map(lambda x: timeHandle(x))
    # train_new_o.SaleDate = pd.to_datetime(train_new_o.SaleDate)
    # test_o.SaleDate = test_o.SaleDate.map(lambda x: timeHandle(x))
    # test_o.SaleDate = pd.to_datetime(train_new_o.SaleDate)

    test_o.loc[:,'saleCount'] = 0

    train_start = '2015-08-01'
    train_end = '2015-09-01'
    # train_start = '2015-04-01'
    # train_end = '2015-05-01'
    # train_end = '2015-05-01'

    # 特征1： 提取固定特征
    train, train_new, test = get_origin_feats(train_o, train_new_o, test_o)
    train_new = exclude_abnormal_value(train_new)
    # train_new = exclude_abnormal_value_coupon(train_new)
    print 'Filter abnormal value.'

    # Coupon 有毒，暂时不用
    # # 提取五月前的Coupon特征
    # del train_new['Coupon']
    # train_coupon = pd.read_csv(train_coupon_path)
    # train_coupon['SaleDate'] = pd.to_datetime(train_coupon['SaleDate'])
    # train_new = pd.merge(train_new,train_coupon[['Class','SaleDate','Coupon']],on=['Class','SaleDate'],how='left')
    # train_new['Coupon'].fillna(0,inplace=True)
    # print train_new['Coupon'].unique()
    # # 提取五月前的Coupon特征
    # #将预测的5月预测的Coupon特征合并到5月测试集中
    # # del test['Coupon']
    # test_coupon = pd.read_csv(coupon_May_pre_path)
    # test_coupon['SaleDate'] = pd.to_datetime(test_coupon['SaleDate'])
    # test = pd.merge(test,test_coupon[['Class','SaleDate','Coupon']],on=['Class','SaleDate'],how='left')
    # test['Coupon'].fillna(0,inplace=True)
    # print test['Coupon'].unique()
    # #将预测的5月预测的Coupon特征合并到5月测试集中


    # test_5 = test[test['SaleDate'].isin(week_5[0])]
    # 特征2： 提取滚动特征
    # train_test = merge_train_test(train_new, test_5)
    # train_test,l_roll_feats = get_roll_feats(train_test)

    train_feat = train_new[train_new['SaleDate'] >= train_start]    #使用二月份以后的数据
    train_feat = train_feat[train_feat['SaleDate'] < train_end]   #训练集为2-3月份

    test_feat = test[test['SaleDate'] >= '2015-09-01']
    test_feat = test_feat[test_feat['SaleDate'] <= '2015-10-29'] #测试集
    # test_valid = test_valid[test_valid['SaleDate'] < '2015-04-27']
    test_feat.loc[:,'saleCount'] = 0


    train_feat.fillna(0,inplace=True)
    test_feat.fillna(0,inplace=True)

    # test_feat_1 = test_feat[test_feat['SaleDate'].isin(week_5[0])]
    # test_feat_1.fillna(0,inplace=True)
    # test_feat_1.loc[:,'saleCount'] = 0
    feature_names = list(train_feat.columns)
    do_not_use_for_training = ['SaleDate','saleCount',
                               'Coupon',
                               'dayOfYear','price_mean','price_median',
                               # 'parCumtype','parClass',
                               # 'parHotPast1MonthIndex',
                               # 'dayOn21DayDiff',
                               'lastWeekSaleCount_mean',
                               # 'expweighted_14_avg',
                               'trend_7',
                               'trend_14',
                               # 'expweighted_7_avg',
                               'classWeekdayRatio',
                               # 'parClassWeekdayRatio',
                               'moving_30_avg',
                               'expweighted_30_avg',
                               'moving_21_avg',
                               'expweighted_21_avg',
                               'moving_14_avg',
                               'expweighted_14_avg',
                               'moving_7_avg',
                               # 'expweighted_7_avg',
                               # 'disholDaySaleCount_mean',
                               'disholDaySaleCount_max',
                               'last2WeekSaleCount_max','last3WeekSaleCount_max',
                               'Class',
                               'parClass',
                               'last3wTot','last4wTot','last2wTot','last1wTot',
                               'last21d','last28d','last7d','last14d',
                               'last7d_mean','last21d_mean','last14d_mean','last30d_mean',
                               # 'last1d',
                               'last1wMean','last2wMean','last3wMean','last4wMean',
                               # 'diff_2',
                               # 'weekly_fb',
                               'saleCount_fb',
                               'trend_fb',
                               'hotIndex',#反效果
                               # 'wkHolRatio',
                               # 'diswkDaySaleCount_mean',
                               # 'holDaySaleCount_min',
                               # 'holDaySaleCount_min',
                               # 'holDaySaleCount_median',
                               # 'holDaySaleCount_mean','month','parHotIndex','dayOn5DayDiff'
                               # 'last2d','last4d','last',
                               'cumType','parClassWeekdayRatio','parHotIndex','parCumType',
                               'classWeekdayRatio_std','classWeekdayRatio_median','classWeekdayRatio_max',
                               'parclassWeekdayRatio_std','parclassWeekdayRatio_median','parclassWeekdayRatio_max',#反效果
                               'diswkDaySaleCount_std','diswkDaySaleCount_median','diswkDaySaleCount_max',
                               'wkDaySaleCount_std','wkDaySaleCount_median','wkDaySaleCount_max',
                               'holDaySaleCount_std','holDaySaleCount_median','holDaySaleCount_max',
                               'disholDaySaleCount_std','disholDaySaleCount_median','disholDaySaleCount_max',#反效果
                               'holDaySaleCount_std','holDaySaleCount_median','holDaySaleCount_max',
                               'parclassWeekdayRatio_mean',
                               ]


    predictors = [f for f in feature_names if f not in do_not_use_for_training]

    params = {'min_child_weight': 10, 'eta': 0.02, 'colsample_bytree': 0.3, 'max_depth': 6,
                'subsample': 0.8, 'lambda': 2, 'nthread': 4, 'booster' : 'gbtree', 'silent': 1,
                'eval_metric': 'rmse', 'objective': 'reg:linear'}
    boostRound = 1000
    # params = {'min_child_weight': 10, 'eta': 0.02, 'colsample_bytree': 0.3, 'max_depth': 6,
    #             'subsample': 0.8, 'lambda': 2, 'nthread': 4, 'booster' : 'gbtree', 'silent': 1,
    #             'eval_metric': 'rmse', 'objective': 'reg:linear'}
    # boostRound = 1000


    # predictors = [f for f in feature_names if f not in do_not_use_for_training]
    #
    # params = {'min_child_weight': 100, 'eta': 0.02, 'colsample_bytree': 0.3, 'max_depth': 7,
    #             'subsample': 0.8, 'lambda': 2, 'nthread': 4, 'booster' : 'gbtree', 'silent': 1,
    #             'eval_metric': 'rmse', 'objective': 'reg:linear'}
    # boostRound = 1000
    # # ---------------------------  lgb ---------------------------------
    del test_feat['saleCount']
    X = train_feat[predictors]
    y = train_feat['saleCount']
    print predictors
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)
    lgb_model = fit_gbm(x_train, x_test, y_train, y_test)
    test_feat.loc[:,'saleCount'] = lgb_model.predict(test_feat[predictors])
    result = test_feat[['Class','SaleDate','saleCount']]
    # test_valid_1 = test_valid[test_valid['SaleDate'].isin(week_4[0])]
    # test_valid_1.fillna(0,inplace=True)
    # test_valid = test_valid_o.copy()
    # result = pd.merge(test[['Class','SaleDate']], result, on=['Class','SaleDate'], how='left')
    # result['saleCount'][result['saleCount'] < 0] = 0
    # result.fillna(0,inplace=True)
    result = pd.merge(test[['Class','SaleDate']], result, on=['Class','SaleDate'], how='left')
    result.fillna(0,inplace=True)
    result['saleCount'][result['saleCount'] < 0] = 0
    result['saleCount'] = result['saleCount'].astype('int')
    result['Class'] = result['Class'].astype('int')
    # test_o = test_o[test_o['Class'].isin(coupon_class)]
    result['SaleDate'] = test_o['SaleDate']
    # result.to_csv('result.csv',index=False)
    result.rename(columns={'Class':'bianma','saleCount':'xiaoliang','SaleDate':'riqi'},inplace=True)
    result.to_csv('lgb_ret_8_ful2.csv',index=False,encoding='gbk')
    # score_1 = score(result['saleCount'],test_valid_o['saleCount'])
    # print "the 1th day predictive score:{}".format(score_1)
    # # ---------------------------  lgb ---------------------------------

    # xgbtrain = xgb.DMatrix(train_feat[predictors], train_feat['saleCount'])
    # xgbvalid = xgb.DMatrix(test_feat[predictors])
    # model = xgb.train(params, xgbtrain, num_boost_round=boostRound)
    # param_score = pd.Series(model.get_fscore()).sort_values(ascending=False)
    # print "Parameter score: "
    # print param_score, len(predictors)
    # param_score.to_csv('param_score_train.csv')
    # test_feat.loc[:,'saleCount'] = model.predict(xgbvalid)
    # result = test_feat[['Class','SaleDate','saleCount']]
    # # test_valid_1 = test[test['SaleDate'].isin(week_5[0])]
    # # test_valid_1.fillna(0,inplace=True)
    #
    #
    # result = pd.merge(test[['Class','SaleDate']], result, on=['Class','SaleDate'], how='left')
    # result.fillna(0,inplace=True)
    # result['saleCount'][result['saleCount'] < 0] = 0
    # result['saleCount'] = result['saleCount'].astype('int')
    # result['Class'] = result['Class'].astype('int')
    # # test_o = test_o[test_o['Class'].isin(coupon_class)]
    # result['SaleDate'] = test_o['SaleDate']
    # result.to_csv('result.csv',index=False)
    # result.rename(columns={'Class':'bianma','saleCount':'xiaoliang','SaleDate':'riqi'},inplace=True)
    # result.to_csv('xgb_ret_89_full.csv',index=False,encoding='gbk')
    # # test_valid.to_csv('test_valid.csv',index=False)
    # # score = score(test_valid['saleCount'],result['saleCount'])
    # print "Elapse time is {} minutes".format((time.time() - t0) / (1.0 * 60))
    # # print result
    # # result['saleCount'][result['saleCount'] < 0] = 0
    # # score_1 = score(result['saleCount'],test_valid_1['saleCount'])
    # # print "the 1th day predictive score:{}".format(score_1)



    ### 验证 4月份数据

#     ########################## 提交训练结果 #####################################
#
#     t0 = time.time()
#     # do_not_use_class = [1507,3208,3311,3413]
#     train_o = pd.read_csv(train_path,encoding='gbk',engine='python')
#     test_o = pd.read_csv(test_path)
#     train_o,train_new_o = reshape_train(train_o)
#     # test_o, train_new_o = exclude_class(train_new_o, test_o, coupon_class)
#
#     # train_new_o.SaleDate = train_new_o.SaleDate.map(lambda x: timeHandle(x))
#     # train_new_o.SaleDate = pd.to_datetime(train_new_o.SaleDate)
#     # test_o.SaleDate = test_o.SaleDate.map(lambda x: timeHandle(x))
#     # test_o.SaleDate = pd.to_datetime(train_new_o.SaleDate)
#
#     test_o.loc[:,'saleCount'] = 0
#
#     train_start = '2015-01-01'
#     train_end = '2015-04-28'
#     # train_end = '2015-05-01'
#
#     # 特征1： 提取固定特征
#     train, train_new, test = get_origin_feats(train_o, train_new_o, test_o)
#     train_new = exclude_abnormal_value(train_new)
#     # train_new = exclude_abnormal_value_coupon(train_new)
#     print 'Filter abnormal value.'
#     # 提取五月前的Coupon特征
#     del train_new['Coupon']
#     train_coupon = pd.read_csv(train_coupon_path)
#     train_coupon['SaleDate'] = pd.to_datetime(train_coupon['SaleDate'])
#     train_new = pd.merge(train_new,train_coupon[['Class','SaleDate','Coupon']],on=['Class','SaleDate'],how='left')
#     train_new['Coupon'].fillna(0,inplace=True)
#     print train_new['Coupon'].unique()
#     # 提取五月前的Coupon特征
#     #将预测的5月预测的Coupon特征合并到5月测试集中
#     # del test['Coupon']
#     test_coupon = pd.read_csv(coupon_May_pre_path)
#     test_coupon['SaleDate'] = pd.to_datetime(test_coupon['SaleDate'])
#     test = pd.merge(test,test_coupon[['Class','SaleDate','Coupon']],on=['Class','SaleDate'],how='left')
#     test['Coupon'].fillna(0,inplace=True)
#     print test['Coupon'].unique()
#     #将预测的5月预测的Coupon特征合并到5月测试集中
#
#
#     test_5 = test[test['SaleDate'].isin(week_5[0])]
#     # 特征2： 提取滚动特征
#     train_test = merge_train_test(train_new, test_5)
#     train_test,l_roll_feats = get_roll_feats(train_test)
#
#     train_feat = train_test[train_test['SaleDate'] >= '2015-01-01']    #使用二月份以后的数据
#     train_feat_1 = train_feat[train_feat['SaleDate'] < train_end]   #训练集为2-3月份
#
#     test_feat = train_test[train_test['SaleDate'] >= '2015-05-01']
#     test_feat = test_feat[test_feat['SaleDate'] <= '2015-05-31'] #验证集为四月份
#     # test_valid = test_valid[test_valid['SaleDate'] < '2015-04-27']
#     test_feat.loc[:,'saleCount'] = 0
#
#
#     train_feat_1.fillna(0,inplace=True)
#     test_feat.fillna(0,inplace=True)
#
#     test_feat_1 = test_feat[test_feat['SaleDate'].isin(week_5[0])]
#     test_feat_1.fillna(0,inplace=True)
#     test_feat_1.loc[:,'saleCount'] = 0
#     feature_names = list(train_feat_1.columns)
#     do_not_use_for_training = ['SaleDate','saleCount',
#                                # 'Coupon',
#                                'dayOfYear','price_mean','price_median',
#                                # 'parCumtype','parClass',
#                                # 'parHotPast1MonthIndex',
#                                # 'dayOn21DayDiff',
#                                'lastWeekSaleCount_mean',
#                                # 'expweighted_14_avg',
#                                'trend_7',
#                                'trend_14',
#                                # 'expweighted_7_avg',
#                                'classWeekdayRatio',
#                                # 'parClassWeekdayRatio',
#                                'moving_30_avg',
#                                'expweighted_30_avg',
#                                'moving_21_avg',
#                                'expweighted_21_avg',
#                                'moving_14_avg',
#                                'expweighted_14_avg',
#                                'moving_7_avg',
#                                # 'expweighted_7_avg',
#                                # 'disholDaySaleCount_mean',
#                                'disholDaySaleCount_max',
#                                'last2WeekSaleCount_max','last3WeekSaleCount_max',
#                                # 'Class',
#                                'last3wTot','last4wTot','last2wTot','last1wTot',
#                                'last21d','last28d','last7d','last14d',
#                                'last7d_mean','last21d_mean','last14d_mean','last30d_mean',
#                                # 'last1d',
#                                'last1wMean','last2wMean','last3wMean','last4wMean'
#                                # 'diff_2',
#                                # 'parClass',
#                                # 'holDaySaleCount_min',
#                                # 'holDaySaleCount_min',
#                                # 'holDaySaleCount_median',
#                                # 'holDaySaleCount_mean','month','parHotIndex','dayOn5DayDiff'
#                                # 'last2d','last4d','last'
#                                ]
#
#
#     predictors = [f for f in feature_names if f not in do_not_use_for_training]
#
#     params = {'min_child_weight': 100, 'eta': 0.02, 'colsample_bytree': 0.3, 'max_depth': 7,
#                 'subsample': 0.8, 'lambda': 2, 'nthread': 4, 'booster' : 'gbtree', 'silent': 1,
#                 'eval_metric': 'rmse', 'objective': 'reg:linear'}
#     boostRound = 1500
#
#     xgbtrain = xgb.DMatrix(train_feat_1[predictors], train_feat_1['saleCount'])
#     xgbvalid = xgb.DMatrix(test_feat_1[predictors])
#     model = xgb.train(params, xgbtrain, num_boost_round=boostRound)
#     param_score = pd.Series(model.get_fscore()).sort_values(ascending=False)
#     print "Parameter score: "
#     print param_score, len(predictors)
#     param_score.to_csv('param_score_train.csv')
#     test_feat_1.loc[:,'saleCount'] = model.predict(xgbvalid)
#     result = test_feat_1[['Class','SaleDate','saleCount']]
#     test_valid_1 = test[test['SaleDate'].isin(week_5[0])]
#     # test_valid_1.fillna(0,inplace=True)
#     result = pd.merge(test_valid_1[['Class','SaleDate']], result, on=['Class','SaleDate'], how='left')
#
#
#     result.fillna(0,inplace=True)
#     # print result
#     # result['saleCount'][result['saleCount'] < 0] = 0
#     # score_1 = score(result['saleCount'],test_valid_1['saleCount'])
#     # print "the 1th day predictive score:{}".format(score_1)
#
#
#     # 第二轮
#     # 特征2： 提取滚动特征
#     # 4月2号 - 4月28号
#     for i in range(1,30):
#         # l_roll_feats = []
#         train_test = train_test[train_test['SaleDate'] < train_end]
#         train_test = merge_train_test(train_test, test_feat_1)
#         feats = [f for f in train_test.columns if f not in l_roll_feats]
#         train_test = train_test[feats]
#
#         test_i = test[test['SaleDate'].isin(week_5[i])]
#         train_test = merge_train_test(train_test, test_i)
#         train_test,l_roll_feats = get_roll_feats(train_test)
#
#         train_feat = train_test[train_test['SaleDate'] >= '2015-01-01']    #使用二月份以后的数据
#         # train_feat_1 = train_feat[train_feat['SaleDate'] < '2015-04-08']   #训练集为2-3月份
#         # 排除掉这轮要预测的日期
#         train_feat = train_feat[~train_feat['SaleDate'].isin(week_5[i])]
#         # print week_4[i]
#         # print train_feat['SaleDate']
#
#         test_feat = train_test[train_test['SaleDate'] >= '2015-05-01']
#         test_feat = test_feat[test_feat['SaleDate'] <= '2015-05-30'] #验证集为四月份
#         test_feat.loc[:,'saleCount'] = 0
#
#
#         train_feat_1.fillna(0,inplace=True)
#         test_feat.fillna(0,inplace=True)
#
#         test_feat_1 = test_feat[test_feat['SaleDate'].isin(week_5[i])]
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
#         #             'subsample': 0.6, 'lambda': 1, 'nthread': 4, 'booster' : 'gbtree', 'silent': 1,
#         #             'eval_metric': 'rmse', 'objective': 'reg:linear'}
#
#         xgbtrain = xgb.DMatrix(train_feat_1[predictors], train_feat_1['saleCount'])
#         xgbvalid = xgb.DMatrix(test_feat_1[predictors])
#         if i%7 == 0 :
#           model = xgb.train(params, xgbtrain, num_boost_round=boostRound)
#         param_score = pd.Series(model.get_fscore()).sort_values(ascending=False)
#         # print "Parameter score: "
#         # print param_score
#         test_feat_1.loc[:,'saleCount'] = model.predict(xgbvalid)
#         result_i = test_feat_1[['Class','SaleDate','saleCount']]
#         print len(result_i['saleCount'])
#         test_valid_i = test[test['SaleDate'].isin(week_5[i])]
#         test_valid_i.fillna(0,inplace=True)
#         result_i = pd.merge(test_valid_i[['Class','SaleDate']], result_i, on=['Class','SaleDate'], how='left')
#         result_i.fillna(0,inplace=True)
#         print week_5[i]
#         # result['saleCount'][result['saleCount'] < 0] = 0
#         # print result
#         # score_i = score(test_valid_i['saleCount'],result_i['saleCount'])
#
#         result = pd.concat([result, result_i], axis=0)
#         # print result
#         # print "the {}th day predictive score:{}".format(i + 1,score_i)
#
# del test['saleCount']
# # test = test[test['Class'].isin(coupon_class)]
# result = pd.merge(test[['Class','SaleDate']], result, on=['Class','SaleDate'], how='left')
# result['saleCount'][result['saleCount'] < 0] = 0
# result['saleCount'] = result['saleCount'].astype('int')
# result['Class'] = result['Class'].astype('int')
# # test_o = test_o[test_o['Class'].isin(coupon_class)]
# result['SaleDate'] = test_o['SaleDate']
# result.to_csv('result.csv',index=False)
# result.rename(columns={'Class':u'编码','saleCount':u'销量','SaleDate':u'日期'},inplace=True)
# result.to_csv('result_coupon_full_fb.csv',index=False,encoding='gbk')
# # test_valid.to_csv('test_valid.csv',index=False)
# # score = score(test_valid['saleCount'],result['saleCount'])
# print "Elapse time is {} minutes".format((time.time() - t0) / (1.0 * 60))
# # print "Total predictive score:{}".format(score)
#
# # test_o = pd.read_csv(test_path)
# # result_1739_o = pd.read_csv('result_1739_o.csv',encoding='gbk')
# # result_coupon_full = pd.read_csv('result_coupon_full.csv',encoding='gbk')
# # result_coupon_full = result_coupon_full[result_coupon_full[u'编码'].isin(coupon_class)]
# # result_1739_o = result_1739_o[~result_1739_o[u'编码'].isin(coupon_class)]
# # reslut_1739_combine = pd.concat([result_1739_o,result_coupon_full],axis=0)
# # reslut_1739_combine[u'日期'] = test_o['SaleDate']
# # reslut_1739_combine.to_csv('reslut_1739_combine_reverse_exc3.csv',index=False,encoding='gbk')
#
#     ########################## 提交训练结果 ####################################