# -*- coding:utf-8 -*-
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

import time
# Handle table like and matrices
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Support functions
from feature_process import *
from dataClean_supermarket import *
from util_supermarket import *

# Machine tool kits
import xgboost as xgb
from xgboost import plot_tree
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

train_path = '../input/train.csv'
test_path = '../input/test.csv'
hol_path = '../input/holiday.csv'
train_date_path = '../input/train_date.csv'
cache_path = '../input/cache/'
output_path = '../output/'

def fit_evaluate(df,df_test, train_feature,l_roll_feats, params):
    # print "in procedure"
    # print train_feature
    valid_data = df_test.copy()

    X = df[predictors].values
    y = df['saleCount'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    xgbtrain = xgb.DMatrix(X_train, y_train)
    xgbvalid = xgb.DMatrix(X_test,y_test)
    watchlist = [(xgbtrain, 'train'), (xgbvalid, 'valid')]
    regressor = xgb.train(params, xgbtrain, 60, evals=watchlist, early_stopping_rounds=20,
                  maximize=False, verbose_eval=10)

    # params = {'min_child_weight': 100, 'eta': 0.05, 'colsample_bytree': 0.3, 'max_depth': 7,
    #             'subsample': 0.8, 'lambda': 1, 'nthread': 4, 'booster' : 'gbtree', 'silent': 1,
    #             'eval_metric': 'rmse', 'objective': 'reg:linear'}
    # X = df.copy()
    # xgbtrain = xgb.DMatrix(X[train_feature], X['saleCount'])
    # model = xgb.train(params, xgbtrain, num_boost_round=120)
    # regressor = xgb.train(params, xgbtrain, num_boost_round=60)

    feature_importance_dict = regressor.get_fscore()
    fs = ['f%i' % i for i in range(len(feature_names))]
    f1 = pd.DataFrame({'f': list(feature_importance_dict.keys()),
                       'importance': list(feature_importance_dict.values())})
    f2 = pd.DataFrame({'f': fs, 'feature_name': feature_names})
    feature_importance = pd.merge(f1, f2, how='right', on='f')
    feature_importance = feature_importance.fillna(0)
    print feature_importance[['feature_name', 'importance']].sort_values(by='importance', ascending=False)
    print regressor.best_score,regressor.best_iteration
    return regressor, cross_valid(regressor, df,valid_data,train_feature,l_roll_feats),regressor.best_iteration, regressor.best_score

def training(df, train_feature,l_roll_feats, params, best ,vis=False):
    train1 = df[df['SaleDate'] <= '2015-01-31']
    train2 = df[(df['SaleDate'] >= '2015-02-01') & (df['SaleDate'] <= '2015-02-28')]
    train3 = df[(df['SaleDate'] >= '2015-03-01') & (df['SaleDate'] <= '2015-03-30')]
    train4 = df[(df['SaleDate'] >= '2015-04-01') & (df['SaleDate'] <= '2015-04-30')]
    train12 = df[df['SaleDate'] <= '2015-02-28']
    train23 = df[(df['SaleDate'] >= '2015-02-01') & (df['SaleDate'] <= '2015-03-30')]
    train123 = df[df['SaleDate'] <= '2015-03-30']

    f_t = train_feature
    f_t.append('saleCount')
    # print "**" * 20
    # print "train round 1"
    # regressor, loss1, best_iteration1, best_score1 = fit_evaluate(train1, train2,train_feature,l_roll_feats,
    #                                                               params)
    # print "**" * 20
    # print best_iteration1, best_score1, loss1
    #
    # print "train round 2"
    # regressor, loss2, best_iteration2, best_score2 = fit_evaluate(train2, train3,train_feature,l_roll_feats,
    #                                                               params)
    # print "**" * 20
    # print best_iteration2, best_score2, loss2
    #
    # print "train round 3"
    # regressor, loss3, best_iteration3, best_score3 = fit_evaluate(train3, train4,train_feature,l_roll_feats,
    #                                                               params)
    # print "**" * 20
    # print best_iteration3, best_score3, loss3

    print "train round 1"
    regressor, loss1, best_iteration1, best_score1 = fit_evaluate(train12[f_t], train3,train_feature,l_roll_feats,params)
    print "**" * 20
    print best_iteration1, best_score1, loss1

    print "train round 2"
    regressor, loss2, best_iteration2, best_score2 = fit_evaluate(train23, train4,train_feature,l_roll_feats, params)
    print "**" * 20
    print best_iteration2, best_score2, loss2

    print "train round 3"
    regressor, loss3, best_iteration3, best_score3 = fit_evaluate(train123, train4,train_feature,l_roll_feats,params)
    print "**" * 20
    print best_iteration3, best_score3, loss3


    fig, ax = plt.subplots(figsize=(12,12))
    xgb.plot_importance(regressor,ax=ax)
    plt.show()

    # loss = [loss1, loss2, loss3,loss4,loss5,loss6]
    loss = [loss1,loss2,loss3]
    params['score_std'] = np.std(loss)
    params['score'] = str(loss)
    params['mean_score'] = np.mean(loss)
    params['n_estimators'] = str([best_iteration1,best_iteration2,best_iteration3])
    params['best_score'] = str([best_score1,best_score2,best_score3])

    if np.mean(loss) <= best:
        best = np.mean(loss)
        print "best with: " + str(params)

    return best

def submit(regressor,train_test,test, train_feature,month,l_roll_feats,params,test_o):
    feats = [f for f in train_test.columns if f not in l_roll_feats]
    test_template = test[['Class','SaleDate']].copy()
    batch_0 = train_test[train_test['SaleDate'] == month[0]].copy()
    batch_0['saleCount'] = 0
    xgbvalid = xgb.DMatrix(batch_0[train_feature])
    batch_0['saleCount'] = regressor.predict(xgbvalid)
    result = batch_0[['Class','SaleDate','saleCount']]
    train_test['saleCount'][train_test['SaleDate'].isin([month[0]])] = batch_0['saleCount']
    for index in range(1,len(month)):
        train_test = train_test[feats]
        batch = test[test['SaleDate'] == month[index]].copy()
        train_test = pd.concat([train_test,batch],axis=0)
        train_test,l_roll_feats = get_roll_feats(train_test)
        train_test.fillna(0,inplace=True)
        batch = train_test[train_test['SaleDate'] == month[index]].copy()
        batch['saleCount'] = 0
        xgbvalid = xgb.DMatrix(batch[train_feature])
        batch['saleCount'] = regressor.predict(xgbvalid)
        result_i = batch[['Class','SaleDate','saleCount']]
        result = pd.concat([result,result_i],axis=0)
        train_test['saleCount'][train_test['SaleDate'].isin([month[index]])] = batch['saleCount']
    result = pd.merge(test_template[['Class','SaleDate']], result,on = ['Class','SaleDate'],how='left')
    result.fillna(0,inplace=True)
    result['saleCount'][result['saleCount'] < 0] = 0
    result['saleCount'] = result['saleCount'].astype('int')
    result['Class'] = result['Class'].astype('int')
    result['SaleDate'] = test_o['SaleDate']
    result.rename(columns={'Class':u'编码','saleCount':u'销量','SaleDate':u'日期'},inplace=True)
    result.to_csv('result.csv',index=False,encoding='gbk')


    # train_df = df.loc[df['time_interval_begin'] < pd.to_datetime('2017-07-01')]
    #
    # train_df = train_df.dropna()
    # X = train_df[train_feature].values
    # y = train_df['travel_time'].values
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    #
    # eval_set = [(X_test, y_test)]
    # regressor = xgb.XGBRegressor(learning_rate=params['learning_rate'], n_estimators=params['n_estimators'],
    #                              booster='gbtree', objective='reg:linear', n_jobs=-1, subsample=params['subsample'],
    #                              colsample_bytree=params['colsample_bytree'], random_state=0,
    #                              max_depth=params['max_depth'], gamma=params['gamma'],
    #                              min_child_weight=params['min_child_weight'], reg_alpha=params['reg_alpha'])
    # regressor.fit(X_train, y_train, verbose=True, early_stopping_rounds=10, eval_metric=mape_ln,
    #               eval_set=eval_set)
    # feature_vis(regressor, train_feature)
    # joblib.dump(regressor, 'model/xgbr.pkl')
    # print regressor
    # submission(train_feature, regressor, df, 'submission/xgbr1.txt', 'submission/xgbr2.txt', 'submission/xgbr3.txt',
    #            'submission/xgbr4.txt')

if __name__ == "__main__":
    month_4 = [
                '2015-04-01','2015-04-02','2015-04-03','2015-04-04','2015-04-05','2015-04-06','2015-04-07',
                '2015-04-08','2015-04-09','2015-04-10','2015-04-11','2015-04-12','2015-04-13','2015-04-14',
                '2015-04-15','2015-04-16','2015-04-17','2015-04-18','2015-04-19','2015-04-20','2015-04-21',
                '2015-04-22','2015-04-23','2015-04-24','2015-04-25','2015-04-26','2015-04-27','2015-04-28',
                '2015-04-29','2015-04-30'
            ]
    month_5 = [
            '2015-05-01','2015-05-02','2015-05-03','2015-05-04','2015-05-05','2015-05-06','2015-05-07',
            '2015-05-08','2015-05-09','2015-05-10','2015-05-11','2015-05-12','2015-05-13','2015-05-14',
            '2015-05-15','2015-05-16','2015-05-17','2015-05-18','2015-05-19','2015-05-20','2015-05-21',
            '2015-05-22','2015-05-23','2015-05-24','2015-05-25','2015-05-26','2015-05-27','2015-05-28',
            '2015-05-29','2015-05-30'
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



    t0 = time.time()
    train = pd.read_csv(train_path,encoding='gbk',engine='python')
    test_o = pd.read_csv(test_path)
    train,train_new = reshape_train(train)
    # 验证集中不预测的类
    # train, train_new = exclude_class(train_new, train, do_not_use_class)

    # 特征1： 提取固定特征
    train_new = exclude_abnormal_value(train_new)
    print 'Filter abnormal value.'
    train, train_new, test = get_origin_feats(train, train_new, test_o)
    # 特征2： 提取滚动特征
# #-------------------  submission -------------------------
#     test.fillna(0,inplace=True)
#     test_0 =  test[test['SaleDate'] == month_5[0]].copy()
#     test_0.fillna(0,inplace=True)
#     train_test  = pd.concat([train_new, test_0],axis=0)
#     train_test,l_roll_feats = get_roll_feats(train_test)
#     train_new,l_roll_feats = get_roll_feats(train_new)
#     train_test.fillna(0,inplace=True)
#     train_new.fillna(0,inplace=True)
# #-------------------  submission -------------------------

# -------------------  training  --------------------
    train_new,l_roll_feats = get_roll_feats(train_new)
    train_new.fillna(0,inplace=True)
# -------------------  training  --------------------

    do_not_use_for_training = ['SaleDate','saleCount','Coupon',
                               'dayOfYear','price_mean','price_median',
                               # 'parCumtype','parClass',
                               # 'parHotPast1MonthIndex',
                               # 'dayOn21DayDiff',
                               'lastWeekSaleCount_mean',
                               # 'expweighted_14_avg',

                               'trend_7','expweighted_7_avg',
                               # 'classWeekdayRatio',
                               'trend_7',
                               # 'expweighted_7_avg',
                               'classWeekdayRatio',
                               # 'parClassWeekdayRatio',
                               'moving_30_avg','expweighted_30_avg',
                               # 'disholDaySaleCount_mean',
                               'disholDaySaleCount_max',
                               'last2WeekSaleCount_max','last3WeekSaleCount_max',
                               'Class'
                               # 'parClass',
                               # 'holDaySaleCount_min',
                               # 'holDaySaleCount_min',
                               # 'holDaySaleCount_median',
                               # 'holDaySaleCount_mean','month','parHotIndex','dayOn5DayDiff'
                               ]
    feature_names = list(train_new.columns)
    predictors = [f for f in feature_names if f not in do_not_use_for_training]
    print predictors

    params = {
        'learning_rate': 0.05,
        'n_estimators': 10,
        'subsample': 0.8,
        'colsample_bytree': 0.3,
        'max_depth': 7,
        'min_child_weight': 100,
        'reg_alpha': 2,
        'lambda': 1,
        'gamma': 0,
        'nthread': 4,
        'silent':1,
        'booster' : 'gbtree'
    }

# -------------------  training  --------------------
    best = 1
    best = training(train_new, predictors,l_roll_feats, params, best ,vis=True)
    print "Mean predictive score is: {}".format(best)
# -------------------  training  --------------------

# #-------------------  submission -------------------------
#     X = train_new[predictors].values
#     y = train_new['saleCount'].values
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
#
#     xgbtrain = xgb.DMatrix(X_train, y_train)
#     xgbvalid = xgb.DMatrix(X_test,y_test)
#     watchlist = [(xgbtrain, 'train'), (xgbvalid, 'valid')]
#     regressor = xgb.train(params, xgbtrain, 60, evals=watchlist, early_stopping_rounds=50,
#                   maximize=False, verbose_eval=10)
#     feature_importance_dict = regressor.get_fscore()
#     fs = ['f%i' % i for i in range(len(feature_names))]
#     f1 = pd.DataFrame({'f': list(feature_importance_dict.keys()),
#                        'importance': list(feature_importance_dict.values())})
#     f2 = pd.DataFrame({'f': fs, 'feature_name': feature_names})
#     feature_importance = pd.merge(f1, f2, how='right', on='f')
#     feature_importance = feature_importance.fillna(0)
#     print feature_importance[['feature_name', 'importance']].sort_values(by='importance', ascending=False)
#     print regressor.best_score,regressor.best_iteration
#     submit(regressor,train_test,test, predictors,month_5,l_roll_feats,params,test_o)
# #-------------------  submission -------------------------
    print "Elapse time is {} minutes".format((time.time() - t0) / (1.0 * 60))






#     xgbtrain = xgb.DMatrix(train_feat_1[predictors], train_feat_1['saleCount'])
#     xgbvalid = xgb.DMatrix(test_feat_1[predictors])
#
#     X = train_feat_1[predictors].values
#     y = train_feat_1['saleCount'].values
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
#
#     eval_set = [(X_test, y_test)]
#     regressor = xgb.XGBRegressor(learning_rate=params['learning_rate'], n_estimators=params['n_estimators'],
#                                 objective='reg:linear', nthread=4, subsample=params['subsample'],
#                                  colsample_bytree=params['colsample_bytree'],
#                                  max_depth=params['max_depth'], gamma=params['gamma'],
#                                  min_child_weight=params['min_child_weight'], reg_alpha=params['reg_alpha'])
#     regressor.fit(X_train, y_train, verbose=True, early_stopping_rounds=20, eval_metric='rmse',
#                   eval_set=eval_set)
#
#
#
#
#     boostRound = 120
#
#     # print train_feat_1[predictors]
#
#     # xgbtrain = xgb.DMatrix(train_feat_1[predictors], train_feat_1['saleCount'])
#     # xgbvalid = xgb.DMatrix(test_feat_1[predictors])
#     # model = xgb.train(params, xgbtrain, num_boost_round=boostRound)
#     # param_score = pd.Series(model.get_fscore()).sort_values(ascending=False)
#     # print "Parameter score: "
#     # print param_score, len(predictors)
#     # test_feat_1.loc[:,'saleCount'] = model.predict(xgbvalid)
#     test_feat_1.loc[:,'saleCount'] = regressor.predict(test_feat_1[predictors].values)
#     result = test_feat_1[['Class','SaleDate','saleCount']]
#
#     # result['saleCount'] = 1.3 *  result['saleCount']
#     test_valid_1 = test_valid[test_valid['SaleDate'].isin(week_4[0])]
#     test_valid_1.fillna(0,inplace=True)
#     result = pd.merge(test_valid_1[['Class','SaleDate']], result, on=['Class','SaleDate'], how='left')
#     result['saleCount'][result['saleCount'] < 0] = 0
#     result.fillna(0,inplace=True)
#
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
#
#         feature_names = list(train_feat_1.columns)
#         predictors = [f for f in feature_names if f not in do_not_use_for_training]
#         # feature_names = list(train_feat_1.columns)
#         # predictors = [f for f in feature_names if f not in do_not_use_for_training]
#
#
#         xgbtrain = xgb.DMatrix(train_feat_1[predictors], train_feat_1['saleCount'])
#         xgbvalid = xgb.DMatrix(test_feat_1[predictors])
#
#         model = xgb.train(params, xgbtrain, num_boost_round=boostRound)
#         # model = xgb.train(params, xgbtrain, num_boost_round=boostRound)
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
#         score_i = score(test_valid_i['saleCount'],result_i['saleCount'])
#         result = pd.concat([result, result_i], axis=0)
#         print "the {}th day predictive score:{}".format(i + 1,score_i)
#
# result = pd.merge(test_valid[['Class','SaleDate']], result, on=['Class','SaleDate'], how='left')
# result['saleCount'][result['saleCount'] < 0] = 0
#
#
#
# print result['saleCount']
# result.to_csv('result.csv',index=False)
# test_valid.to_csv('test_valid.csv',index=False)
# score_f = score(test_valid['saleCount'],result['saleCount'])
#
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



    ### 验证 4月份数据

# #     ########################## 提交训练结果 #####################################
# #
#     t0 = time.time()
#     # do_not_use_class = [1507,3208,3311,3413]
#     train_o = pd.read_csv(train_path,encoding='gbk',engine='python')
#     test_o = pd.read_csv(test_path)
#     train_o,train_new_o = reshape_train(train_o)
#     # test_o, train_new_o = exclude_class(train_new_o, test_o, do_not_use_class)

#
#     # train_new_o.SaleDate = train_new_o.SaleDate.map(lambda x: timeHandle(x))
#     # train_new_o.SaleDate = pd.to_datetime(train_new_o.SaleDate)
#     # test_o.SaleDate = test_o.SaleDate.map(lambda x: timeHandle(x))
#     # test_o.SaleDate = pd.to_datetime(train_new_o.SaleDate)
#
#
#     # train_new_o.SaleDate = train_new_o.SaleDate.map(lambda x: timeHandle(x))
#     # train_new_o.SaleDate = pd.to_datetime(train_new_o.SaleDate)
#     # test_o.SaleDate = test_o.SaleDate.map(lambda x: timeHandle(x))
#     # test_o.SaleDate = pd.to_datetime(train_new_o.SaleDate)
#
#     test_o.loc[:,'saleCount'] = 0
#
#     # 特征1： 提取固定特征
#     train, train_new, test = get_origin_feats(train_o, train_new_o, test_o)
#     train_new = exclude_abnormal_value(train_new)
#     print 'Filter abnormal value.'
#
#
#     test_5 = test[test['SaleDate'].isin(week_5[0])]
#     # 特征2： 提取滚动特征
#     train_test = merge_train_test(train_new, test_5)
#     train_test,l_roll_feats = get_roll_feats(train_test)
#
#     train_feat = train_test[train_test['SaleDate'] >= '2015-01-01']    #使用二月份以后的数据
#     train_feat_1 = train_feat[train_feat['SaleDate'] < '2015-05-01']   #训练集为2-3月份
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
#     do_not_use_for_training = ['SaleDate','saleCount','Coupon',
#                                'dayOfYear','price_mean','price_median',
#                                # 'parCumtype','parClass',
#                                # 'parHotPast1MonthIndex',
#                                # 'dayOn21DayDiff',
#                                'lastWeekSaleCount_mean',
#                                # 'expweighted_14_avg',

#                                'trend_7','expweighted_7_avg',
#                                # 'classWeekdayRatio', 'parClassWeekdayRatio',
#                                'trend_7',
#                                # 'expweighted_7_avg',
#                                'classWeekdayRatio',
#                                # 'parClassWeekdayRatio',
#                                'moving_30_avg','expweighted_30_avg',
#                                # 'disholDaySaleCount_mean',
#                                'disholDaySaleCount_max',
#                                'last2WeekSaleCount_max','last3WeekSaleCount_max',
#                                'Class',
#                                # 'parClass',
#                                # 'holDaySaleCount_min',
#                                # 'holDaySaleCount_min',
#                                # 'holDaySaleCount_median',
#                                # 'holDaySaleCount_mean','month','parHotIndex','dayOn5DayDiff'
#                                ]
#     predictors = [f for f in feature_names if f not in do_not_use_for_training]
#
#     params = {'min_child_weight': 100, 'eta': 0.05, 'colsample_bytree': 0.3, 'max_depth': 7,
#                 'subsample': 0.8, 'lambda': 1, 'nthread': 4, 'booster' : 'gbtree', 'silent': 1,
#                 'eval_metric': 'rmse', 'objective': 'reg:linear'}
#     boostRound = 120
#
#     xgbtrain = xgb.DMatrix(train_feat_1[predictors], train_feat_1['saleCount'])
#     xgbvalid = xgb.DMatrix(test_feat_1[predictors])
#     model = xgb.train(params, xgbtrain, num_boost_round=boostRound)
#     param_score = pd.Series(model.get_fscore()).sort_values(ascending=False)
#     print "Parameter score: "
#     print param_score, len(predictors)
#     test_feat_1.loc[:,'saleCount'] = model.predict(xgbvalid)
#     result = test_feat_1[['Class','SaleDate','saleCount']]
#     test_valid_1 = test[test['SaleDate'].isin(week_5[0])]
#     # test_valid_1.fillna(0,inplace=True)
#     result = pd.merge(test_valid_1[['Class','SaleDate']], result, on=['Class','SaleDate'], how='left')
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
#     for i in range(1,5):
#         # l_roll_feats = []
#         train_test = train_test[train_test['SaleDate'] < '2015-05-01']
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

#         model = xgb.train(params, xgbtrain, num_boost_round=boostRound)

#         # model = xgb.train(params, xgbtrain, num_boost_round=boostRound)

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
#         result = pd.concat([result, result_i], axis=0)
#         # print result
#         # print "the {}th day predictive score:{}".format(i + 1,score_i)
#
# del test['saleCount']
# result = pd.merge(test[['Class','SaleDate']], result, on=['Class','SaleDate'], how='left')
# result['saleCount'][result['saleCount'] < 0] = 0
# result['saleCount'] = result['saleCount'].astype('int')
# result['Class'] = result['Class'].astype('int')
# result['SaleDate'] = test_o['SaleDate']
# result.to_csv('result.csv',index=False)
# result.rename(columns={'Class':u'编码','saleCount':u'销量','SaleDate':u'日期'},inplace=True)
# result.to_csv('result.csv',index=False,encoding='gbk')
# # test_valid.to_csv('test_valid.csv',index=False)
# # score = score(test_valid['saleCount'],result['saleCount'])
# print "Elapse time is {} minutes".format((time.time() - t0) / (1.0 * 60))
# # print "Total predictive score:{}".format(score)
#
#     ########################## 提交训练结果 ####################################