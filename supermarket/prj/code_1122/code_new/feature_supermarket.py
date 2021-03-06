# -*- coding:utf-8 -*-
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Handle table like and matrices
import pandas as pd
import numpy as np

train_path = '../input_new/train.csv'
test_path = '../input_new/test.csv'
hol_path = '../input_new/holiday.csv'
train_date_path = '../input_new/train_date.csv'
# cache_path = '../input/cache/'
# output_path = '../output/'
# train_fb_path = '../input/train_fb.csv'

def reshape_train(train):
    train_date = pd.read_csv(train_date_path)
    # train.rename(columns = {'BigCode':'parClass','MidCode':'Class'},inplace = True)
    train.loc[:,'saleCount'] = 1
    coord_class = train.groupby(['Class','SaleDate'],as_index=False)['saleCount'].sum()
    coord_parClass = train.groupby(['parClass','SaleDate'],as_index=False)['saleCount'].sum()

    coord_class_c = train.groupby(['Class','SaleDate'],as_index=False)['Coupon'].sum()
    coord_parClass_c = train.groupby(['parClass','SaleDate'],as_index=False)['Coupon'].sum()
    coord_class = pd.merge(coord_class, coord_class_c, on= ['Class','SaleDate'], how='left')
    coord_parClass = pd.merge(coord_parClass, coord_parClass_c, on= ['parClass','SaleDate'], how='left')
    # coord_class['Coupon'].fillna(0,inplace=True)
    # coord_parClass['Coupon'].fillna(0,inplace=True)

    # print coord_class
    coord_parClass.rename(columns = {'parClass':'Class'},inplace=True)
    train_new = pd.concat([coord_class,coord_parClass],axis = 0)
    train_new.loc[:,'parClass'] = train_new.Class.map(lambda x: str(x)[:2])
    train_new.loc[:,'parClass'] = train_new.parClass.map(lambda x: int(x))
    l = train_new.Class.unique()
    tmp1 = train_date.copy()
    tmp2 = train_date.copy()
    tmp1.loc[:,'Class'] = 0
    for i in l:
        tmp2.loc[:,'Class'] = i
        tmp1 = pd.concat([tmp1,tmp2],axis=0)
    tmp1 = tmp1[tmp1.Class > 0]
    tmp1 = tmp1[['Class','SaleDate']]
    tmp1 = pd.merge(tmp1, train_new, on=['Class','SaleDate'], how='left')
    tmp1.saleCount.fillna(0,inplace=True)
    tmp1.loc[:,'parClass'] = tmp1.Class.map(lambda x: str(x)[:2])
    tmp1.parClass = tmp1.parClass.astype('int')
    tmp1.saleCount = tmp1.saleCount.astype('int')
    tmp1 = tmp1[['Class','parClass','SaleDate','saleCount','Coupon']]
    print 'Are count numbers equal: ', train.saleCount.sum() * 2 == tmp1.saleCount.sum()
    train_new = tmp1.copy()

    train['Coupon'].fillna(0,inplace=True)
    train_new['Coupon'].fillna(0,inplace=True)
    return train, train_new

def get_hol_feats(train, train_new, test):
    holiday = pd.read_csv(hol_path)
    train = pd.merge(train, holiday, on = 'SaleDate',how = 'left')
    train_new = pd.merge(train_new, holiday, on = 'SaleDate',how = 'left')
    test = pd.merge(test, holiday, on = 'SaleDate',how = 'left')
    return train, train_new, test

def timeHandle(s):
    s = str(s)
    s = [s[:4],s[4:6],s[6:]]
    return '-'.join(s)

def get_time_feats(train, train_new, test):
    train.SaleDate = train.SaleDate.map(lambda x: timeHandle(x))
    train.SaleDate = pd.to_datetime(train.SaleDate)
    train.loc[:,'month'] = train.SaleDate.dt.month
    train.loc[:,'dayOfWeek'] = train.SaleDate.dt.dayofweek
    train.loc[:,'dayOfYear'] = train.SaleDate.dt.dayofyear
    train.loc[:,'weekOfYear'] = train.SaleDate.dt.weekofyear

    train_new.SaleDate = train_new.SaleDate.map(lambda x: timeHandle(x))
    train_new.SaleDate = pd.to_datetime(train_new.SaleDate)
    train_new.loc[:,'month'] = train_new.SaleDate.dt.month
    train_new.loc[:,'dayOfWeek'] = train_new.SaleDate.dt.dayofweek
    train_new.loc[:,'dayOfYear'] = train_new.SaleDate.dt.dayofyear
    train_new.loc[:,'weekOfYear'] = train_new.SaleDate.dt.weekofyear

    test.SaleDate = test.SaleDate.map(lambda x: timeHandle(x))
    test.SaleDate = pd.to_datetime(test.SaleDate)
    test.loc[:,'month'] = test.SaleDate.dt.month
    test.loc[:,'dayOfWeek'] = test.SaleDate.dt.dayofweek
    test.loc[:,'dayOfYear'] = test.SaleDate.dt.dayofyear
    test.loc[:,'weekOfYear'] = test.SaleDate.dt.weekofyear
    return train, train_new, test

def  get_commodity_class(train, train_new, test):
    cumDict = {u'一般商品':0.6089,u'生鲜':0.3782,u'联营商品':0.0129}
    train.CumType = train.CumType.map(cumDict)
    midClassSet = set(train.Class)
    bigClassSet = set(train.parClass)
    midClassDict = {}
    bigClassDict = {}
    classDict = {}
    for eachMid in midClassSet:
        coord = train[train.Class == eachMid].groupby('CumType')['Class'].count()
        sum = 0
        for i in range(len(coord)):
            sum += coord.index[i] *  coord.values[i]
        rate = round(sum / (1.0 * len(coord) + 1),2) ##修正
        midClassDict[eachMid] = rate
        classDict[eachMid] = rate
    for eachBig in bigClassSet:
        coord = train[train.parClass == eachBig].groupby('CumType')['Class'].count()
        sum = 0
        for i in range(len(coord)):
            sum += coord.index[i] *  coord.values[i]
        rate = round(sum / (1.0 * len(coord) + 1),2)
        bigClassDict[eachBig] = rate
        classDict[eachBig] = rate
    train.loc[:,'cumType'] = train.Class.map(midClassDict)
    train.loc[:,'parCumType'] = train.parClass.map(bigClassDict)

    train_new.loc[:,'cumType'] = train_new.Class.map(midClassDict)
    train_new.loc[:,'parCumType'] = train_new.parClass.map(bigClassDict)
    train_new['cumType'][train_new['cumType'].isnull()] = train_new['parCumType'][train_new['cumType'].isnull()]

    #最开始就需要改名
    test.rename(columns={'Code':'Class'},inplace = True)
    test.loc[:,'parClass'] = test.Class.map(lambda x: str(x)[:2])
    test.loc[:,'parClass'] = test.parClass.map(lambda x: int(x))
    test.loc[:,'cumType'] = test.Class.map(classDict)
    test.loc[:,'parCumType'] = test.parClass.map(bigClassDict)

    # 测试集中类cumType的缺失值处理,使用临近中类的type值
    test = test.fillna(method = 'pad')
    return train, train_new, test

def get_commodity_hot_index(train,train_new,test):
    midClassSet = set(train.Class)
    bigClassSet = set(train.parClass)
    hotIndexDict = {}
    parHotIndexDict = {}
    totHotIndexDict = {}
    totSaleCount = train.shape[0]
    for eachMid in midClassSet:
        rate = round(train[train.Class == eachMid].shape[0] / (1.0 * totSaleCount),5)
        hotIndexDict[eachMid] = rate
        totHotIndexDict[eachMid] = rate
    for eachBig in bigClassSet:
        rate = round(train[train.parClass == eachBig].shape[0] / (1.0 * totSaleCount),5)
        parHotIndexDict[eachBig] = rate
        totHotIndexDict[eachBig] = rate

    train.loc[:,'hotIndex'] = train.Class.map(hotIndexDict)
    train.loc[:,'parHotIndex'] = train.parClass.map(parHotIndexDict)

    train_new.loc[:,'hotIndex'] = train.Class.map(hotIndexDict)
    train_new.loc[:,'parHotIndex'] = train.parClass.map(parHotIndexDict)

    test.loc[:,'hotIndex'] = train.Class.map(totHotIndexDict)
    test.loc[:,'parHotIndex'] = train.parClass.map(parHotIndexDict)
    return train,train_new,test

def get_weekday_ratio_feats(train_new,test):
    coord = train_new.groupby(['Class','dayOfWeek'],as_index=False)['saleCount'].agg({'dayOfWeekCount':'sum'})
    var = train_new.groupby(['Class'],as_index=False)['saleCount'].agg({'classCount':'sum'})
    coord = pd.merge(coord, var, on = 'Class',how='left' )
    coord.loc[:,'classWeekdayRatio'] = coord['dayOfWeekCount'] / (1.0 * (coord['classCount'] + 1))
    train_new = pd.merge(train_new, coord[['Class','dayOfWeek','classWeekdayRatio']], on = ['Class','dayOfWeek'], how='left' )
    test = pd.merge(test, coord[['Class','dayOfWeek','classWeekdayRatio']], on = ['Class','dayOfWeek'], how='left' )

    coord = train_new.groupby(['parClass','dayOfWeek'],as_index=False)['saleCount'].agg({'dayOfWeekCount':'sum'})
    var = train_new.groupby(['parClass'],as_index=False)['saleCount'].agg({'parClassCount':'sum'})
    coord = pd.merge(coord, var, on = 'parClass',how='left' )
    coord.loc[:,'parClassWeekdayRatio'] = coord['dayOfWeekCount'] / (1.0 * (coord['parClassCount'] + 1))
    train_new = pd.merge(train_new, coord[['parClass','dayOfWeek','parClassWeekdayRatio']], on = ['parClass','dayOfWeek'], how='left' )
    test = pd.merge(test, coord[['parClass','dayOfWeek','parClassWeekdayRatio']], on = ['parClass','dayOfWeek'], how='left' )

    coord = train_new.groupby(['Class','dayOfWeek'],as_index=False)['saleCount'].agg({'dayOfWeekCount':'mean'})
    var = train_new.groupby(['Class'],as_index=False)['saleCount'].agg({'classCount':'mean'})
    coord = pd.merge(coord, var, on = 'Class',how='left' )
    coord.loc[:,'classWeekdayRatio_mean'] = coord['dayOfWeekCount'] / (1.0 * (coord['classCount'] + 1))
    train_new = pd.merge(train_new, coord[['Class','dayOfWeek','classWeekdayRatio_mean']], on = ['Class','dayOfWeek'], how='left' )
    test = pd.merge(test, coord[['Class','dayOfWeek','classWeekdayRatio_mean']], on = ['Class','dayOfWeek'], how='left' )

    coord = train_new.groupby(['parClass','dayOfWeek'],as_index=False)['saleCount'].agg({'dayOfWeekCount':'mean'})
    var = train_new.groupby(['parClass'],as_index=False)['saleCount'].agg({'parClassCount':'mean'})
    coord = pd.merge(coord, var, on = 'parClass',how='left' )
    coord.loc[:,'parclassWeekdayRatio_mean'] = coord['dayOfWeekCount'] / np.round((1.0 * (coord['parClassCount'] + 1)))
    train_new = pd.merge(train_new, coord[['parClass','dayOfWeek','parclassWeekdayRatio_mean']], on = ['parClass','dayOfWeek'], how='left' )
    test = pd.merge(test, coord[['parClass','dayOfWeek','parclassWeekdayRatio_mean']], on = ['parClass','dayOfWeek'], how='left' )

    coord = train_new.groupby(['Class','dayOfWeek'],as_index=False)['saleCount'].agg({'dayOfWeekCount':'median'})
    var = train_new.groupby(['Class'],as_index=False)['saleCount'].agg({'classCount':'median'})
    coord = pd.merge(coord, var, on = 'Class',how='left' )
    coord.loc[:,'classWeekdayRatio_median'] = coord['dayOfWeekCount'] / (1.0 * (coord['classCount'] + 1))
    train_new = pd.merge(train_new, coord[['Class','dayOfWeek','classWeekdayRatio_median']], on = ['Class','dayOfWeek'], how='left' )
    test = pd.merge(test, coord[['Class','dayOfWeek','classWeekdayRatio_median']], on = ['Class','dayOfWeek'], how='left' )

    coord = train_new.groupby(['parClass','dayOfWeek'],as_index=False)['saleCount'].agg({'dayOfWeekCount':'median'})
    var = train_new.groupby(['parClass'],as_index=False)['saleCount'].agg({'parClassCount':'median'})
    coord = pd.merge(coord, var, on = 'parClass',how='left' )
    coord.loc[:,'parclassWeekdayRatio_median'] = coord['dayOfWeekCount'] / (1.0 * (coord['parClassCount'] + 1))
    train_new = pd.merge(train_new, coord[['parClass','dayOfWeek','parclassWeekdayRatio_median']], on = ['parClass','dayOfWeek'], how='left' )
    test = pd.merge(test, coord[['parClass','dayOfWeek','parclassWeekdayRatio_median']], on = ['parClass','dayOfWeek'], how='left' )

    coord = train_new.groupby(['Class','dayOfWeek'],as_index=False)['saleCount'].agg({'dayOfWeekCount':'max'})
    var = train_new.groupby(['Class'],as_index=False)['saleCount'].agg({'classCount':'max'})
    coord = pd.merge(coord, var, on = 'Class',how='left' )
    coord.loc[:,'classWeekdayRatio_max'] = coord['dayOfWeekCount'] / (1.0 * (coord['classCount'] + 1))
    train_new = pd.merge(train_new, coord[['Class','dayOfWeek','classWeekdayRatio_max']], on = ['Class','dayOfWeek'], how='left' )
    test = pd.merge(test, coord[['Class','dayOfWeek','classWeekdayRatio_max']], on = ['Class','dayOfWeek'], how='left' )

    coord = train_new.groupby(['parClass','dayOfWeek'],as_index=False)['saleCount'].agg({'dayOfWeekCount':'max'})
    var = train_new.groupby(['parClass'],as_index=False)['saleCount'].agg({'parClassCount':'max'})
    coord = pd.merge(coord, var, on = 'parClass',how='left' )
    coord.loc[:,'parclassWeekdayRatio_max'] = coord['dayOfWeekCount'] / (1.0 * (coord['parClassCount'] + 1))
    train_new = pd.merge(train_new, coord[['parClass','dayOfWeek','parclassWeekdayRatio_max']], on = ['parClass','dayOfWeek'], how='left' )
    test = pd.merge(test, coord[['parClass','dayOfWeek','parclassWeekdayRatio_max']], on = ['parClass','dayOfWeek'], how='left' )

    coord = train_new.groupby(['Class','dayOfWeek'],as_index=False)['saleCount'].agg({'dayOfWeekCount':'std'})
    var = train_new.groupby(['Class'],as_index=False)['saleCount'].agg({'classCount':'std'})
    coord = pd.merge(coord, var, on = 'Class',how='left' )
    coord.loc[:,'classWeekdayRatio_std'] = coord['dayOfWeekCount'] / (1.0 * (coord['classCount'] + 1))
    train_new = pd.merge(train_new, coord[['Class','dayOfWeek','classWeekdayRatio_std']], on = ['Class','dayOfWeek'], how='left' )
    test = pd.merge(test, coord[['Class','dayOfWeek','classWeekdayRatio_std']], on = ['Class','dayOfWeek'], how='left' )

    coord = train_new.groupby(['parClass','dayOfWeek'],as_index=False)['saleCount'].agg({'dayOfWeekCount':'std'})
    var = train_new.groupby(['parClass'],as_index=False)['saleCount'].agg({'parClassCount':'std'})
    coord = pd.merge(coord, var, on = 'parClass',how='left' )
    coord.loc[:,'parclassWeekdayRatio_std'] = coord['dayOfWeekCount'] / (1.0 * (coord['parClassCount'] + 1))
    train_new = pd.merge(train_new, coord[['parClass','dayOfWeek','parclassWeekdayRatio_std']], on = ['parClass','dayOfWeek'], how='left' )
    test = pd.merge(test, coord[['parClass','dayOfWeek','parclassWeekdayRatio_std']], on = ['parClass','dayOfWeek'], how='left' )
    return train_new,test

def get_hol_sale_feats(train_new,test):
    train_wk = train_new[train_new.holidayCluster == 1]
    train_hol = train_new[train_new.holidayCluster != 1]

    #节假日
    coord  = train_new.groupby(['Class','disHoliday_detail'],as_index=False)['saleCount'].agg({'disholDaySaleCount_mean':'mean'})
    train_new = pd.merge(train_new, coord, on = ['Class','disHoliday_detail'], how='left')
    test = pd.merge(test, coord, on = ['Class','disHoliday_detail'], how='left')

    coord  = train_new.groupby(['Class','disHoliday_detail'],as_index=False)['saleCount'].agg({'disholDaySaleCount_max':'max'})
    train_new = pd.merge(train_new, coord, on = ['Class','disHoliday_detail'], how='left')
    test = pd.merge(test, coord, on = ['Class','disHoliday_detail'], how='left')

    coord  = train_new.groupby(['Class','disHoliday_detail'],as_index=False)['saleCount'].agg({'disholDaySaleCount_std':'std'})
    train_new = pd.merge(train_new, coord, on = ['Class','disHoliday_detail'], how='left')
    test = pd.merge(test, coord, on = ['Class','disHoliday_detail'], how='left')

    #工作日
    coord  = train_new.groupby(['Class','disWorkday_detail'],as_index=False)['saleCount'].agg({'diswkDaySaleCount_mean':'mean'})
    train_new = pd.merge(train_new, coord, on = ['Class','disWorkday_detail'], how='left')
    test = pd.merge(test, coord, on = ['Class','disWorkday_detail'], how='left')

    coord  = train_new.groupby(['Class','disWorkday_detail'],as_index=False)['saleCount'].agg({'diswkDaySaleCount_median':'median'})
    train_new = pd.merge(train_new, coord, on = ['Class','disWorkday_detail'], how='left')
    test = pd.merge(test, coord, on = ['Class','disWorkday_detail'], how='left')

    coord  = train_new.groupby(['Class','disWorkday_detail'],as_index=False)['saleCount'].agg({'diswkDaySaleCount_max':'max'})
    train_new = pd.merge(train_new, coord, on = ['Class','disWorkday_detail'], how='left')
    test = pd.merge(test, coord, on = ['Class','disWorkday_detail'], how='left')

    coord  = train_new.groupby(['Class','disWorkday_detail'],as_index=False)['saleCount'].agg({'diswkDaySaleCount_std':'std'})
    train_new = pd.merge(train_new, coord, on = ['Class','disWorkday_detail'], how='left')
    test = pd.merge(test, coord, on = ['Class','disWorkday_detail'], how='left')

    # coord  = train_new.groupby(['Class','disHoliday_detail'],as_index=False)['saleCount'].agg({'disholDaySaleCount_std':'min'})
    # train_new = pd.merge(train_new, coord, on = ['Class','disHoliday_detail'], how='left')
    # test = pd.merge(test, coord, on = ['Class','disHoliday_detail'], how='left')


    coord = train_new[train_new['disHoliday_detail'] == 0].groupby(['Class','disHoliday_detail'],as_index=False)['saleCount'].agg({'disholDaySaleCount_0_mean':'mean'})
    coord.fillna(0.01,inplace=True)
    train_new = pd.merge(train_new, coord, on = ['Class','disHoliday_detail'], how='left')
    test = pd.merge(test, coord, on = ['Class','disHoliday_detail'], how='left')
    train_new.loc[:,'diswkHolRatio'] = train_new['disholDaySaleCount_mean'] / train_new['disholDaySaleCount_0_mean']
    test.loc[:,'diswkHolRatio'] = test['disholDaySaleCount_mean'] / test['disholDaySaleCount_0_mean']
    del train_new['disholDaySaleCount_0_mean'],test['disholDaySaleCount_0_mean']

    coord = train_wk.groupby('Class',as_index = False)['saleCount'].agg({'wkDaySaleCount_median':'median'})
    train_new = pd.merge(train_new, coord, on = 'Class', how='left')
    test = pd.merge(test, coord, on = 'Class', how='left')
    coord = train_wk.groupby('Class',as_index = False)['saleCount'].agg({'wkDaySaleCount_mean':'mean'})
    train_new = pd.merge(train_new, coord, on = 'Class', how='left')
    test = pd.merge(test, coord, on = 'Class', how='left')
    coord = train_wk.groupby('Class',as_index = False)['saleCount'].agg({'wkDaySaleCount_max':'max'})
    train_new = pd.merge(train_new, coord, on = 'Class', how='left')
    test = pd.merge(test, coord, on = 'Class', how='left')
    coord = train_wk.groupby('Class',as_index = False)['saleCount'].agg({'wkDaySaleCount_min':'min'})
    train_new = pd.merge(train_new, coord, on = 'Class', how='left')
    test = pd.merge(test, coord, on = 'Class', how='left')

    coord = train_hol.groupby('Class',as_index = False)['saleCount'].agg({'holDaySaleCount_median':'median'})
    train_new = pd.merge(train_new, coord, on = 'Class', how='left')
    test = pd.merge(test, coord, on = 'Class', how='left')
    coord = train_hol.groupby('Class',as_index = False)['saleCount'].agg({'holDaySaleCount_mean':'mean'})
    train_new = pd.merge(train_new, coord, on = 'Class', how='left')
    test = pd.merge(test, coord, on = 'Class', how='left')
    coord = train_hol.groupby('Class',as_index = False)['saleCount'].agg({'holDaySaleCount_max':'max'})
    train_new = pd.merge(train_new, coord, on = 'Class', how='left')
    test = pd.merge(test, coord, on = 'Class', how='left')
    coord = train_hol.groupby('Class',as_index = False)['saleCount'].agg({'holDaySaleCount_min':'min'})
    train_new = pd.merge(train_new, coord, on = 'Class', how='left')
    test = pd.merge(test, coord, on = 'Class', how='left')

    coord = train_hol.groupby('Class',as_index=False)['saleCount'].agg({'holSaleCount':'sum'})
    train_new = pd.merge(train_new, coord, on = 'Class', how='left')
    coord = train_wk.groupby('Class',as_index=False)['saleCount'].agg({'wkSaleCount':'sum'})
    train_new = pd.merge(train_new, coord, on = 'Class', how='left')
    train_new.loc[:,'wkHolRatio'] = train_new['wkSaleCount'] / (1.0 * (train_new['holSaleCount'] + 1))
    # train_new.loc[:,'wkDaySaleCount_cv'] = train_new['wkDaySaleCount_max'] - train_new['wkDaySaleCount_min']
    # train_new.loc[:,'holDaySaleCount_cv'] = train_new['holDaySaleCount_max'] - train_new['holDaySaleCount_min']


    coord = train_new.groupby('Class',as_index=False)['wkHolRatio'].mean()
    test = pd.merge(test, coord, on = 'Class', how='left')

    del train_new['wkSaleCount'],train_new['holSaleCount']
    del train_wk,train_hol
    return train_new,test

def get_price_feats(train,train_new, test):
    coord_class = train.groupby('Class',as_index = False)['UnitPrice'].agg({'price_mean':'mean'})
    coord_par_class = train.groupby('parClass',as_index = False)['UnitPrice'].agg({'price_mean':'mean'})
    coord_par_class.rename(columns = {'parClass':'Class'},inplace = True)
    coord = pd.concat([coord_class, coord_par_class],axis = 0)
    train_new = pd.merge(train_new, coord, on = 'Class', how = 'left')
    test = pd.merge(test, coord, on = 'Class', how = 'left')

    coord_class = train.groupby('Class',as_index = False)['UnitPrice'].agg({'price_median':'median'})
    coord_par_class = train.groupby('parClass',as_index = False)['UnitPrice'].agg({'price_median':'median'})
    coord_par_class.rename(columns = {'parClass':'Class'},inplace = True)
    coord = pd.concat([coord_class, coord_par_class],axis = 0)
    train_new = pd.merge(train_new, coord, on = 'Class', how = 'left')
    test = pd.merge(test, coord, on = 'Class', how = 'left')
    return train_new, test

# 商品促销时销量与不促销时销量的比值
def get_coupon_feats(train, train_new, test):
    coord_class_bonus_count = train[train['Coupon'] == 1].groupby('Class',as_index=False)['saleCount'].agg({'classBonusSaleCount':'count'})
    coord_parclass_bonus_count = train[train['Coupon'] == 1].groupby('parClass',as_index=False)['saleCount'].agg({'classBonusSaleCount':'count'})
    coord_parclass_bonus_count.rename(columns={'parClass':'Class'},inplace = True)
    coord = pd.concat([coord_class_bonus_count,coord_parclass_bonus_count],axis=0)
    train_new = pd.merge(train_new, coord, on = 'Class', how = 'left')
    test = pd.merge(test, coord, on = 'Class', how = 'left')
    train_new['classBonusSaleCount'] = train_new['classBonusSaleCount'].fillna(0)
    test['classBonusSaleCount'] = test['classBonusSaleCount'].fillna(0)

    coord_class_notbonus_count = train[train['Coupon'] == 0].groupby('Class',as_index=False)['saleCount'].agg({'classNotBonusSaleCount':'count'})
    coord_parclass_notbonus_count = train[train['Coupon'] == 0].groupby('parClass',as_index=False)['saleCount'].agg({'classNotBonusSaleCount':'count'})
    coord_parclass_notbonus_count.rename(columns={'parClass':'Class'},inplace = True)
    coord = pd.concat([coord_class_notbonus_count,coord_parclass_notbonus_count],axis=0)
    train_new = pd.merge(train_new, coord, on = 'Class', how = 'left')
    test = pd.merge(test, coord, on = 'Class', how = 'left')
    train_new['classNotBonusSaleCount'] = train_new['classNotBonusSaleCount'].fillna(1)
    test['classNotBonusSaleCount'] = test['classNotBonusSaleCount'].fillna(1)

    # 计算促销与非促销的比值
    train_new.loc[:,'bonusRatio'] = np.round(train_new['classBonusSaleCount'] / (1.0 * (train_new['classNotBonusSaleCount'] + 1)),4)
    del train_new['classBonusSaleCount'],train_new['classNotBonusSaleCount']
    test.loc[:,'bonusRatio'] = np.round(test['classBonusSaleCount'] / (1.0 * (test['classNotBonusSaleCount'] + 1)),4)
    del test['classBonusSaleCount'],test['classNotBonusSaleCount']
    return train_new, test


def get_coupon_hol_feats(train , train_new, test):
    # 商品节假日时促销销量与不促销销量的比值
    train_wk = train[train.holidayCluster == 1]
    train_hol = train[train.holidayCluster != 1]

    coord_class_bonus_count = train_hol[train_hol['Coupon'] == 1].groupby('Class',as_index=False)['saleCount'].agg({'classBonusSaleCount':'count'})
    coord_parclass_bonus_count = train_hol[train_hol['Coupon'] == 1].groupby('parClass',as_index=False)['saleCount'].agg({'classBonusSaleCount':'count'})
    coord_parclass_bonus_count.rename(columns={'parClass':'Class'},inplace = True)
    coord = pd.concat([coord_class_bonus_count,coord_parclass_bonus_count],axis=0)
    train_new = pd.merge(train_new, coord, on = 'Class', how = 'left')
    test = pd.merge(test, coord, on = 'Class', how = 'left')
    train_new['classBonusSaleCount'] = train_new['classBonusSaleCount'].fillna(0)
    test['classBonusSaleCount'] = test['classBonusSaleCount'].fillna(0)

    coord_class_notbonus_count = train_hol[train_hol['Coupon'] == 0].groupby('Class',as_index=False)['saleCount'].agg({'classNotBonusSaleCount':'count'})
    coord_parclass_notbonus_count = train_hol[train_hol['Coupon'] == 0].groupby('parClass',as_index=False)['saleCount'].agg({'classNotBonusSaleCount':'count'})
    coord_parclass_notbonus_count.rename(columns={'parClass':'Class'},inplace = True)
    coord = pd.concat([coord_class_notbonus_count,coord_parclass_notbonus_count],axis=0)
    train_new = pd.merge(train_new, coord, on = 'Class', how = 'left')
    test = pd.merge(test, coord, on = 'Class', how = 'left')
    train_new['classNotBonusSaleCount'] = train_new['classNotBonusSaleCount'].fillna(1)
    test['classNotBonusSaleCount'] = test['classNotBonusSaleCount'].fillna(1)

    # 计算促销与非促销的比值
    train_new.loc[:,'bonusHolRatio'] = np.round(train_new['classBonusSaleCount'] / (1.0 * (train_new['classNotBonusSaleCount'] + 1)),4)
    del train_new['classBonusSaleCount'],train_new['classNotBonusSaleCount']
    test.loc[:,'bonusHolRatio'] = np.round(test['classBonusSaleCount'] / (1.0 * (test['classNotBonusSaleCount'] + 1)),4)
    del test['classBonusSaleCount'],test['classNotBonusSaleCount']

    # 商品非节假日时促销销量与不促销销量的比值
    coord_class_bonus_count = train_wk[train_wk['Coupon'] == 1].groupby('Class',as_index=False)['saleCount'].agg({'classBonusSaleCount':'count'})
    coord_parclass_bonus_count = train_wk[train_wk['Coupon'] == 1].groupby('parClass',as_index=False)['saleCount'].agg({'classBonusSaleCount':'count'})
    coord_parclass_bonus_count.rename(columns={'parClass':'Class'},inplace = True)
    coord = pd.concat([coord_class_bonus_count,coord_parclass_bonus_count],axis=0)
    train_new = pd.merge(train_new, coord, on = 'Class', how = 'left')
    test = pd.merge(test, coord, on = 'Class', how = 'left')
    train_new['classBonusSaleCount'] = train_new['classBonusSaleCount'].fillna(0)
    test['classBonusSaleCount'] = test['classBonusSaleCount'].fillna(0)

    coord_class_notbonus_count = train_wk[train_wk['Coupon'] == 0].groupby('Class',as_index=False)['saleCount'].agg({'classNotBonusSaleCount':'count'})
    coord_parclass_notbonus_count = train_wk[train_wk['Coupon'] == 0].groupby('parClass',as_index=False)['saleCount'].agg({'classNotBonusSaleCount':'count'})
    coord_parclass_notbonus_count.rename(columns={'parClass':'Class'},inplace = True)
    coord = pd.concat([coord_class_notbonus_count,coord_parclass_notbonus_count],axis=0)
    train_new = pd.merge(train_new, coord, on = 'Class', how = 'left')
    test = pd.merge(test, coord, on = 'Class', how = 'left')
    train_new['classNotBonusSaleCount'] = train_new['classNotBonusSaleCount'].fillna(1)
    test['classNotBonusSaleCount'] = test['classNotBonusSaleCount'].fillna(1)

    # 计算促销与非促销的比值
    train_new.loc[:,'bonusNotHolRatio'] = np.round(train_new['classBonusSaleCount'] / (1.0 * (train_new['classNotBonusSaleCount'] + 1)),4)
    del train_new['classBonusSaleCount'],train_new['classNotBonusSaleCount']
    test.loc[:,'bonusNotHolRatio'] = np.round(test['classBonusSaleCount'] / (1.0 * (test['classNotBonusSaleCount'] + 1)),4)
    del test['classBonusSaleCount'],test['classNotBonusSaleCount']
    del train_wk,train_hol

    return train_new, test

# 商品周几促销的比例
def get_coupon_weekday_feats(train, train_new, test):
    train_coupon = train[train.Coupon == 1]
    coord = train_coupon.groupby(['Class','dayOfWeek'],as_index=False)['dayOfWeek'].agg({'dayOfWeekCount':'count'})
    var = train_coupon.groupby(['Class'],as_index=False)['dayOfWeek'].agg({'classCouponCount':'count'})
    coord = pd.merge(coord, var, on = 'Class',how='left' )
    coord.loc[:,'bonusWeekProb'] = coord['dayOfWeekCount'] / np.round((1.0 * (coord['classCouponCount'] + 1)))
    coord_c = coord.copy()

    coord = train_coupon.groupby(['parClass','dayOfWeek'],as_index=False)['dayOfWeek'].agg({'dayOfWeekCount':'count'})
    var = train_coupon.groupby(['parClass'],as_index=False)['dayOfWeek'].agg({'classCouponCount':'count'})
    coord = pd.merge(coord, var, on = 'parClass',how='left' )
    coord.loc[:,'bonusWeekProb'] = coord['dayOfWeekCount'] / np.round((1.0 * (coord['classCouponCount'] + 1)))
    coord.rename(columns={'parClass':'Class'},inplace=True)
    coord_pc = coord.copy()

    coord = pd.concat([coord_c,coord_pc],axis=0)

    train_new = pd.merge(train_new, coord[['Class','dayOfWeek','bonusWeekProb']], on = ['Class','dayOfWeek'],how='left')
    train_new['bonusWeekProb'] = train_new['bonusWeekProb'].fillna(0)

    test = pd.merge(test, coord[['Class','dayOfWeek','bonusWeekProb']], on = ['Class','dayOfWeek'],how='left')
    test['bonusWeekProb'] = test['bonusWeekProb'].fillna(0)
    return train_new, test

# def get_fb_feats(train_new,test):
#     train_fb = pd.read_csv(train_fb_path)
#     train_fb['SaleDate'] = pd.to_datetime(train_fb['SaleDate'])
#     train_fb.fillna(0,inplace = True)
#     train_new = pd.merge(train_new,train_fb,on=['Class','SaleDate'],how='left')
#     test = pd.merge(test,train_fb,on=['Class','SaleDate'],how='left')
#     return train_new,test

# 提取商品固有特征
def get_origin_feats(train, train_new, test):
    print "Start extract commodity orginal features:......."
    train, train_new, test = get_hol_feats(train, train_new, test)
    print "Holiday features done."
    train, train_new, test = get_time_feats(train, train_new, test)
    print "Time features done."
    train, train_new, test = get_commodity_class(train, train_new, test)
    print "Commodity class features done."
    train,train_new,test = get_commodity_hot_index(train,train_new,test)
    print "Commodity hot index features done."
    train_new,test = get_hol_sale_feats(train_new,test)
    print "Commodity holiday features done."
    train_new,test = get_weekday_ratio_feats(train_new,test)

    # train_new,test = get_fb_feats(train_new,test)

    # train_new, test = get_price_feats(train,train_new, test)
    # print "Commodity price features done."
    # train_new, test = get_coupon_feats(train, train_new, test)
    # print "Coupon features done."
    # train_new, test = get_coupon_hol_feats(train , train_new, test)
    # print "Coupon holiday features done."
    # train_new, test = get_coupon_weekday_feats(train, train_new, test)
    print "Coupon weekday features done."
    print "Commodity original features done."
    return train, train_new, test

def test_split(train_new, test, week):
    test.loc[:,'saleCount'] = 0
    # Step is one week
    test_1 = test[test['SaleDate'].isin(week[0])]   #第一周
    test_2 = test[test['SaleDate'].isin(week[1])]
    test_3 = test[test['SaleDate'].isin(week[2])]
    test_4 = test[test['SaleDate'].isin(week[3])]
    test_5 = test[test['SaleDate'].isin(week[4])]
    test_1.to_csv(output_path + 'week1.csv',index = False)
    test_2.to_csv(output_path + 'week2.csv',index = False)
    test_3.to_csv(output_path + 'week3.csv',index = False)
    test_4.to_csv(output_path + 'week4.csv',index = False)
    test_5.to_csv(output_path + 'week5.csv',index = False)
    test.to_csv(output_path + 'week.csv',index = False)
    print  np.setdiff1d(test.columns,train_new.columns)
    print  np.setdiff1d(train_new.columns, test.columns)
    return test_1, test_2, test_3, test_4, test_5,test

def valid_split(train_new, week,month):
    test_valid = train_new[['Class','SaleDate','saleCount']]
    test_valid = test_valid[test_valid['SaleDate'].isin(month)]

    # test_valid_1 = test_valid[test_valid['SaleDate'].isin(week[0])]
    # test_valid_2 = test_valid[test_valid['SaleDate'].isin(week[1])]
    # test_valid_3 = test_valid[test_valid['SaleDate'].isin(week[2])]
    # test_valid_4 = test_valid[test_valid['SaleDate'].isin(week[3])]
    # test_valid_5 = test_valid[test_valid['SaleDate'].isin(week[4])]
    # test_valid_1.to_csv(output_path + 'test_valid_week1.csv',index = False)
    # test_valid_2.to_csv(output_path + 'test_valid_week2.csv',index = False)
    # test_valid_3.to_csv(output_path + 'test_valid_week3.csv',index = False)
    # test_valid_4.to_csv(output_path + 'test_valid_week4.csv',index = False)
    # test_valid_5.to_csv(output_path + 'test_valid_week5.csv',index = False)
    # test_valid.to_csv(output_path + 'test_valid_Apri.csv',index = False)
    # print  np.setdiff1d(test.columns,train_new.columns)
    # return test_valid_1,test_valid_2,test_valid_3,test_valid_4,test_valid_5,test_valid
    return test_valid


def merge_train_test(train, test):
    train_test = pd.concat([train, test], axis = 0) # 第一次用train_new, test_1
    return train_test

def get_roll_hot_index_feats(train_test):
    # 类别上周，上上周，上个月总销量
    lastWeekSaleCount = train_test.groupby(['Class','weekOfYear'],as_index=False)['saleCount'].agg({'saleCount':'sum'})
    lastMonthSaleCount = train_test.groupby(['Class','month'],as_index=False)['saleCount'].agg({'lastMonthSaleCount':'sum'})
    lastWeekSaleCount['lastWeekSaleCount'] = lastWeekSaleCount.groupby('Class')['saleCount'].shift(1)
    lastWeekSaleCount['lastWeekSaleCount'].fillna(method='bfill',inplace=True)
    lastWeekSaleCount['last2WeekSaleCount'] = lastWeekSaleCount.groupby('Class')['lastWeekSaleCount'].shift(1)
    lastWeekSaleCount['last2WeekSaleCount'].fillna(method='bfill',inplace=True)
    lastWeekSaleCount['last3WeekSaleCount'] = lastWeekSaleCount.groupby('Class')['last2WeekSaleCount'].shift(1)
    lastWeekSaleCount['last3WeekSaleCount'].fillna(method='bfill',inplace=True)
    lastMonthSaleCount['lastMonthSaleCount'] = lastMonthSaleCount.groupby('Class')['lastMonthSaleCount'].shift(1)
    lastMonthSaleCount['lastMonthSaleCount'].fillna(method='bfill',inplace=True)
    del lastWeekSaleCount['saleCount']
    # del lastMonthSaleCount['lastMonthSaleCount']

    lastWeekTotSaleCount = train_test.groupby(['weekOfYear'],as_index=False)['saleCount'].agg({'saleCount':'sum'})
    lastMonthTotSaleCount = train_test.groupby(['month'],as_index=False)['saleCount'].agg({'monthTotSaleCount':'sum'})
    lastWeekTotSaleCount['lastWeekTotSaleCount'] = lastWeekTotSaleCount['saleCount'].shift(1)
    lastWeekTotSaleCount['lastWeekTotSaleCount'].fillna(method='bfill',inplace=True)
    lastWeekTotSaleCount['last2WeekTotSaleCount'] = lastWeekTotSaleCount['lastWeekTotSaleCount'].shift(1)
    lastWeekTotSaleCount['last2WeekTotSaleCount'].fillna(method='bfill',inplace=True)
    lastWeekTotSaleCount['last3WeekTotSaleCount'] = lastWeekTotSaleCount['last2WeekTotSaleCount'].shift(1)
    lastWeekTotSaleCount['last3WeekTotSaleCount'].fillna(method='bfill',inplace=True)
    lastMonthTotSaleCount['lastMonthTotSaleCount'] = lastMonthTotSaleCount['monthTotSaleCount'].shift(1)
    lastMonthTotSaleCount['lastMonthTotSaleCount'].fillna(method='bfill',inplace=True)
    del lastWeekTotSaleCount['saleCount']
    del lastMonthTotSaleCount['monthTotSaleCount']

    tmp = pd.merge(lastWeekSaleCount,lastWeekTotSaleCount, on = 'weekOfYear',how = 'left')
    lastWeekSaleCount = tmp.copy()
    tmp = pd.merge(lastMonthSaleCount,lastMonthTotSaleCount, on = 'month',how = 'left')
    lastMonthSaleCount = tmp.copy()

    # 用于合并
    lastWeekSaleCount.loc[:,'hotPast1WeekIndex'] = np.round(lastWeekSaleCount.lastWeekSaleCount / (1.0 * lastWeekSaleCount.lastWeekTotSaleCount + 1),4)
    lastWeekSaleCount.loc[:,'hotPast2WeekIndex'] = np.round(lastWeekSaleCount.last2WeekSaleCount / (1.0 * lastWeekSaleCount.last2WeekTotSaleCount + 1),4)
    lastWeekSaleCount.loc[:,'hotPast3WeekIndex'] = np.round(lastWeekSaleCount.last3WeekSaleCount / (1.0 * lastWeekSaleCount.last3WeekTotSaleCount +1),4)
    lastMonthSaleCount.loc[:,'hotPast1MonthIndex'] = np.round(lastMonthSaleCount.lastMonthSaleCount / (1.0 * lastMonthSaleCount.lastMonthTotSaleCount +1),4)


    # 父类
    # 父类别上周，上上周，上个月总销量
    # 类别上周，上上周，上个月总销量
    parLastWeekSaleCount = train_test.groupby(['parClass','weekOfYear'],as_index=False)['saleCount'].agg({'saleCount':'sum'})
    parLastMonthSaleCount = train_test.groupby(['parClass','month'],as_index=False)['saleCount'].agg({'parLastMonthSaleCount':'sum'})
    parLastWeekSaleCount['parLastWeekSaleCount'] = parLastWeekSaleCount.groupby('parClass')['saleCount'].shift(1)
    parLastWeekSaleCount['parLastWeekSaleCount'].fillna(method='bfill',inplace=True)
    parLastWeekSaleCount['parLast2WeekSaleCount'] = parLastWeekSaleCount.groupby('parClass')['parLastWeekSaleCount'].shift(1)
    parLastWeekSaleCount['parLast2WeekSaleCount'].fillna(method='bfill',inplace=True)
    parLastWeekSaleCount['parLast3WeekSaleCount'] = parLastWeekSaleCount.groupby('parClass')['parLast2WeekSaleCount'].shift(1)
    parLastWeekSaleCount['parLast3WeekSaleCount'].fillna(method='bfill',inplace=True)
    parLastMonthSaleCount['parLastMonthSaleCount'] = parLastMonthSaleCount.groupby('parClass')['parLastMonthSaleCount'].shift(1)
    parLastMonthSaleCount['parLastMonthSaleCount'].fillna(method='bfill',inplace=True)
    del parLastWeekSaleCount['saleCount']
    # del parLastMonthSaleCount['parLastMonthSaleCount']

    parLastWeekTotSaleCount = train_test.groupby(['weekOfYear'],as_index=False)['saleCount'].agg({'saleCount':'sum'})
    parLastMonthTotSaleCount = train_test.groupby(['month'],as_index=False)['saleCount'].agg({'monthTotSaleCount':'sum'})
    parLastWeekTotSaleCount['parLastWeekTotSaleCount'] = parLastWeekTotSaleCount['saleCount'].shift(1)
    parLastWeekTotSaleCount['parLastWeekTotSaleCount'].fillna(method='bfill',inplace=True)
    parLastWeekTotSaleCount['parLast2WeekTotSaleCount'] = parLastWeekTotSaleCount['parLastWeekTotSaleCount'].shift(1)
    parLastWeekTotSaleCount['parLast2WeekTotSaleCount'].fillna(method='bfill',inplace=True)
    parLastWeekTotSaleCount['parLast3WeekTotSaleCount'] = parLastWeekTotSaleCount['parLast2WeekTotSaleCount'].shift(1)
    parLastWeekTotSaleCount['parLast3WeekTotSaleCount'].fillna(method='bfill',inplace=True)
    parLastMonthTotSaleCount['parLastMonthTotSaleCount'] = parLastMonthTotSaleCount['monthTotSaleCount'].shift(1)
    parLastMonthTotSaleCount['parLastMonthTotSaleCount'].fillna(method='bfill',inplace=True)
    del parLastWeekTotSaleCount['saleCount']
    del parLastMonthTotSaleCount['monthTotSaleCount']

    tmp = pd.merge(parLastWeekSaleCount,parLastWeekTotSaleCount, on = 'weekOfYear',how = 'left')
    parLastWeekSaleCount = tmp.copy()
    tmp = pd.merge(parLastMonthSaleCount,parLastMonthTotSaleCount, on = 'month',how = 'left')
    parLastMonthSaleCount = tmp.copy()

    # 用于合并
    lastWeekSaleCount.loc[:,'hotPast1WeekIndex'] = np.round(lastWeekSaleCount['lastWeekSaleCount']/ (1.0 * lastWeekSaleCount['lastWeekTotSaleCount'] + 1),4)
    lastWeekSaleCount.loc[:,'hotPast2WeekIndex'] = np.round(lastWeekSaleCount['last2WeekSaleCount'] / (1.0 * lastWeekSaleCount['last2WeekTotSaleCount'] + 1),4)
    lastWeekSaleCount.loc[:,'hotPast3WeekIndex'] = np.round(lastWeekSaleCount['last3WeekSaleCount'] / (1.0 * lastWeekSaleCount['last3WeekTotSaleCount'] + 1),4)
    lastMonthSaleCount.loc[:,'hotPast1MonthIndex'] = np.round(lastMonthSaleCount['lastMonthSaleCount'] / (1.0 * lastMonthSaleCount['lastMonthTotSaleCount'] +1),4)

    # 用于合并
    parLastWeekSaleCount.loc [:,'parHotPast1WeekIndex']  = np.round(parLastWeekSaleCount['parLastWeekSaleCount'] /   (1.0 * parLastWeekSaleCount['parLastWeekTotSaleCount'] + 1),4)
    parLastWeekSaleCount.loc[:,'parHotPast2WeekIndex']  = np.round(parLastWeekSaleCount['parLast2WeekSaleCount'] / (1.0 * parLastWeekSaleCount['parLast2WeekTotSaleCount'] + 1),4)
    parLastWeekSaleCount.loc[:,'parHotPast3WeekIndex']  = np.round(parLastWeekSaleCount['parLast3WeekSaleCount'] / (1.0 * parLastWeekSaleCount['parLast3WeekTotSaleCount'] + 1),4)
    parLastMonthSaleCount.loc[:,'parHotPast1MonthIndex'] = np.round(parLastMonthSaleCount['parLastMonthSaleCount'] / (1.0 * parLastMonthSaleCount['parLastMonthTotSaleCount'] + 1),4)

    # 合并 train_test
    del lastWeekSaleCount['lastWeekSaleCount' ],   lastWeekSaleCount['lastWeekTotSaleCount']
    del lastWeekSaleCount['last2WeekSaleCount'],lastWeekSaleCount['last2WeekTotSaleCount']
    del lastWeekSaleCount['last3WeekSaleCount'],lastWeekSaleCount['last3WeekTotSaleCount']
    del lastMonthSaleCount['lastMonthSaleCount'],lastMonthSaleCount['lastMonthTotSaleCount']
    del parLastWeekSaleCount ['parLastWeekSaleCount' ],  parLastWeekSaleCount['parLastWeekTotSaleCount']
    del parLastWeekSaleCount['parLast2WeekSaleCount'],parLastWeekSaleCount['parLast2WeekTotSaleCount']
    del parLastWeekSaleCount['parLast3WeekSaleCount'],parLastWeekSaleCount['parLast3WeekTotSaleCount']
    del parLastMonthSaleCount['parLastMonthSaleCount'],parLastMonthSaleCount['parLastMonthTotSaleCount']

    tmp = pd.merge(train_test,lastWeekSaleCount,on=['Class','weekOfYear'],how='left')
    tmp = pd.merge(tmp,lastMonthSaleCount,on=['Class','month'],how='left')
    tmp = pd.merge(tmp,parLastWeekSaleCount,on=['parClass','weekOfYear'],how='left')
    tmp = pd.merge(tmp,parLastMonthSaleCount,on=['parClass','month'],how='left')
    # print 'new added features:',np.setdiff1d(tmp.columns, train_test.columns)
    train_test = tmp.copy()

    return train_test

def get_roll_price_feats(train_test):
    # 类别上周，上2周，上3周，上个月销量统计量 - 总量
    lastWeeksSaleCount_sum = train_test.groupby(['Class','weekOfYear'],as_index=False)['saleCount'].agg({'WeekSaleCount_sum':'sum'})
    lastWeeksSaleCount_sum['lastWeekSaleCount_sum']   = lastWeeksSaleCount_sum.groupby('Class')['WeekSaleCount_sum'].shift(1)
    lastWeeksSaleCount_sum['lastWeekSaleCount_sum'].fillna(method='bfill',inplace=True)
    lastWeeksSaleCount_sum['last2WeekSaleCount_sum']   = lastWeeksSaleCount_sum.groupby('Class')['lastWeekSaleCount_sum'].shift(1)
    lastWeeksSaleCount_sum['last2WeekSaleCount_sum'].fillna(method='bfill',inplace=True)
    lastWeeksSaleCount_sum['last3WeekSaleCount_sum']   = lastWeeksSaleCount_sum.groupby('Class')['last2WeekSaleCount_sum'].shift(1)
    lastWeeksSaleCount_sum['last3WeekSaleCount_sum'].fillna(method='bfill',inplace=True)
    del lastWeeksSaleCount_sum['WeekSaleCount_sum']
    lastWeeksSaleCount_sum.loc[:,'last1WeekTo2WeekRate_sum'] = lastWeeksSaleCount_sum['lastWeekSaleCount_sum'] / (1.0* (lastWeeksSaleCount_sum['last2WeekSaleCount_sum'] + 0.1))
    lastWeeksSaleCount_sum.loc[:,'last1WeekTo3WeekRate_sum'] = lastWeeksSaleCount_sum['lastWeekSaleCount_sum'] / (1.0* (lastWeeksSaleCount_sum['last3WeekSaleCount_sum'] + 0.1))
    lastWeeksSaleCount_sum.loc[:,'last2WeekTo3WeekRate_sum'] = lastWeeksSaleCount_sum['last2WeekSaleCount_sum'] / (1.0* (lastWeeksSaleCount_sum['last3WeekSaleCount_sum'] + 0.1))
    lastWeeksSaleCount_sum.loc[:,'last1WeekTo3WeekTotRate_sum'] = lastWeeksSaleCount_sum['lastWeekSaleCount_sum'] / (1.0* (lastWeeksSaleCount_sum['lastWeekSaleCount_sum'] + lastWeeksSaleCount_sum['last2WeekSaleCount_sum'] + lastWeeksSaleCount_sum['last3WeekSaleCount_sum'] + 0.1))
    lastMonthSaleCount_sum = train_test.groupby(['Class','month'],as_index=False)['saleCount'].agg({'monthSaleCount_sum':'sum'})
    lastMonthSaleCount_sum['lastMonthSaleCount_sum'] = lastMonthSaleCount_sum.groupby('Class')['monthSaleCount_sum'].shift(1)
    lastMonthSaleCount_sum['lastMonthSaleCount_sum'].fillna(method='bfill',inplace=True)
    del lastMonthSaleCount_sum['monthSaleCount_sum']


    # 类别上周，上2周，上3周，上个月销量统计量 - 均值
    lastWeeksSaleCount_mean = train_test.groupby(['Class','weekOfYear'],as_index=False)['saleCount'].agg({'WeekSaleCount_mean':'mean'})
    lastWeeksSaleCount_mean['lastWeekSaleCount_mean']   = lastWeeksSaleCount_mean.groupby('Class')['WeekSaleCount_mean'].shift(1)
    lastWeeksSaleCount_mean['lastWeekSaleCount_mean'].fillna(method='bfill',inplace=True)
    lastWeeksSaleCount_mean['last2WeekSaleCount_mean']   = lastWeeksSaleCount_mean.groupby('Class')['lastWeekSaleCount_mean'].shift(1)
    lastWeeksSaleCount_mean['last2WeekSaleCount_mean'].fillna(method='bfill',inplace=True)
    lastWeeksSaleCount_mean['last3WeekSaleCount_mean']   = lastWeeksSaleCount_mean.groupby('Class')['last2WeekSaleCount_mean'].shift(1)
    lastWeeksSaleCount_mean['last3WeekSaleCount_mean'].fillna(method='bfill',inplace=True)
    del lastWeeksSaleCount_mean['WeekSaleCount_mean']
    lastMonthSaleCount_mean = train_test.groupby(['Class','month'],as_index=False)['saleCount'].agg({'monthSaleCount_mean':'mean'})
    lastMonthSaleCount_mean['lastMonthSaleCount_mean'] = lastMonthSaleCount_mean.groupby('Class')['monthSaleCount_mean'].shift(1)
    lastMonthSaleCount_mean['lastMonthSaleCount_mean'].fillna(method='bfill',inplace=True)
    del lastMonthSaleCount_mean['monthSaleCount_mean']

    # 类别上周，上2周，上3周，上个月销量统计量 - 最大值
    lastWeeksSaleCount_max = train_test.groupby(['Class','weekOfYear'],as_index=False)['saleCount'].agg({'WeekSaleCount_max':'max'})
    lastWeeksSaleCount_max['lastWeekSaleCount_max']   = lastWeeksSaleCount_max.groupby('Class')['WeekSaleCount_max'].shift(1)
    lastWeeksSaleCount_max['lastWeekSaleCount_max'].fillna(method='bfill',inplace=True)
    lastWeeksSaleCount_max['last2WeekSaleCount_max']   = lastWeeksSaleCount_max.groupby('Class')['lastWeekSaleCount_max'].shift(1)
    lastWeeksSaleCount_max['last2WeekSaleCount_max'].fillna(method='bfill',inplace=True)
    lastWeeksSaleCount_max['last3WeekSaleCount_max']   = lastWeeksSaleCount_max.groupby('Class')['last2WeekSaleCount_max'].shift(1)
    lastWeeksSaleCount_max['last3WeekSaleCount_max'].fillna(method='bfill',inplace=True)
    del lastWeeksSaleCount_max['WeekSaleCount_max']
    lastMonthSaleCount_max = train_test.groupby(['Class','month'],as_index=False)['saleCount'].agg({'monthSaleCount_max':'max'})
    lastMonthSaleCount_max['lastMonthSaleCount_max'] = lastMonthSaleCount_max.groupby('Class')['monthSaleCount_max'].shift(1)
    lastMonthSaleCount_max['lastMonthSaleCount_max'].fillna(method='bfill',inplace=True)
    del lastMonthSaleCount_max['monthSaleCount_max']

    # 类别上周，上2周，上3周，上个月销量统计量 - 标准差
    lastWeeksSaleCount_std = train_test.groupby(['Class','weekOfYear'],as_index=False)['saleCount'].agg({'WeekSaleCount_std':'std'})
    lastWeeksSaleCount_std['lastWeekSaleCount_std']   = lastWeeksSaleCount_std.groupby('Class')['WeekSaleCount_std'].shift(1)
    lastWeeksSaleCount_std['lastWeekSaleCount_std'].fillna(method='bfill',inplace=True)
    lastWeeksSaleCount_std['last2WeekSaleCount_std']   = lastWeeksSaleCount_std.groupby('Class')['lastWeekSaleCount_std'].shift(1)
    lastWeeksSaleCount_std['last2WeekSaleCount_std'].fillna(method='bfill',inplace=True)
    lastWeeksSaleCount_std['last3WeekSaleCount_std']   = lastWeeksSaleCount_std.groupby('Class')['last2WeekSaleCount_std'].shift(1)
    lastWeeksSaleCount_std['last3WeekSaleCount_std'].fillna(method='bfill',inplace=True)
    del lastWeeksSaleCount_std['WeekSaleCount_std']
    lastMonthSaleCount_std = train_test.groupby(['Class','month'],as_index=False)['saleCount'].agg({'monthSaleCount_std':'std'})
    lastMonthSaleCount_std['lastMonthSaleCount_std'] = lastMonthSaleCount_std.groupby('Class')['monthSaleCount_std'].shift(1)
    lastMonthSaleCount_std['lastMonthSaleCount_std'].fillna(method='bfill',inplace=True)
    del lastMonthSaleCount_std['monthSaleCount_std']

    parLastWeeksSaleCount_mean = train_test.groupby(['parClass','weekOfYear'],as_index=False)['saleCount'].agg({'WeekSaleCount_mean':'mean'})
    parLastWeeksSaleCount_mean['parLastWeekSaleCount_mean']   = parLastWeeksSaleCount_mean.groupby('parClass')['WeekSaleCount_mean'].shift(1)
    parLastWeeksSaleCount_mean['parLastWeekSaleCount_mean'].fillna(method='bfill',inplace=True)
    parLastWeeksSaleCount_mean['parLast2WeekSaleCount_mean']   = parLastWeeksSaleCount_mean.groupby('parClass')['parLastWeekSaleCount_mean'].shift(1)
    parLastWeeksSaleCount_mean['parLast2WeekSaleCount_mean'].fillna(method='bfill',inplace=True)
    parLastWeeksSaleCount_mean['parLast3WeekSaleCount_mean']   = parLastWeeksSaleCount_mean.groupby('parClass')['parLast2WeekSaleCount_mean'].shift(1)
    parLastWeeksSaleCount_mean['parLast3WeekSaleCount_mean'].fillna(method='bfill',inplace=True)
    parLastWeeksSaleCount_mean['parLast3WeekSaleCount_mean']   = parLastWeeksSaleCount_mean.groupby('parClass')['parLast2WeekSaleCount_mean'].shift(1)
    parLastWeeksSaleCount_mean['parLast3WeekSaleCount_mean'].fillna(method='bfill',inplace=True)
    del parLastWeeksSaleCount_mean['WeekSaleCount_mean']
    parLastMonthSaleCount_mean = train_test.groupby(['parClass','month'],as_index=False)['saleCount'].agg({'monthSaleCount_mean':'mean'})
    parLastMonthSaleCount_mean['parLastMonthSaleCount_mean'] = parLastMonthSaleCount_mean.groupby('parClass')['monthSaleCount_mean'].shift(1)
    parLastMonthSaleCount_mean['parLastMonthSaleCount_mean'].fillna(method='bfill',inplace=True)
    del parLastMonthSaleCount_mean['monthSaleCount_mean']

    # 类别上周，上2周，上3周，上个月销量统计量 - 最大值
    parLastWeeksSaleCount_max = train_test.groupby(['parClass','weekOfYear'],as_index=False)['saleCount'].agg({'WeekSaleCount_max':'max'})
    parLastWeeksSaleCount_max['parLastWeekSaleCount_max']   = parLastWeeksSaleCount_max.groupby('parClass')['WeekSaleCount_max'].shift(1)
    parLastWeeksSaleCount_max['parLastWeekSaleCount_max'].fillna(method='bfill',inplace=True)
    parLastWeeksSaleCount_max['parLast2WeekSaleCount_max']   = parLastWeeksSaleCount_max.groupby('parClass')['parLastWeekSaleCount_max'].shift(1)
    parLastWeeksSaleCount_max['parLast2WeekSaleCount_max'].fillna(method='bfill',inplace=True)
    parLastWeeksSaleCount_max['parLast3WeekSaleCount_max']   = parLastWeeksSaleCount_max.groupby('parClass')['parLast2WeekSaleCount_max'].shift(1)
    parLastWeeksSaleCount_max['parLast3WeekSaleCount_max'].fillna(method='bfill',inplace=True)
    parLastWeeksSaleCount_max['parLast3WeekSaleCount_max']   = parLastWeeksSaleCount_max.groupby('parClass')['parLast2WeekSaleCount_max'].shift(1)
    parLastWeeksSaleCount_max['parLast3WeekSaleCount_max'].fillna(method='bfill',inplace=True)
    del parLastWeeksSaleCount_max['WeekSaleCount_max']
    parLastMonthSaleCount_max = train_test.groupby(['parClass','month'],as_index=False)['saleCount'].agg({'monthSaleCount_max':'max'})
    parLastMonthSaleCount_max['parLastMonthSaleCount_max'] = parLastMonthSaleCount_max.groupby('parClass')['monthSaleCount_max'].shift(1)
    parLastMonthSaleCount_max['parLastMonthSaleCount_max'].fillna(method='bfill',inplace=True)
    del parLastMonthSaleCount_max['monthSaleCount_max']

    # 父类别上周，上2周，上3周，上个月销量统计量 - 标准差
    parLastWeeksSaleCount_std = train_test.groupby(['parClass','weekOfYear'],as_index=False)['saleCount'].agg({'WeekSaleCount_std':'std'})
    parLastWeeksSaleCount_std['parLastWeekSaleCount_std']   = parLastWeeksSaleCount_std.groupby('parClass')['WeekSaleCount_std'].shift(1)
    parLastWeeksSaleCount_std['parLastWeekSaleCount_std'].fillna(method='bfill',inplace=True)
    parLastWeeksSaleCount_std['parLast2WeekSaleCount_std']   = parLastWeeksSaleCount_std.groupby('parClass')['parLastWeekSaleCount_std'].shift(1)
    parLastWeeksSaleCount_std['parLast2WeekSaleCount_std'].fillna(method='bfill',inplace=True)
    parLastWeeksSaleCount_std['parLast3WeekSaleCount_std']   = parLastWeeksSaleCount_std.groupby('parClass')['parLast2WeekSaleCount_std'].shift(1)
    parLastWeeksSaleCount_std['parLast3WeekSaleCount_std'].fillna(method='bfill',inplace=True)
    parLastWeeksSaleCount_std['parLast3WeekSaleCount_std']   = parLastWeeksSaleCount_std.groupby('parClass')['parLast2WeekSaleCount_std'].shift(1)
    parLastWeeksSaleCount_std['parLast3WeekSaleCount_std'].fillna(method='bfill',inplace=True)
    del parLastWeeksSaleCount_std['WeekSaleCount_std']
    parLastMonthSaleCount_std = train_test.groupby(['parClass','month'],as_index=False)['saleCount'].agg({'monthSaleCount_std':'std'})
    parLastMonthSaleCount_std['parLastMonthSaleCount_std'] = parLastMonthSaleCount_std.groupby('parClass')['monthSaleCount_std'].shift(1)
    parLastMonthSaleCount_std['parLastMonthSaleCount_std'].fillna(method='bfill',inplace=True)
    del parLastMonthSaleCount_std['monthSaleCount_std']

    # 有毒 父类别各种统计值
    # # 父类别上周，上2周，上3周，上个月销量统计量 - 均值

    # 合并 train_test
    tmp = pd.merge(train_test,lastWeeksSaleCount_mean,on=['Class','weekOfYear'],how='left')
    tmp = pd.merge(tmp,lastWeeksSaleCount_sum,on=['Class','weekOfYear'],how='left')
    tmp = pd.merge(tmp,lastMonthSaleCount_mean,on=['Class','month'],how='left')
    tmp = pd.merge(tmp,lastWeeksSaleCount_std,on=['Class','weekOfYear'],how='left')
    tmp = pd.merge(tmp,lastMonthSaleCount_std,on=['Class','month'],how='left')
    tmp = pd.merge(tmp,lastWeeksSaleCount_max,on=['Class','weekOfYear'],how='left')
    tmp = pd.merge(tmp,lastMonthSaleCount_max,on=['Class','month'],how='left')
    train_test = tmp.copy()

    tmp = pd.merge(train_test,parLastWeeksSaleCount_mean,on=['parClass','weekOfYear'],how='left')
    tmp = pd.merge(tmp,parLastMonthSaleCount_mean,on=['parClass','month'],how='left')
    tmp = pd.merge(tmp,parLastWeeksSaleCount_std,on=['parClass','weekOfYear'],how='left')
    tmp = pd.merge(tmp,parLastMonthSaleCount_std,on=['parClass','month'],how='left')
    tmp = pd.merge(tmp,parLastWeeksSaleCount_max,on=['parClass','weekOfYear'],how='left')
    tmp = pd.merge(tmp,parLastMonthSaleCount_max,on=['parClass','month'],how='left')
    train_test = tmp.copy()


    # print 'new added features:',np.setdiff1d(tmp.columns, train_test.columns)

    return train_test

def get_roll_week_sale_feats(train_test):
    #子类
    # weekDayRatio
    lastMonthTotSaleCount = train_test.groupby(['Class','month'],as_index=False)['saleCount'].agg({'lastMonthTotSaleCount':'sum'})
    lastMonthTotSaleCount['lastMonthTotSaleCount'] = lastMonthTotSaleCount.groupby('Class')['lastMonthTotSaleCount'].shift(1)
    lastMonthTotSaleCount['lastMonthTotSaleCount'].fillna(method='bfill',inplace=True)

    lastWeekDayTotSaleCount = train_test.groupby(['Class','month','dayOfWeek'],as_index=False)['saleCount'].agg({'lastWeekDayTotSaleCount':'sum'})
    coord = lastWeekDayTotSaleCount.groupby('Class',as_index=False)['lastWeekDayTotSaleCount'].shift(7)
    coord.fillna(method='bfill',inplace=True)
    lastWeekDayTotSaleCount['lastWeekDayTotSaleCount'] = coord
    lastWeekDayTotSaleCount = pd.merge(lastWeekDayTotSaleCount,lastMonthTotSaleCount,on=['Class','month'],how='left')
    lastWeekDayTotSaleCount.loc[:,'weekDayRatio'] = np.round(lastWeekDayTotSaleCount['lastWeekDayTotSaleCount'] / (1.0 * lastWeekDayTotSaleCount['lastMonthTotSaleCount'] + 1) ,4)


    # # weekOn1WeekRatio，weekOn2WeekRatio
    # last1WeekSaleCount_o = train_test.groupby(['Class','dayOfYear'],as_index=False)['saleCount'].agg({'last1WeekSaleCount':'sum'})
    # last1WeekSaleCount = last1WeekSaleCount_o.shift(7)
    # last1WeekSaleCount['Class'] = last1WeekSaleCount_o['Class']
    # last1WeekSaleCount['dayOfYear'] = last1WeekSaleCount_o['dayOfYear']
    # last1WeekSaleCount['last1WeekSaleCount'].fillna(0,inplace=True)
    #
    # last2WeekSaleCount = last1WeekSaleCount_o.shift(14)
    # last2WeekSaleCount.rename(columns={'last1WeekSaleCount':'last2WeekSaleCount'},inplace=True)
    # last2WeekSaleCount['Class'] = last1WeekSaleCount_o['Class']
    # last2WeekSaleCount['dayOfYear'] = last1WeekSaleCount_o['dayOfYear']
    # last2WeekSaleCount['last2WeekSaleCount'].fillna(1,inplace=True)   #把分母设为1
    # last2WeekSaleCount['last2WeekSaleCount'][last2WeekSaleCount['last2WeekSaleCount'] == 0.0] = 1  #把分母设为1
    #
    # last3WeekSaleCount = last1WeekSaleCount_o.shift(21)
    # last3WeekSaleCount.rename(columns={'last1WeekSaleCount':'last3WeekSaleCount'},inplace=True)
    # last3WeekSaleCount['Class'] = last1WeekSaleCount_o['Class']
    # last3WeekSaleCount['dayOfYear'] = last1WeekSaleCount_o['dayOfYear']
    # last3WeekSaleCount['last3WeekSaleCount'].fillna(0,inplace=True)   #把分母设为1
    # last3WeekSaleCount['last3WeekSaleCount'][last3WeekSaleCount['last3WeekSaleCount'] == 0.0] = 1  #把分母设为1
    #
    #
    # weekOnWeekRatio = pd.merge(last1WeekSaleCount,last2WeekSaleCount, on = ['Class','dayOfYear'], how='left')
    # weekOnWeekRatio = pd.merge(weekOnWeekRatio,last3WeekSaleCount, on = ['Class','dayOfYear'], how='left')
    # weekOnWeekRatio.loc[:,'weekOn1WeekRatio'] = np.round(weekOnWeekRatio['last1WeekSaleCount'] / (1.0 * weekOnWeekRatio['last2WeekSaleCount']) ,4)
    # weekOnWeekRatio.loc[:,'weekOn2WeekRatio'] = np.round(weekOnWeekRatio['last1WeekSaleCount'] / (1.0 * weekOnWeekRatio['last3WeekSaleCount']) ,4)


    #父类
    # parWeekDayRatio
    # parLastMonthTotSaleCount_o = train_test.groupby(['parClass','month'],as_index=False)['saleCount'].agg({'parLastMonthTotSaleCount':'sum'})
    # parLastMonthTotSaleCount = parLastMonthTotSaleCount_o.shift(1)
    # parLastMonthTotSaleCount['parClass'] = parLastMonthTotSaleCount_o['parClass']
    # parLastMonthTotSaleCount['month'] = parLastMonthTotSaleCount_o['month']
    # parLastMonthTotSaleCount.fillna(1,inplace=True)    # 缺失值处理
    # parLastMonthTotSaleCount['parLastMonthTotSaleCount'][parLastMonthTotSaleCount['parLastMonthTotSaleCount'] == 0.0] = 1  #把分母设为1
    # parLastWeekDayTotSaleCount_o = train_test.groupby(['parClass','month','dayOfWeek'],as_index=False)['saleCount'].agg({'parLastWeekDayTotSaleCount':'sum'})
    # parLastWeekDayTotSaleCount = parLastWeekDayTotSaleCount_o.shift(7)
    # parLastWeekDayTotSaleCount['parClass']     = parLastWeekDayTotSaleCount_o['parClass']
    # parLastWeekDayTotSaleCount['dayOfWeek'] = parLastWeekDayTotSaleCount_o['dayOfWeek']
    # parLastWeekDayTotSaleCount['month']     = parLastWeekDayTotSaleCount_o['month']
    # parLastWeekDayTotSaleCount.fillna(1,inplace=True)    # 缺失值处理
    # parLastWeekDayTotSaleCount['parLastWeekDayTotSaleCount'][parLastWeekDayTotSaleCount['parLastWeekDayTotSaleCount'] == 0.0] = 1  #把分母设为1
    # parLastWeekDayTotSaleCount = pd.merge(parLastWeekDayTotSaleCount,parLastMonthTotSaleCount,on=['parClass','month'],how='left')
    # parLastWeekDayTotSaleCount.loc[:,'parWeekDayRatio'] = np.round(parLastWeekDayTotSaleCount['parLastWeekDayTotSaleCount'] / (1.0 * parLastWeekDayTotSaleCount['parLastMonthTotSaleCount']) ,4)
    parLastMonthTotSaleCount = train_test.groupby(['parClass','month'],as_index=False)['saleCount'].agg({'parLastMonthTotSaleCount':'sum'})
    parLastMonthTotSaleCount['parLastMonthTotSaleCount'] = parLastMonthTotSaleCount.groupby('parClass')['parLastMonthTotSaleCount'].shift(1)
    parLastMonthTotSaleCount['parLastMonthTotSaleCount'].fillna(method='bfill',inplace=True)

    parLastWeekDayTotSaleCount = train_test.groupby(['parClass','month','dayOfWeek'],as_index=False)['saleCount'].agg({'parLastWeekDayTotSaleCount':'sum'})
    coord = parLastWeekDayTotSaleCount.groupby('parClass',as_index=False)['parLastWeekDayTotSaleCount'].shift(7)
    coord.fillna(method='bfill',inplace=True)
    parLastWeekDayTotSaleCount['parLastWeekDayTotSaleCount'] = coord
    parLastWeekDayTotSaleCount = pd.merge(parLastWeekDayTotSaleCount,parLastMonthTotSaleCount,on=['parClass','month'],how='left')
    parLastWeekDayTotSaleCount.loc[:,'parWeekDayRatio'] = np.round(parLastWeekDayTotSaleCount['parLastWeekDayTotSaleCount'] / (1.0 * parLastWeekDayTotSaleCount['parLastMonthTotSaleCount'] + 1) ,4)


    # # parWeekOn1WeekRatio，parWeekOn2WeekRatio
    # parLast1WeekSaleCount_o = train_test.groupby(['parClass','dayOfYear'],as_index=False)['saleCount'].agg({'parLast1WeekSaleCount':'sum'})
    # parLast1WeekSaleCount = parLast1WeekSaleCount_o.shift(7)
    # parLast1WeekSaleCount['parClass'] = parLast1WeekSaleCount_o['parClass']
    # parLast1WeekSaleCount['dayOfYear'] = parLast1WeekSaleCount_o['dayOfYear']
    # parLast1WeekSaleCount['parLast1WeekSaleCount'].fillna(0,inplace=True)
    #
    # parLast2WeekSaleCount = parLast1WeekSaleCount_o.shift(14)
    # parLast2WeekSaleCount.rename(columns={'parLast1WeekSaleCount':'parLast2WeekSaleCount'},inplace=True)
    # parLast2WeekSaleCount['parClass'] = parLast1WeekSaleCount_o['parClass']
    # parLast2WeekSaleCount['dayOfYear'] = parLast1WeekSaleCount_o['dayOfYear']
    # parLast2WeekSaleCount['parLast2WeekSaleCount'].fillna(1,inplace=True)   #把分母设为1
    # parLast2WeekSaleCount['parLast2WeekSaleCount'][parLast2WeekSaleCount['parLast2WeekSaleCount'] == 0.0] = 1  #把分母设为1
    #
    # parLast3WeekSaleCount = parLast1WeekSaleCount_o.shift(21)
    # parLast3WeekSaleCount.rename(columns={'parLast1WeekSaleCount':'parLast3WeekSaleCount'},inplace=True)
    # parLast3WeekSaleCount['parClass'] = parLast1WeekSaleCount_o['parClass']
    # parLast3WeekSaleCount['dayOfYear'] = parLast1WeekSaleCount_o['dayOfYear']
    # parLast3WeekSaleCount['parLast3WeekSaleCount'].fillna(0,inplace=True)   #把分母设为1
    # parLast3WeekSaleCount['parLast3WeekSaleCount'][parLast3WeekSaleCount['parLast3WeekSaleCount'] == 0.0] = 1  #把分母设为1
    #
    #
    # parWeekOnWeekRatio = pd.merge(parLast1WeekSaleCount,parLast2WeekSaleCount, on = ['parClass','dayOfYear'], how='left')
    # parWeekOnWeekRatio = pd.merge(parWeekOnWeekRatio,parLast3WeekSaleCount, on = ['parClass','dayOfYear'], how='left')
    # parWeekOnWeekRatio.loc[:,'parWeekOn1WeekRatio'] = np.round(parWeekOnWeekRatio['parLast1WeekSaleCount'] / (1.0 * parWeekOnWeekRatio['parLast2WeekSaleCount']) ,4)
    # parWeekOnWeekRatio.loc[:,'parWeekOn2WeekRatio'] = np.round(parWeekOnWeekRatio['parLast1WeekSaleCount'] / (1.0 * parWeekOnWeekRatio['parLast3WeekSaleCount']) ,4)
    # # #用于合并
    # # parWeekOnWeekRatio
    #
    # # day3OoverWeek3TotRatio
    # # 类别上周，上2周，上3周,上4周总销量
    # lastWeekSaleCount_o = train_test.groupby(['Class','weekOfYear'],as_index=False)['saleCount'].agg({'lastWeekSaleCount':'sum'})
    # last2WeekSaleCount_o = train_test.groupby(['Class','weekOfYear'],as_index=False)['saleCount'].agg({'last2WeekSaleCount':'sum'})
    # last3WeekSaleCount_o = train_test.groupby(['Class','weekOfYear'],as_index=False)['saleCount'].agg({'last3WeekSaleCount':'sum'})
    # last4WeekSaleCount_o = train_test.groupby(['Class','weekOfYear'],as_index=False)['saleCount'].agg({'last4WeekSaleCount':'sum'})
    #
    # lastWeekSaleCount = lastWeekSaleCount_o.shift(1)
    # last2WeekSaleCount = last2WeekSaleCount_o.shift(2)
    # last3WeekSaleCount = last3WeekSaleCount_o.shift(3)
    # last4WeekSaleCount = last4WeekSaleCount_o.shift(4)
    #
    # lastWeekSaleCount.weekOfYear = lastWeekSaleCount_o.weekOfYear
    # last2WeekSaleCount.weekOfYear = last2WeekSaleCount_o.weekOfYear
    # last3WeekSaleCount.weekOfYear = last3WeekSaleCount_o.weekOfYear
    # last4WeekSaleCount.weekOfYear = last4WeekSaleCount_o.weekOfYear
    #
    # lastWeekSaleCount.Class = lastWeekSaleCount_o.Class
    # last2WeekSaleCount.Class = last2WeekSaleCount_o.Class
    # last3WeekSaleCount.Class = last3WeekSaleCount_o.Class
    # last4WeekSaleCount.Class = last4WeekSaleCount_o.Class
    #
    # lastWeekSaleCount['lastWeekSaleCount'].fillna(0,inplace=True)
    # last2WeekSaleCount['last2WeekSaleCount'].fillna(1,inplace=True)
    # last3WeekSaleCount['last3WeekSaleCount'].fillna(1,inplace=True)
    # last4WeekSaleCount['last4WeekSaleCount'].fillna(1,inplace=True)
    #
    # day3OoverWeek3Tot = pd.merge(lastWeekSaleCount, last2WeekSaleCount, on=['Class','weekOfYear'],how='left')
    # day3OoverWeek3Tot = pd.merge(day3OoverWeek3Tot, last3WeekSaleCount, on=['Class','weekOfYear'],how='left')
    # day3OoverWeek3Tot = pd.merge(day3OoverWeek3Tot, last4WeekSaleCount, on=['Class','weekOfYear'],how='left')
    # day3OoverWeek3Tot.loc[:,'day3OoverWeek3TotRatio'] =  np.round(day3OoverWeek3Tot['lastWeekSaleCount'] /
    #                                                               (1.0 * (day3OoverWeek3Tot['last2WeekSaleCount'] + day3OoverWeek3Tot['last3WeekSaleCount'] + day3OoverWeek3Tot['last4WeekSaleCount'])) ,4)


    #开始合并
    # 合并 train_test
    del lastWeekDayTotSaleCount['lastWeekDayTotSaleCount' ],lastWeekDayTotSaleCount['lastMonthTotSaleCount' ]
    # del weekOnWeekRatio['last1WeekSaleCount'],weekOnWeekRatio['last2WeekSaleCount'],weekOnWeekRatio['last3WeekSaleCount']
    del parLastWeekDayTotSaleCount['parLastWeekDayTotSaleCount' ],   parLastWeekDayTotSaleCount['parLastMonthTotSaleCount' ]
    # del parWeekOnWeekRatio['parLast1WeekSaleCount'],parWeekOnWeekRatio['parLast2WeekSaleCount'],parWeekOnWeekRatio['parLast3WeekSaleCount']
    # del day3OoverWeek3Tot['lastWeekSaleCount'],day3OoverWeek3Tot['last2WeekSaleCount'],day3OoverWeek3Tot['last3WeekSaleCount'],day3OoverWeek3Tot['last4WeekSaleCount']
    tmp = pd.merge(train_test,lastWeekDayTotSaleCount,on=['Class','month','dayOfWeek'],how='left')
    # tmp = pd.merge(tmp,weekOnWeekRatio,on=['Class','dayOfYear'],how='left')
    tmp = pd.merge(tmp,parLastWeekDayTotSaleCount,on=['parClass','month','dayOfWeek'],how='left')
    # tmp = pd.merge(tmp,parWeekOnWeekRatio,on=['parClass','dayOfYear'],how='left')
    # tmp = pd.merge(tmp,day3OoverWeek3Tot,on=['Class','weekOfYear'],how='left')

    # print 'new added features:',np.setdiff1d(tmp.columns, train_test.columns)
    train_test = tmp.copy()


    # lastWeeksSaleCount = train_test.groupby(['Class','weekOfYear'],as_index=False)['saleCount'].agg({'weekSaleCount':'sum'})
    # # lastWeeksSaleCount['lastWeekSaleCount'] = lastWeeksSaleCount['weekSaleCount'].shift(1)
    # # lastWeeksSaleCount.fillna(method='bfill',inplace=True)
    # # lastWeeksSaleCount['last2WeekSaleCount'] = lastWeeksSaleCount['lastWeekSaleCount'].shift(1)
    # # lastWeeksSaleCount.fillna(method='bfill',inplace=True)
    # # lastWeeksSaleCount['last3WeekSaleCount'] = lastWeeksSaleCount['last2WeekSaleCount'].shift(1)
    # # lastWeeksSaleCount.fillna(method='bfill',inplace=True)
    # # lastWeeksSaleCount['last4WeekSaleCount'] = lastWeeksSaleCount['last3WeekSaleCount'].shift(1)
    # # lastWeeksSaleCount.fillna(method='bfill',inplace=True)
    # lastWeeksSaleCount.loc[:,'lastWeekdiff'] = lastWeeksSaleCount['weekSaleCount'].diff(1)
    # lastWeeksSaleCount.loc[:,'last2Weekdiff'] = lastWeeksSaleCount['weekSaleCount'].diff(2)
    # lastWeeksSaleCount.loc[:,'last3Weekdiff'] = lastWeeksSaleCount['weekSaleCount'].diff(3)
    # lastWeeksSaleCount.loc[:,'last3Weekdiff'] = lastWeeksSaleCount['weekSaleCount'].diff(4)
    # del lastWeeksSaleCount['lastWeekdiff']
    # train_test = pd.merge(train_test, lastWeeksSaleCount, on=['Class','weekOfYear'], how='left')

    return train_test

def get_roll_diff_feats(train_test):
    #dayOn1DayDiff,dayOn2DayDiff,dayOn7DayDiff,dayOn14DayDiff
    last7DaysSaleCount = train_test.groupby(['Class','dayOfYear'],as_index=False)['saleCount'].agg({'last7DaysSaleCount':'sum'})
    last7DaysSaleCount['last7DaysSaleCount'] = last7DaysSaleCount.groupby('Class')['last7DaysSaleCount'].shift(7)
    last7DaysSaleCount['last7DaysSaleCount'].fillna(method='bfill',inplace=True)

    last8DaysSaleCount = train_test.groupby(['Class','dayOfYear'],as_index=False)['saleCount'].agg({'last8DaysSaleCount':'sum'})
    last8DaysSaleCount['last8DaysSaleCount'] = last8DaysSaleCount.groupby('Class')['last8DaysSaleCount'].shift(8)
    last8DaysSaleCount['last8DaysSaleCount'].fillna(method='bfill',inplace=True)

    last9DaysSaleCount = train_test.groupby(['Class','dayOfYear'],as_index=False)['saleCount'].agg({'last9DaysSaleCount':'sum'})
    last9DaysSaleCount['last9DaysSaleCount'] = last9DaysSaleCount.groupby('Class')['last9DaysSaleCount'].shift(9)
    last9DaysSaleCount['last9DaysSaleCount'].fillna(method='bfill',inplace=True)

    last10DaysSaleCount = train_test.groupby(['Class','dayOfYear'],as_index=False)['saleCount'].agg({'last10DaysSaleCount':'sum'})
    last10DaysSaleCount['last10DaysSaleCount'] = last10DaysSaleCount.groupby('Class')['last10DaysSaleCount'].shift(10)
    last10DaysSaleCount['last10DaysSaleCount'].fillna(method='bfill',inplace=True)

    last11DaysSaleCount = train_test.groupby(['Class','dayOfYear'],as_index=False)['saleCount'].agg({'last11DaysSaleCount':'sum'})
    last11DaysSaleCount['last11DaysSaleCount'] = last11DaysSaleCount.groupby('Class')['last11DaysSaleCount'].shift(11)
    last11DaysSaleCount['last11DaysSaleCount'].fillna(method='bfill',inplace=True)

    last12DaysSaleCount = train_test.groupby(['Class','dayOfYear'],as_index=False)['saleCount'].agg({'last12DaysSaleCount':'sum'})
    last12DaysSaleCount['last12DaysSaleCount'] = last12DaysSaleCount.groupby('Class')['last12DaysSaleCount'].shift(12)
    last12DaysSaleCount['last12DaysSaleCount'].fillna(method='bfill',inplace=True)

    last13DaysSaleCount = train_test.groupby(['Class','dayOfYear'],as_index=False)['saleCount'].agg({'last13DaysSaleCount':'sum'})
    last13DaysSaleCount['last13DaysSaleCount'] = last13DaysSaleCount.groupby('Class')['last13DaysSaleCount'].shift(13)
    last13DaysSaleCount['last13DaysSaleCount'].fillna(method='bfill',inplace=True)

    last14DaysSaleCount = train_test.groupby(['Class','dayOfYear'],as_index=False)['saleCount'].agg({'last14DaysSaleCount':'sum'})
    last14DaysSaleCount['last14DaysSaleCount'] = last14DaysSaleCount.groupby('Class')['last14DaysSaleCount'].shift(14)
    last14DaysSaleCount['last14DaysSaleCount'].fillna(method='bfill',inplace=True)

    last21DaysSaleCount = train_test.groupby(['Class','dayOfYear'],as_index=False)['saleCount'].agg({'last21DaysSaleCount':'sum'})
    last21DaysSaleCount['last21DaysSaleCount'] = last21DaysSaleCount.groupby('Class')['last21DaysSaleCount'].shift(21)
    last21DaysSaleCount['last21DaysSaleCount'].fillna(method='bfill',inplace=True)

    diff = pd.merge(last7DaysSaleCount,last8DaysSaleCount,on = ['Class','dayOfYear'], how='left')
    diff = pd.merge(diff,last9DaysSaleCount,on = ['Class','dayOfYear'], how='left')
    diff = pd.merge(diff,last10DaysSaleCount,on = ['Class','dayOfYear'], how='left')
    diff = pd.merge(diff,last11DaysSaleCount,on = ['Class','dayOfYear'], how='left')
    diff = pd.merge(diff,last12DaysSaleCount,on = ['Class','dayOfYear'], how='left')
    diff = pd.merge(diff,last13DaysSaleCount,on = ['Class','dayOfYear'], how='left')
    diff = pd.merge(diff,last14DaysSaleCount,on = ['Class','dayOfYear'], how='left')
    diff = pd.merge(diff,last21DaysSaleCount,on = ['Class','dayOfYear'], how='left')

    #前三天均值
    diff.loc[:,'last4Days_mean'] = (diff['last7DaysSaleCount'] + diff['last8DaysSaleCount'] + diff['last9DaysSaleCount'] + diff['last10DaysSaleCount'])/4.0
    diff.loc[:,'last6Days_mean'] = (diff['last7DaysSaleCount'] + diff['last8DaysSaleCount'] + diff['last9DaysSaleCount'] + diff['last10DaysSaleCount'] + diff['last11DaysSaleCount'] + diff['last12DaysSaleCount'])/6.0
    diff.loc[:,'last7Days_mean'] = (diff['last7DaysSaleCount'] + diff['last8DaysSaleCount'] + diff['last9DaysSaleCount'] + diff['last10DaysSaleCount'] + diff['last11DaysSaleCount'] + diff['last12DaysSaleCount'] + diff['last13DaysSaleCount'])/7.0
    diff.loc[:,'last3Days_mean'] = (diff['last7DaysSaleCount'] + diff['last8DaysSaleCount'] + diff['last9DaysSaleCount'])/3.0
    diff.loc[:,'last3InterDays1_mean'] = (diff['last8DaysSaleCount'] + diff['last9DaysSaleCount'] + diff['last10DaysSaleCount'])/3.0
    diff.loc[:,'last3InterDays2_mean'] = (diff['last9DaysSaleCount'] + diff['last10DaysSaleCount'] + diff['last11DaysSaleCount'])/3.0
    diff.loc[:,'last3InterDays3_mean'] = (diff['last10DaysSaleCount'] + diff['last11DaysSaleCount'] + diff['last12DaysSaleCount'])/3.0
    diff.loc[:,'last3InterDays4_mean'] = (diff['last11DaysSaleCount'] + diff['last12DaysSaleCount'] + diff['last13DaysSaleCount'])/3.0
    diff.loc[:,'last3InterDays5_mean'] = (diff['last12DaysSaleCount'] + diff['last13DaysSaleCount'] + diff['last14DaysSaleCount'])/3.0

    #用于合并
    diff.loc[:,'dayOn1DayDiff'] = diff['last7DaysSaleCount'] - diff['last8DaysSaleCount']
    diff.loc[:,'dayOn2DayDiff'] = diff['last7DaysSaleCount'] - diff['last9DaysSaleCount']
    diff.loc[:,'dayOn3DayDiff'] = diff['last7DaysSaleCount'] - diff['last10DaysSaleCount']
    diff.loc[:,'dayOn4DayDiff'] = diff['last7DaysSaleCount'] - diff['last11DaysSaleCount']
    diff.loc[:,'dayOn5DayDiff'] = diff['last7DaysSaleCount'] - diff['last12DaysSaleCount']
    diff.loc[:,'dayOn6DayDiff'] = diff['last7DaysSaleCount'] - diff['last13DaysSaleCount']
    diff.loc[:,'dayOn7DayDiff'] = diff['last7DaysSaleCount'] - diff['last14DaysSaleCount']
    diff.loc[:,'dayOn14DayDiff'] = diff['last7DaysSaleCount'] - diff['last21DaysSaleCount']

    #开始合并
    tmp = pd.merge(train_test,diff[['Class','dayOfYear','dayOn1DayDiff','dayOn2DayDiff','dayOn3DayDiff','dayOn4DayDiff','dayOn5DayDiff','dayOn6DayDiff','dayOn7DayDiff','dayOn14DayDiff','last3Days_mean','last6Days_mean','last3InterDays1_mean','last4Days_mean']],on=['Class','dayOfYear'],how='left')
    # tmp = pd.merge(train_test,diff[['Class','dayOfYear','dayOn1DayDiff','dayOn2DayDiff','dayOn3DayDiff','dayOn4DayDiff','dayOn5DayDiff','dayOn6DayDiff','dayOn7DayDiff','dayOn14DayDiff','last3Days_mean','last6Days_mean','last3InterDays_mean']],on=['Class','dayOfYear'],how='left')
    # del diff['last7DaysSaleCount'],diff['last8DaysSaleCount'],diff['last9DaysSaleCount'],diff['last10DaysSaleCount'],diff['last11DaysSaleCount'],diff['last12DaysSaleCount'],diff['last13DaysSaleCount'],diff['last14DaysSaleCount'],diff['last21DaysSaleCount']
    # tmp = pd.merge(train_test,diff[['Class','dayOfYear']],on=['Class','dayOfYear'],how='left')
    # print 'new added features:',np.setdiff1d(tmp.columns, train_test.columns)
    train_test = tmp.copy()
    return train_test

def get_trend(train_test):
    from statsmodels.tsa.seasonal import seasonal_decompose
    do_not_use_class = [1507,3208,3311,3413]
    train_test.loc[:,'trend_30'] = 0
    # train_test.loc[:,'expweighted_30_avg'] = 0
    # train_test.loc[:,'moving_30_avg'] = 0
    train_test.loc[:,'trend_14'] = 0
    train_test.loc[:,'expweighted_14_avg'] = 0
    # train_test.loc[:,'moving_14_avg'] = 0
    # train_test.loc[:,'trend_7'] = 0
    train_test.loc[:,'expweighted_7_avg'] = 0
    # train_test.loc[:,'moving_7_avg'] = 0
    step_30 = 30
    step_14 = 14
    step_7 = 7
    l = train_test['Class'].unique()
    l = [f for f in l if f not in do_not_use_class]
    for i in l:
        var = train_test[train_test['Class'] == i]['saleCount']
        var = var.values
        # print i
        decomposition_30 = seasonal_decompose(var,freq = step_30)
        # expweighted_avg_30 = pd.ewma(var,halflife= step_30)
        # moving_avg_30 = pd.rolling_mean(var,step_30)
        trend = decomposition_30.trend
        train_test['trend_30'][train_test['Class'] == i] = trend
        # train_test['expweighted_30_avg'][train_test['Class'] == i] = expweighted_avg_30
        # train_test['moving_30_avg'][train_test['Class'] == i] = moving_avg_30
        # decomposition_7 = seasonal_decompose(var,freq = step_7)
        expweighted_avg_7 = pd.ewma(var,halflife= step_7)
        # moving_avg_7 = pd.rolling_mean(var,step_7)
        # trend = decomposition_7.trend
        # train_test['trend_7'][train_test['Class'] == i] = trend
        # train_test['expweighted_7_avg'][train_test['Class'] == i] = expweighted_avg_7
        # train_test['moving_7_avg'][train_test['Class'] == i] = moving_avg_7
        decomposition_14 = seasonal_decompose(var,freq = step_14)
        expweighted_avg_14 = pd.ewma(var,halflife= step_14)
        # moving_avg_14 = pd.rolling_mean(var,step_14)
        trend = decomposition_14.trend
        train_test['trend_14'][train_test['Class'] == i] = trend
        train_test['expweighted_14_avg'][train_test['Class'] == i] = expweighted_avg_14
        # train_test['moving_14_avg'][train_test['Class'] == i] = moving_avg_14
    coord = train_test.groupby(['Class','SaleDate'],as_index=False)['trend_30'].mean()
    coord_shift = coord.shift(step_30)
    coord_shift[['Class','SaleDate']] = coord[['Class','SaleDate']]
    del train_test['trend_30']
    train_test = pd.merge(train_test, coord_shift, on=['Class','SaleDate'], how='left')
    train_test['trend_30'].fillna(0,inplace=True)

    # coord = train_test.groupby(['Class','SaleDate'],as_index=False)['expweighted_30_avg'].mean()
    # coord_shift = coord.shift(step_30)
    # coord_shift[['Class','SaleDate']] = coord[['Class','SaleDate']]
    # del train_test['expweighted_30_avg']
    # train_test = pd.merge(train_test, coord_shift, on=['Class','SaleDate'], how='left')
    # train_test['expweighted_30_avg'].fillna(0,inplace=True)

    # coord = train_test.groupby(['Class','SaleDate'],as_index=False)['moving_30_avg'].mean()
    # coord_shift = coord.shift(step_30)
    # coord_shift[['Class','SaleDate']] = coord[['Class','SaleDate']]
    # del train_test['moving_30_avg']
    # train_test = pd.merge(train_test, coord_shift, on=['Class','SaleDate'], how='left')
    # train_test['moving_30_avg'].fillna(0,inplace=True)

    coord = train_test.groupby(['Class','SaleDate'],as_index=False)['trend_14'].mean()
    coord_shift = coord.shift(step_14)
    coord_shift[['Class','SaleDate']] = coord[['Class','SaleDate']]
    del train_test['trend_14']
    train_test = pd.merge(train_test, coord_shift, on=['Class','SaleDate'], how='left')
    train_test['trend_14'].fillna(0,inplace=True)

    coord = train_test.groupby(['Class','SaleDate'],as_index=False)['expweighted_14_avg'].mean()
    coord_shift = coord.shift(step_14)
    coord_shift[['Class','SaleDate']] = coord[['Class','SaleDate']]
    del train_test['expweighted_14_avg']
    train_test = pd.merge(train_test, coord_shift, on=['Class','SaleDate'], how='left')
    train_test['expweighted_14_avg'].fillna(0,inplace=True)

    # coord = train_test.groupby(['Class','SaleDate'],as_index=False)['moving_14_avg'].mean()
    # coord_shift = coord.shift(step_14)
    # coord_shift[['Class','SaleDate']] = coord[['Class','SaleDate']]
    # del train_test['moving_14_avg']
    # train_test = pd.merge(train_test, coord_shift, on=['Class','SaleDate'], how='left')
    # train_test['moving_14_avg'].fillna(0,inplace=True)

    # coord = train_test.groupby(['Class','SaleDate'],as_index=False)['trend_7'].mean()
    # coord_shift = coord.shift(step_7)
    # coord_shift[['Class','SaleDate']] = coord[['Class','SaleDate']]
    # del train_test['trend_7']
    # train_test = pd.merge(train_test, coord_shift, on=['Class','SaleDate'], how='left')
    # train_test['trend_7'].fillna(0,inplace=True)

    coord = train_test.groupby(['Class','SaleDate'],as_index=False)['expweighted_7_avg'].mean()
    coord_shift = coord.shift(step_7)
    coord_shift[['Class','SaleDate']] = coord[['Class','SaleDate']]
    del train_test['expweighted_7_avg']
    train_test = pd.merge(train_test, coord_shift, on=['Class','SaleDate'], how='left')
    train_test['expweighted_7_avg'].fillna(0,inplace=True)

    # coord = train_test.groupby(['Class','SaleDate'],as_index=False)['moving_7_avg'].mean()
    # coord_shift = coord.shift(step_7)
    # coord_shift[['Class','SaleDate']] = coord[['Class','SaleDate']]
    # del train_test['moving_7_avg']
    # train_test = pd.merge(train_test, coord_shift, on=['Class','SaleDate'], how='left')
    # train_test['moving_7_avg'].fillna(0,inplace=True)

    return train_test

def get_roll_oneday_feats(train_test):
    coord = train_test.groupby(['Class','dayOfYear'],as_index=False)['saleCount'].agg({'saleCount':'mean'})
    #上周
    coord['last1d'] = coord.groupby('Class')['saleCount'].shift(1)
    coord['last1d'].fillna(method='bfill',inplace=True)
    coord['last2d'] = coord.groupby('Class')['last1d'].shift(1)
    coord['last2d'].fillna(method='bfill',inplace=True)
    coord['last3d'] = coord.groupby('Class')['last2d'].shift(1)
    coord['last3d'].fillna(method='bfill',inplace=True)
    coord['last4d'] = coord.groupby('Class')['last3d'].shift(1)
    coord['last4d'].fillna(method='bfill',inplace=True)
    coord['last5d'] = coord.groupby('Class')['last4d'].shift(1)
    coord['last5d'].fillna(method='bfill',inplace=True)
    coord['last6d'] = coord.groupby('Class')['last5d'].shift(1)
    coord['last6d'].fillna(method='bfill',inplace=True)
    coord['last7d'] = coord.groupby('Class')['last6d'].shift(1)
    coord['last7d'].fillna(method='bfill',inplace=True)
    coord.loc[:,'last1wTot'] = coord['last1d']+coord['last2d']+coord['last3d']+coord['last4d']+coord['last5d']+coord['last6d']+coord['last7d']
    coord.loc[:,'last1wMean'] = (coord['last1d']+coord['last2d']+coord['last3d']+coord['last4d']+coord['last5d']+coord['last6d']+coord['last7d'])/7

    #上2周
    coord['last8d'] = coord.groupby('Class')['last7d'].shift(1)
    coord['last8d'].fillna(method='bfill',inplace=True)
    coord['last9d'] = coord.groupby('Class')['last8d'].shift(1)
    coord['last9d'].fillna(method='bfill',inplace=True)
    coord['last10d'] = coord.groupby('Class')['last9d'].shift(1)
    coord['last10d'].fillna(method='bfill',inplace=True)
    coord['last11d'] = coord.groupby('Class')['last10d'].shift(1)
    coord['last11d'].fillna(method='bfill',inplace=True)
    coord['last12d'] = coord.groupby('Class')['last11d'].shift(1)
    coord['last12d'].fillna(method='bfill',inplace=True)
    coord['last13d'] = coord.groupby('Class')['last12d'].shift(1)
    coord['last13d'].fillna(method='bfill',inplace=True)
    coord['last14d'] = coord.groupby('Class')['last13d'].shift(1)
    coord['last14d'].fillna(method='bfill',inplace=True)
    coord.loc[:,'last2wTot'] = coord['last8d']+coord['last9d']+coord['last10d']+coord['last11d']+coord['last12d']+coord['last13d']+coord['last14d']
    coord.loc[:,'last2wMean'] = (coord['last8d']+coord['last9d']+coord['last10d']+coord['last11d']+coord['last12d']+coord['last13d']+coord['last14d'])/7

    #上3周
    coord['last15d'] = coord.groupby('Class')['last14d'].shift(1)
    coord['last15d'].fillna(method='bfill',inplace=True)
    coord['last16d'] = coord.groupby('Class')['last15d'].shift(1)
    coord['last16d'].fillna(method='bfill',inplace=True)
    coord['last17d'] = coord.groupby('Class')['last16d'].shift(1)
    coord['last17d'].fillna(method='bfill',inplace=True)
    coord['last18d'] = coord.groupby('Class')['last17d'].shift(1)
    coord['last18d'].fillna(method='bfill',inplace=True)
    coord['last19d'] = coord.groupby('Class')['last18d'].shift(1)
    coord['last19d'].fillna(method='bfill',inplace=True)
    coord['last20d'] = coord.groupby('Class')['last19d'].shift(1)
    coord['last20d'].fillna(method='bfill',inplace=True)
    coord['last21d'] = coord.groupby('Class')['last20d'].shift(1)
    coord['last21d'].fillna(method='bfill',inplace=True)
    coord.loc[:,'last3wTot'] = coord['last15d']+coord['last16d']+coord['last17d']+coord['last18d']+coord['last19d']+coord['last20d']+coord['last21d']
    coord.loc[:,'last3wMean'] = (coord['last15d']+coord['last16d']+coord['last17d']+coord['last18d']+coord['last19d']+coord['last20d']+coord['last21d'])/7

    #上4周
    coord['last22d'] = coord.groupby('Class')['last21d'].shift(1)
    coord['last22d'].fillna(method='bfill',inplace=True)
    coord['last23d'] = coord.groupby('Class')['last22d'].shift(1)
    coord['last23d'].fillna(method='bfill',inplace=True)
    coord['last24d'] = coord.groupby('Class')['last23d'].shift(1)
    coord['last24d'].fillna(method='bfill',inplace=True)
    coord['last25d'] = coord.groupby('Class')['last24d'].shift(1)
    coord['last25d'].fillna(method='bfill',inplace=True)
    coord['last26d'] = coord.groupby('Class')['last25d'].shift(1)
    coord['last26d'].fillna(method='bfill',inplace=True)
    coord['last27d'] = coord.groupby('Class')['last26d'].shift(1)
    coord['last27d'].fillna(method='bfill',inplace=True)
    coord['last28d'] = coord.groupby('Class')['last27d'].shift(1)
    coord['last28d'].fillna(method='bfill',inplace=True)
    coord.loc[:,'last4wTot'] = coord['last22d']+coord['last23d']+coord['last24d']+coord['last25d']+coord['last26d']+coord['last27d']+coord['last28d']
    coord.loc[:,'last4wMean'] = (coord['last22d']+coord['last23d']+coord['last24d']+coord['last25d']+coord['last26d']+coord['last27d']+coord['last28d'])/7

    coord['last29d'] = coord.groupby('Class')['last28d'].shift(1)
    coord['last29d'].fillna(method='bfill',inplace=True)
    coord['last30d'] = coord.groupby('Class')['last29d'].shift(1)
    coord['last30d'].fillna(method='bfill',inplace=True)
    coord['last31d'] = coord.groupby('Class')['last30d'].shift(1)
    coord['last31d'].fillna(method='bfill',inplace=True)


    #dif天数
    # coord.loc[:,'diff2d'] = coord['last1d'] - coord['last3d']
    # coord.loc[:,'diff3d'] = coord['last1d'] - coord['last4d']
    # coord.loc[:,'diff4d'] = coord['last1d'] - coord['last5d']
    # coord.loc[:,'diff5d'] = coord['last1d'] - coord['last6d']
    # coord.loc[:,'diff6d'] = coord['last1d'] - coord['last7d']
    # coord.loc[:,'diff7d'] = coord['last1d'] - coord['last8d']
    # coord.loc[:,'diff14d'] = coord['last1d'] - coord['last15d']
    # coord.loc[:,'diff21d'] = coord['last1d'] - coord['last22d']

    #dif周
    coord.loc[:,'diff2w'] = coord['last1wTot'] - coord['last2wTot']
    coord.loc[:,'diff3w'] = coord['last1wTot'] - coord['last3wTot']
    coord.loc[:,'diff4w'] = coord['last1wTot'] - coord['last4wTot']
    coord.loc[:,'diff2wRate'] = coord['diff2w'] / (1.0 * coord['last2wTot'] + 1)
    coord.loc[:,'diff3wRate'] = coord['diff3w'] / (1.0 * coord['last3wTot'] + 1)
    coord.loc[:,'diff4wRate'] = coord['diff4w'] / (1.0 * coord['last4wTot'] + 1)
    coord.loc[:,'diff2wRate_mean'] = (coord['last1wMean'] - coord['last2wMean']) / (1.0 * coord['last2wMean'] + 1)
    coord.loc[:,'diff3wRate_mean'] = (coord['last1wMean'] - coord['last3wMean']) / (1.0 * coord['last3wMean'] + 1)
    coord.loc[:,'diff4wRate_mean'] = (coord['last1wMean'] - coord['last4wMean']) / (1.0 * coord['last4wMean'] + 1)

    coord.loc[:,'diffRoll2w'] = coord['last1wTot'] - coord['last2wTot']
    coord.loc[:,'diffRoll3w'] = coord['last2wTot'] - coord['last3wTot']
    coord.loc[:,'diffRoll4w'] = coord['last3wTot'] - coord['last4wTot']
    coord.loc[:,'diffRoll2wRate'] = coord['diffRoll2w'] / (1.0 * coord['last2wTot'] + 1)
    coord.loc[:,'diffRoll3wRate'] = coord['diffRoll3w'] / (1.0 * coord['last3wTot'] + 1)
    coord.loc[:,'diffRoll4wRate'] = coord['diffRoll4w'] / (1.0 * coord['last4wTot'] + 1)
    coord.loc[:,'diffRoll2wRate_mean'] = (coord['last1wMean'] - coord['last2wMean']) / (1.0 * coord['last2wMean'] + 1)
    coord.loc[:,'diffRoll3wRate_mean'] = (coord['last2wMean'] - coord['last3wMean']) / (1.0 * coord['last3wMean'] + 1)
    coord.loc[:,'diffRoll4wRate_mean'] = (coord['last3wMean'] - coord['last4wMean']) / (1.0 * coord['last4wMean'] + 1)
    coord.loc[:,'weekOn1WeekRatio']= coord['last1wTot'] / (coord['last2wTot'] + 1 )
    coord.loc[:,'weekOn2WeekRatio']= coord['last1wTot'] / (coord['last3wTot'] + 1 )
    coord.loc[:,'weekOn3WeekRatio']= coord['last1wTot'] / (coord['last4wTot'] + 1 )
    coord.loc[:,'day3OoverWeek3TotRatio'] = coord['last1wTot'] / (coord['last2wTot'] + coord['last3wTot'] + coord['last4wTot'] + 1)

    #1，2阶差分
    coord.loc[:,'diff_1'] = coord['last1d'] - coord.groupby('Class')['last1d'].shift(1)
    coord.loc[:,'diff_1'].fillna(0,inplace=True)
    coord.loc[:,'diff_2'] = coord['diff_1'] - coord.groupby('Class')['diff_1'].shift(1)
    coord.loc[:,'diff_2'].fillna(0,inplace=True)

    # 滑动统计量
    roll7_mean = coord.groupby(['Class'])['last1d'].rolling(7).mean().reset_index()
    roll7_mean['last1d'] = roll7_mean.groupby('Class')['last1d'].shift(1)
    roll7_mean['last1d'].fillna(method='bfill',inplace=True)
    roll7_mean.rename(columns={'last1d':'last7d_mean'},inplace=True)

    roll7_std = coord.groupby(['Class'])['last1d'].rolling(7).std().reset_index()
    roll7_std['last1d'] = roll7_std.groupby('Class')['last1d'].shift(1)
    roll7_std['last1d'].fillna(method='bfill',inplace=True)
    roll7_std.rename(columns={'last1d':'last7d_std'},inplace=True)

    roll7_median = coord.groupby(['Class'])['last1d'].rolling(7).median().reset_index()
    roll7_median['last1d'] = roll7_median.groupby('Class')['last1d'].shift(1)
    roll7_median['last1d'].fillna(method='bfill',inplace=True)
    roll7_median.rename(columns={'last1d':'last7d_median'},inplace=True)

    roll7_max = coord.groupby(['Class'])['last1d'].rolling(7).max().reset_index()
    roll7_max['last1d'] = roll7_max.groupby('Class')['last1d'].shift(1)
    roll7_max['last1d'].fillna(method='bfill',inplace=True)
    roll7_max.rename(columns={'last1d':'last7d_max'},inplace=True)

    roll14_mean = coord.groupby(['Class'])['last1d'].rolling(14).mean().reset_index()
    roll14_mean['last1d'] = roll14_mean.groupby('Class')['last1d'].shift(1)
    roll14_mean['last1d'].fillna(method='bfill',inplace=True)
    roll14_mean.rename(columns={'last1d':'last14d_mean'},inplace=True)

    roll14_std = coord.groupby(['Class'])['last1d'].rolling(14).std().reset_index()
    roll14_std['last1d'] = roll14_std.groupby('Class')['last1d'].shift(1)
    roll14_std['last1d'].fillna(method='bfill',inplace=True)
    roll14_std.rename(columns={'last1d':'last14d_std'},inplace=True)

    roll14_median = coord.groupby(['Class'])['last1d'].rolling(14).median().reset_index()
    roll14_median['last1d'] = roll14_median.groupby('Class')['last1d'].shift(1)
    roll14_median['last1d'].fillna(method='bfill',inplace=True)
    roll14_median.rename(columns={'last1d':'last14d_median'},inplace=True)

    roll14_max = coord.groupby(['Class'])['last1d'].rolling(14).max().reset_index()
    roll14_max['last1d'] = roll14_max.groupby('Class')['last1d'].shift(1)
    roll14_max['last1d'].fillna(method='bfill',inplace=True)
    roll14_max.rename(columns={'last1d':'last14d_max'},inplace=True)

    roll21_mean = coord.groupby(['Class'])['last1d'].rolling(21).mean().reset_index()
    roll21_mean['last1d'] = roll21_mean.groupby('Class')['last1d'].shift(1)
    roll21_mean['last1d'].fillna(method='bfill',inplace=True)
    roll21_mean.rename(columns={'last1d':'last21d_mean'},inplace=True)

    roll21_std = coord.groupby(['Class'])['last1d'].rolling(21).std().reset_index()
    roll21_std['last1d'] = roll21_std.groupby('Class')['last1d'].shift(1)
    roll21_std['last1d'].fillna(method='bfill',inplace=True)
    roll21_std.rename(columns={'last1d':'last21d_std'},inplace=True)

    roll21_median = coord.groupby(['Class'])['last1d'].rolling(21).median().reset_index()
    roll21_median['last1d'] = roll21_median.groupby('Class')['last1d'].shift(1)
    roll21_median['last1d'].fillna(method='bfill',inplace=True)
    roll21_median.rename(columns={'last1d':'last21d_median'},inplace=True)

    roll21_max = coord.groupby(['Class'])['last1d'].rolling(21).max().reset_index()
    roll21_max['last1d'] = roll21_max.groupby('Class')['last1d'].shift(1)
    roll21_max['last1d'].fillna(method='bfill',inplace=True)
    roll21_max.rename(columns={'last1d':'last21d_max'},inplace=True)

    roll30_mean = coord.groupby(['Class'])['last1d'].rolling(30).mean().reset_index()
    roll30_mean['last1d'] = roll30_mean.groupby('Class')['last1d'].shift(1)
    roll30_mean['last1d'].fillna(method='bfill',inplace=True)
    roll30_mean.rename(columns={'last1d':'last30d_mean'},inplace=True)

    roll30_std = coord.groupby(['Class'])['last1d'].rolling(30).std().reset_index()
    roll30_std['last1d'] = roll30_std.groupby('Class')['last1d'].shift(1)
    roll30_std['last1d'].fillna(method='bfill',inplace=True)
    roll30_std.rename(columns={'last1d':'last30d_std'},inplace=True)

    roll30_median = coord.groupby(['Class'])['last1d'].rolling(30).median().reset_index()
    roll30_median['last1d'] = roll30_median.groupby('Class')['last1d'].shift(1)
    roll30_median['last1d'].fillna(method='bfill',inplace=True)
    roll30_median.rename(columns={'last1d':'last30d_median'},inplace=True)

    roll30_max = coord.groupby(['Class'])['last1d'].rolling(30).max().reset_index()
    roll30_max['last1d'] = roll30_max.groupby('Class')['last1d'].shift(1)
    roll30_max['last1d'].fillna(method='bfill',inplace=True)
    roll30_max.rename(columns={'last1d':'last30d_max'},inplace=True)

    # 滑动统计量 - 短期幅值变化
    roll2_max = coord.groupby(['Class'])['last1d'].rolling(2).max().reset_index()
    roll2_max['last1d'] = roll2_max.groupby('Class')['last1d'].shift(1)
    roll2_max['last1d'].fillna(method='bfill',inplace=True)
    roll2_max.rename(columns={'last1d':'last2d_max'},inplace=True)
    roll2_min = coord.groupby(['Class'])['last1d'].rolling(2).min().reset_index()
    roll2_min['last1d'] = roll2_min.groupby('Class')['last1d'].shift(1)
    roll2_min['last1d'].fillna(method='bfill',inplace=True)
    roll2_min.rename(columns={'last1d':'last2d_min'},inplace=True)
    roll2_mean = coord.groupby(['Class'])['last1d'].rolling(2).mean().reset_index()
    roll2_mean['last1d'] = roll2_mean.groupby('Class')['last1d'].shift(1)
    roll2_mean['last1d'].fillna(method='bfill',inplace=True)
    roll2_mean.rename(columns={'last1d':'last2d_mean'},inplace=True)
    coord.loc[:,'last2d_shake'] = roll2_max['last2d_max'] - roll2_min['last2d_min']
    coord.loc[:,'last2d_shake_rate'] = coord['last2d_shake']/roll2_mean['last2d_mean']

    roll3_max = coord.groupby(['Class'])['last1d'].rolling(3).max().reset_index()
    roll3_max['last1d'] = roll3_max.groupby('Class')['last1d'].shift(1)
    roll3_max['last1d'].fillna(method='bfill',inplace=True)
    roll3_max.rename(columns={'last1d':'last3d_max'},inplace=True)
    roll3_min = coord.groupby(['Class'])['last1d'].rolling(3).min().reset_index()
    roll3_min['last1d'] = roll3_min.groupby('Class')['last1d'].shift(1)
    roll3_min['last1d'].fillna(method='bfill',inplace=True)
    roll3_min.rename(columns={'last1d':'last3d_min'},inplace=True)
    roll3_mean = coord.groupby(['Class'])['last1d'].rolling(3).mean().reset_index()
    roll3_mean['last1d'] = roll3_mean.groupby('Class')['last1d'].shift(1)
    roll3_mean['last1d'].fillna(method='bfill',inplace=True)
    roll3_mean.rename(columns={'last1d':'last3d_mean'},inplace=True)
    coord.loc[:,'last3d_max'] = roll3_max['last3d_max']
    coord.loc[:,'last3d_min'] = roll3_min['last3d_min']
    coord.loc[:,'last3d_shake'] = roll3_max['last3d_max'] - roll3_min['last3d_min']
    coord.loc[:,'last3d_shake_rate'] = coord['last3d_shake']/roll3_mean['last3d_mean']

    roll4_max = coord.groupby(['Class'])['last1d'].rolling(4).max().reset_index()
    roll4_max['last1d'] = roll4_max.groupby('Class')['last1d'].shift(1)
    roll4_max['last1d'].fillna(method='bfill',inplace=True)
    roll4_max.rename(columns={'last1d':'last4d_max'},inplace=True)
    roll4_min = coord.groupby(['Class'])['last1d'].rolling(4).min().reset_index()
    roll4_min['last1d'] = roll4_min.groupby('Class')['last1d'].shift(1)
    roll4_min['last1d'].fillna(method='bfill',inplace=True)
    roll4_min.rename(columns={'last1d':'last4d_min'},inplace=True)
    roll4_mean = coord.groupby(['Class'])['last1d'].rolling(4).mean().reset_index()
    roll4_mean['last1d'] = roll4_mean.groupby('Class')['last1d'].shift(1)
    roll4_mean['last1d'].fillna(method='bfill',inplace=True)
    roll4_mean.rename(columns={'last1d':'last4d_mean'},inplace=True)
    coord.loc[:,'last4d_shake'] = roll4_max['last4d_max'] - roll4_min['last4d_min']
    coord.loc[:,'last4d_shake_rate'] = coord['last4d_shake']/roll4_mean['last4d_mean']

    roll5_max = coord.groupby(['Class'])['last1d'].rolling(5).max().reset_index()
    roll5_max['last1d'] = roll5_max.groupby('Class')['last1d'].shift(1)
    roll5_max['last1d'].fillna(method='bfill',inplace=True)
    roll5_max.rename(columns={'last1d':'last5d_max'},inplace=True)
    roll5_min = coord.groupby(['Class'])['last1d'].rolling(5).min().reset_index()
    roll5_min['last1d'] = roll5_min.groupby('Class')['last1d'].shift(1)
    roll5_min['last1d'].fillna(method='bfill',inplace=True)
    roll5_min.rename(columns={'last1d':'last5d_min'},inplace=True)
    roll5_mean = coord.groupby(['Class'])['last1d'].rolling(5).mean().reset_index()
    roll5_mean['last1d'] = roll5_mean.groupby('Class')['last1d'].shift(1)
    roll5_mean['last1d'].fillna(method='bfill',inplace=True)
    roll5_mean.rename(columns={'last1d':'last5d_mean'},inplace=True)
    coord.loc[:,'last5d_shake'] = roll5_max['last5d_max'] - roll5_min['last5d_min']
    # coord.loc[:,'last5d_shake_rate'] = coord['last5d_shake']/roll5_mean['last5d_mean']

    roll7_max = coord.groupby(['Class'])['last1d'].rolling(7).max().reset_index()
    roll7_max['last1d'] = roll7_max.groupby('Class')['last1d'].shift(1)
    roll7_max['last1d'].fillna(method='bfill',inplace=True)
    roll7_max.rename(columns={'last1d':'last7d_max'},inplace=True)
    roll7_min = coord.groupby(['Class'])['last1d'].rolling(7).min().reset_index()
    roll7_min['last1d'] = roll7_min.groupby('Class')['last1d'].shift(1)
    roll7_min['last1d'].fillna(method='bfill',inplace=True)
    roll7_min.rename(columns={'last1d':'last7d_min'},inplace=True)
    roll7_mean = coord.groupby(['Class'])['last1d'].rolling(7).mean().reset_index()
    roll7_mean['last1d'] = roll7_mean.groupby('Class')['last1d'].shift(1)
    roll7_mean['last1d'].fillna(method='bfill',inplace=True)
    roll7_mean.rename(columns={'last1d':'last7d_mean'},inplace=True)
    # coord.loc[:,'last7d_shake'] = roll7_max['last7d_max'] - roll7_min['last7d_min']
    # coord.loc[:,'last7d_shake_rate'] = coord['last7d_shake']/roll7_mean['last7d_mean']


    coord.loc[:,'last7d_mean'] = roll7_mean['last7d_mean']
    # coord.loc[:,'last7d_std'] = roll7_std['last7d_std']
    coord.loc[:,'last7d_median'] = roll7_median['last7d_median']
    # coord.loc[:,'last7d_max'] = roll7_max['last7d_max']
    coord.loc[:,'last14d_mean'] = roll14_mean['last14d_mean']
    # coord.loc[:,'last14d_std'] = roll14_std['last14d_std']
    coord.loc[:,'last14d_median'] = roll14_median['last14d_median']
    # coord.loc[:,'last14d_max'] = roll14_max['last14d_max']
    coord.loc[:,'last21d_mean'] = roll21_mean['last21d_mean']
    # coord.loc[:,'last21d_std'] = roll21_std['last21d_std']
    coord.loc[:,'last21d_median'] = roll21_median['last21d_median']
    # coord.loc[:,'last21d_max'] = roll21_max['last21d_max']
    coord.loc[:,'last30d_mean'] = roll30_mean['last30d_mean']
    # coord.loc[:,'last30d_std'] = roll30_std['last30d_std']
    coord.loc[:,'last30d_median'] = roll30_median['last30d_median']
    # coord.loc[:,'last30d_max'] = roll30_max['last30d_max']


    #鲁棒值
    coord.loc[:,'last7d_div_median'] = coord['last1d'] / (1.0 * coord['last7d_median'] + 0.01) * coord['last7d_mean']
    coord.loc[:,'last7d_div_mean'] = coord['last1d'] / (1.0 * coord['last7d_mean'] + 0.01)
    coord.loc[:,'last14d_div_median'] = coord['last1d'] / (1.0 * coord['last14d_median'] + 0.01) * coord['last14d_mean']
    coord.loc[:,'last14d_div_mean'] = coord['last1d'] / (1.0 * coord['last14d_mean'] + 0.01)
    coord.loc[:,'last21d_div_median'] = coord['last1d'] / (1.0 * coord['last21d_median'] + 0.01) * coord['last21d_mean']
    coord.loc[:,'last21d_div_mean'] = coord['last1d'] / (1.0 * coord['last21d_mean'] + 0.01)

    coord.loc[:,'daySale_pct'] = coord.groupby('Class')['last1d'].pct_change(period=1)
    coord.fillna(0,inplace=True)
    coord['daySale_pct'][np.isinf(coord['daySale_pct'])] = 0

    coord.loc[:,'last1dOn7'] = (coord['last1d'] - coord['last8d'])
    coord.loc[:,'last1dOn14'] = (coord['last1d'] - coord['last15d'])
    coord.loc[:,'last1dOn21'] = (coord['last1d'] - coord['last22d'])
    coord.loc[:,'last1dOn28'] = (coord['last1d'] - coord['last29d'])
    coord.loc[:,'last7dOn14'] = (coord['last7d'] - coord['last15d'])
    coord.loc[:,'last14dOn21'] = (coord['last14d'] - coord['last22d'])
    coord.loc[:,'last21On28'] = (coord['last21d'] - coord['last29d'])


    # del coord['last1d']
    del coord['last2d'],coord['last3d'],coord['last4d'],coord['last5d'],coord['last6d'],coord['last7d'],coord['last8d'],coord['last9d'],coord['last10d'],coord['last11d'],coord['last12d'],coord['last13d'],coord['last14d']
    del coord['last15d'],coord['last16d'],coord['last17d'],coord['last18d'],coord['last19d'],coord['last20d'],coord['last21d'],coord['last22d'],coord['last23d'],coord['last24d'],coord['last25d'],coord['last26d'],coord['last27d'],coord['last28d'],coord['last29d'],coord['last30d'],coord['last31d']
    # del coord['last2d'],coord['last3d'],coord['last4d'],coord['last5d'],coord['last6d'],coord['last8d'],coord['last9d'],coord['last10d'],coord['last11d'],coord['last12d'],coord['last13d']
    # del coord['last15d'],coord['last16d'],coord['last17d'],coord['last18d'],coord['last19d'],coord['last20d'],coord['last22d'],coord['last23d'],coord['last24d'],coord['last25d'],coord['last26d'],coord['last27d']
    del coord['last4wTot'],coord['last3wTot'],coord['last2wTot'],coord['last1wTot'],coord['diff2w'],coord['diff3w'],coord['diff4w']
    del roll7_mean['last7d_mean'],roll14_mean['last14d_mean'],roll21_mean['last21d_mean']
    del coord['saleCount']
    train_test = pd.merge(train_test,coord,on=['Class','dayOfYear'],how='left')


    # 获取趋势残差
    # train_test.loc[:,'residual_14'] = train_test['last1d'] - train_test['trend_14']
    # train_test.loc[:,'residual_exp_14'] = train_test['last1d'] - train_test['expweighted_14_avg']

    train_test.loc[:,'residual_trend_30'] = train_test['last1d'] - train_test['trend_30']

    return train_test

def get_roll_hol_feats(train_test):
    coord =train_test.groupby(['Class','dayOfYear'],as_index=False)['disHoliday_detail'].agg({'disHolidayLast1d_detail':'mean'})
    coord['disHolidayLast1d_detail'] = coord.groupby('Class')['disHolidayLast1d_detail'].shift(1)
    coord['disHolidayLast1d_detail'].fillna(0,inplace=True)
    train_test = pd.merge(train_test,coord,on=['Class','dayOfYear'],how='left')
    coord =train_test.groupby(['Class','disHolidayLast1d_detail'],as_index=False)['residual_trend_30'].agg({'residual_hol_last1d_30':'mean'})
    train_test = pd.merge(train_test,coord,on=['Class','disHolidayLast1d_detail'],how='left')

    coord =train_test.groupby(['Class','dayOfYear'],as_index=False)['disWorkday_detail'].agg({'disWorkdayLast1d_detail':'mean'})
    coord['disWorkdayLast1d_detail'] = coord.groupby('Class')['disWorkdayLast1d_detail'].shift(1)
    coord['disWorkdayLast1d_detail'].fillna(0,inplace=True)
    train_test = pd.merge(train_test,coord,on=['Class','dayOfYear'],how='left')
    coord =train_test.groupby(['Class','disWorkdayLast1d_detail'],as_index=False)['residual_trend_30'].agg({'residual_wk_last1d_30':'mean'})
    train_test = pd.merge(train_test,coord,on=['Class','disWorkdayLast1d_detail'],how='left')

    coord =train_test.groupby(['Class','dayOfYear'],as_index=False)['disholDaySaleCount_mean'].agg({'disholDayLast1dSaleCount_mean':'mean'})
    coord['disholDayLast1dSaleCount_mean'] = coord.groupby('Class')['disholDayLast1dSaleCount_mean'].shift(1)
    coord['disholDayLast1dSaleCount_mean'].fillna(0,inplace=True)
    train_test = pd.merge(train_test,coord,on=['Class','dayOfYear'],how='left')


    return train_test


def get_roll_feats(train_test):
    l_feat_original = train_test.columns
    print "Start extracting rolling features....."
    train_test = get_trend(train_test)
    # print "Roll trend done."
    train_test = get_roll_hot_index_feats(train_test)
    # print "Roll hot index features done."
    train_test = get_roll_price_feats(train_test)
    # print "Roll price features done."
    train_test = get_roll_week_sale_feats(train_test)
    # print "Roll week sale features done."
    train_test = get_roll_diff_feats(train_test)
    # print "Roll differentiate features done."
    train_test = get_roll_oneday_feats(train_test)

    train_test = get_roll_hol_feats(train_test)
    print "Rolling features done. "
    l_feat_new = train_test.columns
    l_roll_feats = np.setdiff1d(l_feat_new,l_feat_original)
    return train_test,l_roll_feats