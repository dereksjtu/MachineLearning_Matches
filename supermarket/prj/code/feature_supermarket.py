# -*- coding:utf-8 -*-
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Handle table like and matrices
import pandas as pd
import numpy as np

train_path = '../input/train.csv'
test_path = '../input/test.csv'
hol_path = '../input/holiday.csv'
train_date_path = '../input/train_date.csv'
cache_path = '../input/cache/'
output_path = '../output/'

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

def get_hol_sale_feats(train_new,test):
    train_wk = train_new[train_new.holidayCluster == 1]
    train_hol = train_new[train_new.holidayCluster != 1]

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
    # train_new, test = get_price_feats(train,train_new, test)
    # print "Commodity price features done."
    train_new, test = get_coupon_feats(train, train_new, test)
    print "Coupon features done."
    train_new, test = get_coupon_hol_feats(train , train_new, test)
    print "Coupon holiday features done."
    train_new, test = get_coupon_weekday_feats(train, train_new, test)
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
    lastWeekSaleCount_o = train_test.groupby(['Class','weekOfYear'],as_index=False)['saleCount'].agg({'lastWeekSaleCount':'sum'})
    last2WeekSaleCount_o = train_test.groupby(['Class','weekOfYear'],as_index=False)['saleCount'].agg({'last2WeekSaleCount':'sum'})
    lastMonthSaleCount_o = train_test.groupby(['Class','month'],as_index=False)['saleCount'].agg({'lastMonthSaleCount':'sum'})
    lastWeekSaleCount = lastWeekSaleCount_o.shift(1)
    last2WeekSaleCount = last2WeekSaleCount_o.shift(2)
    lastMonthSaleCount = lastMonthSaleCount_o.shift(1)
    lastWeekSaleCount.weekOfYear = lastWeekSaleCount_o.weekOfYear
    last2WeekSaleCount.weekOfYear = last2WeekSaleCount_o.weekOfYear
    lastMonthSaleCount.month = lastMonthSaleCount_o.month
    lastWeekSaleCount.Class = lastWeekSaleCount_o.Class
    last2WeekSaleCount.Class = last2WeekSaleCount_o.Class
    lastMonthSaleCount.Class = lastMonthSaleCount_o.Class
    lastWeekSaleCount.lastWeekSaleCount.fillna(0,inplace=True)
    last2WeekSaleCount.last2WeekSaleCount.fillna(0,inplace=True)
    lastMonthSaleCount.lastMonthSaleCount.fillna(0,inplace = True)

    # 上周，上上周，上个月总销量
    lastWeekTotSaleCount_o = train_test.groupby(['weekOfYear'],as_index=False)['saleCount'].agg({'lastWeekTotSaleCount':'sum'})
    last2WeekTotSaleCount_o = train_test.groupby(['weekOfYear'],as_index=False)['saleCount'].agg({'last2WeekTotSaleCount':'sum'})
    lastMonthTotSaleCount_o = train_test.groupby(['month'],as_index=False)['saleCount'].agg({'lastMonthTotSaleCount':'sum'})
    lastWeekTotSaleCount = lastWeekTotSaleCount_o.shift(1)
    last2WeekTotSaleCount = last2WeekTotSaleCount_o.shift(2)
    lastMonthTotSaleCount = lastMonthTotSaleCount_o.shift(1)
    lastWeekTotSaleCount.weekOfYear = lastWeekTotSaleCount_o.weekOfYear
    last2WeekTotSaleCount.weekOfYear = last2WeekTotSaleCount_o.weekOfYear
    lastMonthTotSaleCount.month = lastMonthTotSaleCount_o.month
    lastWeekTotSaleCount.lastWeekTotSaleCount.fillna(1,inplace=True)
    last2WeekTotSaleCount.last2WeekTotSaleCount.fillna(1,inplace=True)
    lastMonthTotSaleCount.lastMonthTotSaleCount.fillna(1,inplace = True)

    lastWeekSaleCount = pd.merge(lastWeekSaleCount,lastWeekTotSaleCount, on = 'weekOfYear',how = 'left')
    last2WeekSaleCount = pd.merge(last2WeekSaleCount,last2WeekTotSaleCount, on = 'weekOfYear',how = 'left')
    lastMonthSaleCount = pd.merge(lastMonthSaleCount,lastMonthTotSaleCount, on = 'month',how = 'left')

    # 用于合并
    lastWeekSaleCount.loc[:,'hotPast1WeekIndex'] = np.round(lastWeekSaleCount.lastWeekSaleCount / (1.0 * lastWeekSaleCount.lastWeekTotSaleCount),4)
    last2WeekSaleCount.loc[:,'hotPast2WeekIndex'] = np.round(last2WeekSaleCount.last2WeekSaleCount / (1.0 * last2WeekSaleCount.last2WeekTotSaleCount),4)
    lastMonthSaleCount.loc[:,'hotPast1MonthIndex'] = np.round(lastMonthSaleCount.lastMonthSaleCount / (1.0 * lastMonthSaleCount.lastMonthTotSaleCount),4)


    # 父类
    # 父类别上周，上上周，上个月总销量
    parLastWeekSaleCount_o = train_test.groupby(['parClass','weekOfYear'],as_index=False)['saleCount'].agg({'lastWeekSaleCount':'sum'})
    parLast2WeekSaleCount_o = train_test.groupby(['parClass','weekOfYear'],as_index=False)['saleCount'].agg({'last2WeekSaleCount':'sum'})
    parLastMonthSaleCount_o = train_test.groupby(['parClass','month'],as_index=False)['saleCount'].agg({'lastMonthSaleCount':'sum'})
    parLastWeekSaleCount =  parLastWeekSaleCount_o.shift(1)
    parLast2WeekSaleCount = parLast2WeekSaleCount_o.shift(2)
    parLastMonthSaleCount = parLastMonthSaleCount_o.shift(1)
    parLastWeekSaleCount.weekOfYear =  parLastWeekSaleCount_o.weekOfYear
    parLast2WeekSaleCount.weekOfYear = parLast2WeekSaleCount_o.weekOfYear
    parLastMonthSaleCount.month = parLastMonthSaleCount_o.month
    parLastWeekSaleCount.parClass  =  parLastWeekSaleCount_o.parClass
    parLast2WeekSaleCount.parClass = parLast2WeekSaleCount_o.parClass
    parLastMonthSaleCount.parClass = parLastMonthSaleCount_o.parClass
    parLastWeekSaleCount.lastWeekSaleCount.fillna(0,inplace=True)
    parLast2WeekSaleCount.last2WeekSaleCount.fillna(0,inplace=True)
    parLastMonthSaleCount.lastMonthSaleCount.fillna(0,inplace = True)

    # 上周总销量
    # lastWeekTotSaleCount_o = train_test.groupby(['weekOfYear'],as_index=False)['saleCount'].agg({'lastWeekTotSaleCount':'sum'})
    # last2WeekTotSaleCount_o = train_test.groupby(['weekOfYear'],as_index=False)['saleCount'].agg({'last2WeekTotSaleCount':'sum'})
    # lastMonthTotSaleCount_o = train_test.groupby(['month'],as_index=False)['saleCount'].agg({'lastMonthTotSaleCount':'sum'})
    # lastWeekTotSaleCount = lastWeekTotSaleCount_o.shift(1)
    # last2WeekTotSaleCount = last2WeekTotSaleCount_o.shift(2)
    # lastMonthTotSaleCount = lastMonthTotSaleCount_o.shift(1)
    # lastWeekTotSaleCount.weekOfYear = lastWeekTotSaleCount_o.weekOfYear
    # last2WeekTotSaleCount.weekOfYear = last2WeekTotSaleCount_o.weekOfYear
    # lastMonthTotSaleCount.month = lastMonthTotSaleCount_o.month
    # lastWeekTotSaleCount.lastWeekTotSaleCount.fillna(1,inplace=True)
    # last2WeekTotSaleCount.last2WeekTotSaleCount.fillna(1,inplace=True)
    # lastMonthTotSaleCount.lastMonthTotSaleCount.fillna(1,inplace = True)

    parLastWeekSaleCount = pd.merge (parLastWeekSaleCount,lastWeekTotSaleCount, on = 'weekOfYear',how = 'left')
    parLast2WeekSaleCount = pd.merge(parLast2WeekSaleCount,last2WeekTotSaleCount, on = 'weekOfYear',how = 'left')
    parLastMonthSaleCount = pd.merge(parLastMonthSaleCount,lastMonthTotSaleCount, on = 'month',how = 'left')

    # 用于合并
    parLastWeekSaleCount.loc [:,'parHotPast1WeekIndex']  = np.round(parLastWeekSaleCount.lastWeekSaleCount /   (1.0 * parLastWeekSaleCount.lastWeekTotSaleCount),4)
    parLast2WeekSaleCount.loc[:,'parHotPast2WeekIndex']  = np.round(parLast2WeekSaleCount.last2WeekSaleCount / (1.0 * parLast2WeekSaleCount.last2WeekTotSaleCount),4)
    parLastMonthSaleCount.loc[:,'parHotPast1MonthIndex'] = np.round(parLastMonthSaleCount.lastMonthSaleCount / (1.0 * parLastMonthSaleCount.lastMonthTotSaleCount),4)

    # 合并 train_test
    del  lastWeekSaleCount['lastWeekSaleCount' ],   lastWeekSaleCount['lastWeekTotSaleCount']
    del last2WeekSaleCount['last2WeekSaleCount'],last2WeekSaleCount['last2WeekTotSaleCount']
    del lastMonthSaleCount['lastMonthSaleCount'],lastMonthSaleCount['lastMonthTotSaleCount']
    del parLastWeekSaleCount ['lastWeekSaleCount' ],  parLastWeekSaleCount['lastWeekTotSaleCount']
    del parLast2WeekSaleCount['last2WeekSaleCount'],parLast2WeekSaleCount['last2WeekTotSaleCount']
    del parLastMonthSaleCount['lastMonthSaleCount'],parLastMonthSaleCount['lastMonthTotSaleCount']
    tmp = pd.merge(train_test,lastWeekSaleCount,on=['Class','weekOfYear'],how='left')
    tmp = pd.merge(tmp,last2WeekSaleCount,on=['Class','weekOfYear'],how='left')
    tmp = pd.merge(tmp,lastMonthSaleCount,on=['Class','month'],how='left')
    tmp = pd.merge(tmp,parLastWeekSaleCount,on=['parClass','weekOfYear'],how='left')
    tmp = pd.merge(tmp,parLast2WeekSaleCount,on=['parClass','weekOfYear'],how='left')
    tmp = pd.merge(tmp,parLastMonthSaleCount,on=['parClass','month'],how='left')
    # print 'new added features:',np.setdiff1d(tmp.columns, train_test.columns)
    train_test = tmp.copy()

    return train_test

def get_roll_price_feats(train_test):
    # 类别上周，上上周，上个月销量统计量 - 均值
    lastWeekSaleCount_mean_o = train_test.groupby(['Class','weekOfYear'],as_index=False)['saleCount'].agg({'lastWeekSaleCount_mean':'mean'})
    last2WeekSaleCount_mean_o = train_test.groupby(['Class','weekOfYear'],as_index=False)['saleCount'].agg({'last2WeekSaleCount_mean':'mean'})
    lastMonthSaleCount_mean_o = train_test.groupby(['Class','month'],as_index=False)['saleCount'].agg({'lastMonthSaleCount_mean':'mean'})
    lastWeekSaleCount_mean   = lastWeekSaleCount_mean_o.shift(1)
    last2WeekSaleCount_mean = last2WeekSaleCount_mean_o.shift(2)
    lastMonthSaleCount_mean = lastMonthSaleCount_mean_o.shift(1)
    lastWeekSaleCount_mean.weekOfYear  = lastWeekSaleCount_mean_o.weekOfYear
    last2WeekSaleCount_mean.weekOfYear = last2WeekSaleCount_mean_o.weekOfYear
    lastMonthSaleCount_mean.month      = lastMonthSaleCount_mean_o.month
    lastWeekSaleCount_mean.Class =  lastWeekSaleCount_mean_o.Class
    last2WeekSaleCount_mean.Class = last2WeekSaleCount_mean_o.Class
    lastMonthSaleCount_mean.Class = lastMonthSaleCount_mean_o.Class
    lastWeekSaleCount_mean.lastWeekSaleCount_mean.fillna(0,inplace=True)
    last2WeekSaleCount_mean.last2WeekSaleCount_mean.fillna(0,inplace=True)
    lastMonthSaleCount_mean.lastMonthSaleCount_mean.fillna(0,inplace = True)

    # # 用于合并
    # lastWeekSaleCount_mean
    # last2WeekSaleCount_mean
    # lastMonthSaleCount_mean

    # 父类
    # 父类别上周，上上周，上个月销量统计量 - 中位数
    lastWeekSaleCount_median_o = train_test.groupby(['Class','weekOfYear'],as_index=False)['saleCount'].agg({'lastWeekSaleCount_median':'median'})
    last2WeekSaleCount_median_o = train_test.groupby(['Class','weekOfYear'],as_index=False)['saleCount'].agg({'last2WeekSaleCount_median':'median'})
    lastMonthSaleCount_median_o = train_test.groupby(['Class','month'],as_index=False)['saleCount'].agg({'lastMonthSaleCount_median':'median'})
    lastWeekSaleCount_median   = lastWeekSaleCount_median_o.shift(1)
    last2WeekSaleCount_median = last2WeekSaleCount_median_o.shift(2)
    lastMonthSaleCount_median = lastMonthSaleCount_median_o.shift(1)
    lastWeekSaleCount_median.weekOfYear  = lastWeekSaleCount_median_o.weekOfYear
    last2WeekSaleCount_median.weekOfYear = last2WeekSaleCount_median_o.weekOfYear
    lastMonthSaleCount_median.month      = lastMonthSaleCount_median_o.month
    lastWeekSaleCount_median.Class =  lastWeekSaleCount_median_o.Class
    last2WeekSaleCount_median.Class = last2WeekSaleCount_median_o.Class
    lastMonthSaleCount_median.Class = lastMonthSaleCount_median_o.Class
    lastWeekSaleCount_median.lastWeekSaleCount_median.fillna(0,inplace=True)
    last2WeekSaleCount_median.last2WeekSaleCount_median.fillna(0,inplace=True)
    lastMonthSaleCount_median.lastMonthSaleCount_median.fillna(0,inplace = True)

    # 类别上周，上上周，上个月销量统计量 - 标准差
    lastWeekSaleCount_std_o = train_test.groupby(['Class','weekOfYear'],as_index=False)['saleCount'].agg({'lastWeekSaleCount_std':'std'})
    last2WeekSaleCount_std_o = train_test.groupby(['Class','weekOfYear'],as_index=False)['saleCount'].agg({'last2WeekSaleCount_std':'std'})
    lastMonthSaleCount_std_o = train_test.groupby(['Class','month'],as_index=False)['saleCount'].agg({'lastMonthSaleCount_std':'std'})
    lastWeekSaleCount_std   = lastWeekSaleCount_std_o.shift(1)
    last2WeekSaleCount_std = last2WeekSaleCount_std_o.shift(2)
    lastMonthSaleCount_std = lastMonthSaleCount_std_o.shift(1)
    lastWeekSaleCount_std.weekOfYear  = lastWeekSaleCount_std_o.weekOfYear
    last2WeekSaleCount_std.weekOfYear = last2WeekSaleCount_std_o.weekOfYear
    lastMonthSaleCount_std.month      = lastMonthSaleCount_std_o.month
    lastWeekSaleCount_std.Class =  lastWeekSaleCount_std_o.Class
    last2WeekSaleCount_std.Class = last2WeekSaleCount_std_o.Class
    lastMonthSaleCount_std.Class = lastMonthSaleCount_std_o.Class
    lastWeekSaleCount_std.lastWeekSaleCount_std.fillna(0,inplace=True)
    last2WeekSaleCount_std.last2WeekSaleCount_std.fillna(0,inplace=True)
    lastMonthSaleCount_std.lastMonthSaleCount_std.fillna(0,inplace = True)

    # # 用于合并
    # lastWeekSaleCount_median
    # last2WeekSaleCount_median
    # lastMonthSaleCount_median

    # 父类别上周，上上周，上个月销量统计量 - 均值
    parLastWeekSaleCount_mean_o = train_test.groupby(['parClass','weekOfYear'],as_index=False)['saleCount'].agg({'parLastWeekSaleCount_mean':'mean'})
    parLast2WeekSaleCount_mean_o = train_test.groupby(['parClass','weekOfYear'],as_index=False)['saleCount'].agg({'parLast2WeekSaleCount_mean':'mean'})
    parLastMonthSaleCount_mean_o = train_test.groupby(['parClass','month'],as_index=False)['saleCount'].agg({'parLastMonthSaleCount_mean':'mean'})
    parLastWeekSaleCount_mean   = parLastWeekSaleCount_mean_o.shift(1)
    parLast2WeekSaleCount_mean = parLast2WeekSaleCount_mean_o.shift(2)
    parLastMonthSaleCount_mean = parLastMonthSaleCount_mean_o.shift(1)
    parLastWeekSaleCount_mean.weekOfYear  = parLastWeekSaleCount_mean_o.weekOfYear
    parLast2WeekSaleCount_mean.weekOfYear = parLast2WeekSaleCount_mean_o.weekOfYear
    parLastMonthSaleCount_mean.month      = parLastMonthSaleCount_mean_o.month
    parLastWeekSaleCount_mean.parClass =  parLastWeekSaleCount_mean_o.parClass
    parLast2WeekSaleCount_mean.parClass = parLast2WeekSaleCount_mean_o.parClass
    parLastMonthSaleCount_mean.parClass = parLastMonthSaleCount_mean_o.parClass
    parLastWeekSaleCount_mean.parLastWeekSaleCount_mean.fillna(0,inplace=True)
    parLast2WeekSaleCount_mean.parLast2WeekSaleCount_mean.fillna(0,inplace=True)
    parLastMonthSaleCount_mean.parLastMonthSaleCount_mean.fillna(0,inplace = True)

    # # 用于合并
    # parLastWeekSaleCount_mean
    # parLast2WeekSaleCount_mean
    # parLastMonthSaleCount_mean

    # 类别上周，上上周，上个月销量统计量 - 中位数
    parLastWeekSaleCount_median_o = train_test.groupby(['parClass','weekOfYear'],as_index=False)['saleCount'].agg({'parLastWeekSaleCount_median':'median'})
    parLast2WeekSaleCount_median_o = train_test.groupby(['parClass','weekOfYear'],as_index=False)['saleCount'].agg({'parLast2WeekSaleCount_median':'median'})
    parLastMonthSaleCount_median_o = train_test.groupby(['parClass','month'],as_index=False)['saleCount'].agg({'parLastMonthSaleCount_median':'median'})
    parLastWeekSaleCount_median   = parLastWeekSaleCount_median_o.shift(1)
    parLast2WeekSaleCount_median = parLast2WeekSaleCount_median_o.shift(2)
    parLastMonthSaleCount_median = parLastMonthSaleCount_median_o.shift(1)
    parLastWeekSaleCount_median.weekOfYear  = parLastWeekSaleCount_median_o.weekOfYear
    parLast2WeekSaleCount_median.weekOfYear = parLast2WeekSaleCount_median_o.weekOfYear
    parLastMonthSaleCount_median.month      = parLastMonthSaleCount_median_o.month
    parLastWeekSaleCount_median.parClass =  parLastWeekSaleCount_median_o.parClass
    parLast2WeekSaleCount_median.parClass = parLast2WeekSaleCount_median_o.parClass
    parLastMonthSaleCount_median.parClass = parLastMonthSaleCount_median_o.parClass
    parLastWeekSaleCount_median.parLastWeekSaleCount_median.fillna(0,inplace=True)
    parLast2WeekSaleCount_median.parLast2WeekSaleCount_median.fillna(0,inplace=True)
    parLastMonthSaleCount_median.parLastMonthSaleCount_median.fillna(0,inplace = True)

    # 类别上周，上上周，上个月销量统计量 - 标准差
    parLastWeekSaleCount_std_o = train_test.groupby(['parClass','weekOfYear'],as_index=False)['saleCount'].agg({'parLastWeekSaleCount_std':'std'})
    parLast2WeekSaleCount_std_o = train_test.groupby(['parClass','weekOfYear'],as_index=False)['saleCount'].agg({'parLast2WeekSaleCount_std':'std'})
    parLastMonthSaleCount_std_o = train_test.groupby(['parClass','month'],as_index=False)['saleCount'].agg({'parLastMonthSaleCount_std':'std'})
    parLastWeekSaleCount_std   = parLastWeekSaleCount_std_o.shift(1)
    parLast2WeekSaleCount_std = parLast2WeekSaleCount_std_o.shift(2)
    parLastMonthSaleCount_std = parLastMonthSaleCount_std_o.shift(1)
    parLastWeekSaleCount_std.weekOfYear  = parLastWeekSaleCount_std_o.weekOfYear
    parLast2WeekSaleCount_std.weekOfYear = parLast2WeekSaleCount_std_o.weekOfYear
    parLastMonthSaleCount_std.month      = parLastMonthSaleCount_std_o.month
    parLastWeekSaleCount_std.parClass =  parLastWeekSaleCount_std_o.parClass
    parLast2WeekSaleCount_std.parClass = parLast2WeekSaleCount_std_o.parClass
    parLastMonthSaleCount_std.parClass = parLastMonthSaleCount_std_o.parClass
    parLastWeekSaleCount_std.parLastWeekSaleCount_std.fillna(0,inplace=True)
    parLast2WeekSaleCount_std.parLast2WeekSaleCount_std.fillna(0,inplace=True)
    parLastMonthSaleCount_std.parLastMonthSaleCount_std.fillna(0,inplace = True)

    # # 类别上周，上上周，上个月销量统计量 - 极差
    # parLastWeekSaleCount_ptp_o = train_test.groupby(['parClass','weekOfYear'],as_index=False)['saleCount'].agg({'parLastWeekSaleCount_ptp':'ptp'})
    # parLast2WeekSaleCount_ptp_o = train_test.groupby(['parClass','weekOfYear'],as_index=False)['saleCount'].agg({'parLast2WeekSaleCount_ptp':'ptp'})
    # parLastMonthSaleCount_ptp_o = train_test.groupby(['parClass','month'],as_index=False)['saleCount'].agg({'parLastMonthSaleCount_ptp':'ptp'})
    # parLastWeekSaleCount_ptp   = parLastWeekSaleCount_ptp_o.shift(1)
    # parLast2WeekSaleCount_ptp = parLast2WeekSaleCount_ptp_o.shift(2)
    # parLastMonthSaleCount_ptp = parLastMonthSaleCount_ptp_o.shift(1)
    # parLastWeekSaleCount_ptp.weekOfYear  = parLastWeekSaleCount_ptp_o.weekOfYear
    # parLast2WeekSaleCount_ptp.weekOfYear = parLast2WeekSaleCount_ptp_o.weekOfYear
    # parLastMonthSaleCount_ptp.month      = parLastMonthSaleCount_ptp_o.month
    # parLastWeekSaleCount_ptp.parClass =  parLastWeekSaleCount_ptp_o.parClass
    # parLast2WeekSaleCount_ptp.parClass = parLast2WeekSaleCount_ptp_o.parClass
    # parLastMonthSaleCount_ptp.parClass = parLastMonthSaleCount_ptp_o.parClass
    # parLastWeekSaleCount_ptp.parLastWeekSaleCount_ptp.fillna(0,inplace=True)
    # parLast2WeekSaleCount_ptp.parLast2WeekSaleCount_ptp.fillna(0,inplace=True)
    # parLastMonthSaleCount_ptp.parLastMonthSaleCount_ptp.fillna(0,inplace = True)

    # # 用于合并
    # parLastWeekSaleCount_median
    # parLast2WeekSaleCount_median
    # parLastMonthSaleCount_median


    # 合并 train_test
    tmp = pd.merge(train_test,lastWeekSaleCount_mean,on=['Class','weekOfYear'],how='left')
    # tmp = pd.merge(tmp,last2WeekSaleCount_mean,on=['Class','weekOfYear'],how='left')
    # tmp = pd.merge(tmp,lastMonthSaleCount_mean,on=['Class','month'],how='left')
    tmp = pd.merge(tmp,lastWeekSaleCount_median,on=['Class','weekOfYear'],how='left')
    # tmp = pd.merge(tmp,last2WeekSaleCount_median,on=['Class','weekOfYear'],how='left')
    tmp = pd.merge(tmp,lastMonthSaleCount_median,on=['Class','month'],how='left')

    tmp = pd.merge(tmp,lastWeekSaleCount_std,on=['Class','weekOfYear'],how='left')
    # tmp = pd.merge(tmp,last2WeekSaleCount_std,on=['Class','weekOfYear'],how='left')
    tmp = pd.merge(tmp,lastMonthSaleCount_std,on=['Class','month'],how='left')

    # tmp = pd.merge(tmp,parLastWeekSaleCount_mean,on=['parClass','weekOfYear'],how='left')
    # tmp = pd.merge(tmp,parLast2WeekSaleCount_mean,on=['parClass','weekOfYear'],how='left')
    # tmp = pd.merge(tmp,parLastMonthSaleCount_mean,on=['parClass','month'],how='left')
    # tmp = pd.merge(tmp,parLastWeekSaleCount_median,on=['parClass','weekOfYear'],how='left')
    # tmp = pd.merge(tmp,parLast2WeekSaleCount_median,on=['parClass','weekOfYear'],how='left')
    # tmp = pd.merge(tmp,parLastMonthSaleCount_median,on=['parClass','month'],how='left')

    # tmp = pd.merge(tmp,parLastWeekSaleCount_std,on=['parClass','weekOfYear'],how='left')
    # tmp = pd.merge(tmp,parLast2WeekSaleCount_std,on=['parClass','weekOfYear'],how='left')
    tmp = pd.merge(tmp,parLastMonthSaleCount_std,on=['parClass','month'],how='left')

    # print 'new added features:',np.setdiff1d(tmp.columns, train_test.columns)
    train_test = tmp.copy()
    return train_test

def get_roll_week_sale_feats(train_test):
    #子类
    # weekDayRatio
    lastMonthTotSaleCount_o = train_test.groupby(['Class','month'],as_index=False)['saleCount'].agg({'lastMonthTotSaleCount':'sum'})
    lastMonthTotSaleCount = lastMonthTotSaleCount_o.shift(1)
    lastMonthTotSaleCount['Class'] = lastMonthTotSaleCount_o['Class']
    lastMonthTotSaleCount['month'] = lastMonthTotSaleCount_o['month']
    lastMonthTotSaleCount.fillna(1,inplace=True)    # 缺失值处理
    lastMonthTotSaleCount['lastMonthTotSaleCount'][lastMonthTotSaleCount['lastMonthTotSaleCount'] == 0.0] = 1  #把分母设为1
    lastWeekDayTotSaleCount_o = train_test.groupby(['Class','month','dayOfWeek'],as_index=False)['saleCount'].agg({'lastWeekDayTotSaleCount':'sum'})
    lastWeekDayTotSaleCount = lastWeekDayTotSaleCount_o.shift(7)
    lastWeekDayTotSaleCount['Class']     = lastWeekDayTotSaleCount_o['Class']
    lastWeekDayTotSaleCount['dayOfWeek'] = lastWeekDayTotSaleCount_o['dayOfWeek']
    lastWeekDayTotSaleCount['month']     = lastWeekDayTotSaleCount_o['month']
    lastWeekDayTotSaleCount.fillna(1,inplace=True)    # 缺失值处理
    lastWeekDayTotSaleCount['lastWeekDayTotSaleCount'][lastWeekDayTotSaleCount['lastWeekDayTotSaleCount'] == 0.0] = 1  #把分母设为1
    lastWeekDayTotSaleCount = pd.merge(lastWeekDayTotSaleCount,lastMonthTotSaleCount,on=['Class','month'],how='left')
    lastWeekDayTotSaleCount.loc[:,'weekDayRatio'] = np.round(lastWeekDayTotSaleCount['lastWeekDayTotSaleCount'] / (1.0 * lastWeekDayTotSaleCount['lastMonthTotSaleCount']) ,4)

    # #用于合并
    # lastWeekDayTotSaleCount   # merge on Class and dayofWeek

    # weekOn1WeekRatio，weekOn2WeekRatio
    last1WeekSaleCount_o = train_test.groupby(['Class','dayOfYear'],as_index=False)['saleCount'].agg({'last1WeekSaleCount':'sum'})
    last1WeekSaleCount = last1WeekSaleCount_o.shift(7)
    last1WeekSaleCount['Class'] = last1WeekSaleCount_o['Class']
    last1WeekSaleCount['dayOfYear'] = last1WeekSaleCount_o['dayOfYear']
    last1WeekSaleCount['last1WeekSaleCount'].fillna(0,inplace=True)

    last2WeekSaleCount = last1WeekSaleCount_o.shift(14)
    last2WeekSaleCount.rename(columns={'last1WeekSaleCount':'last2WeekSaleCount'},inplace=True)
    last2WeekSaleCount['Class'] = last1WeekSaleCount_o['Class']
    last2WeekSaleCount['dayOfYear'] = last1WeekSaleCount_o['dayOfYear']
    last2WeekSaleCount['last2WeekSaleCount'].fillna(1,inplace=True)   #把分母设为1
    last2WeekSaleCount['last2WeekSaleCount'][last2WeekSaleCount['last2WeekSaleCount'] == 0.0] = 1  #把分母设为1

    last3WeekSaleCount = last1WeekSaleCount_o.shift(21)
    last3WeekSaleCount.rename(columns={'last1WeekSaleCount':'last3WeekSaleCount'},inplace=True)
    last3WeekSaleCount['Class'] = last1WeekSaleCount_o['Class']
    last3WeekSaleCount['dayOfYear'] = last1WeekSaleCount_o['dayOfYear']
    last3WeekSaleCount['last3WeekSaleCount'].fillna(0,inplace=True)   #把分母设为1
    last3WeekSaleCount['last3WeekSaleCount'][last3WeekSaleCount['last3WeekSaleCount'] == 0.0] = 1  #把分母设为1


    weekOnWeekRatio = pd.merge(last1WeekSaleCount,last2WeekSaleCount, on = ['Class','dayOfYear'], how='left')
    weekOnWeekRatio = pd.merge(weekOnWeekRatio,last3WeekSaleCount, on = ['Class','dayOfYear'], how='left')
    weekOnWeekRatio.loc[:,'weekOn1WeekRatio'] = np.round(weekOnWeekRatio['last1WeekSaleCount'] / (1.0 * weekOnWeekRatio['last2WeekSaleCount']) ,4)
    weekOnWeekRatio.loc[:,'weekOn2WeekRatio'] = np.round(weekOnWeekRatio['last1WeekSaleCount'] / (1.0 * weekOnWeekRatio['last3WeekSaleCount']) ,4)
    # #用于合并
    # weekOnWeekRatio


    #父类
    # parWeekDayRatio
    parLastMonthTotSaleCount_o = train_test.groupby(['parClass','month'],as_index=False)['saleCount'].agg({'parLastMonthTotSaleCount':'sum'})
    parLastMonthTotSaleCount = parLastMonthTotSaleCount_o.shift(1)
    parLastMonthTotSaleCount['parClass'] = parLastMonthTotSaleCount_o['parClass']
    parLastMonthTotSaleCount['month'] = parLastMonthTotSaleCount_o['month']
    parLastMonthTotSaleCount.fillna(1,inplace=True)    # 缺失值处理
    parLastMonthTotSaleCount['parLastMonthTotSaleCount'][parLastMonthTotSaleCount['parLastMonthTotSaleCount'] == 0.0] = 1  #把分母设为1
    parLastWeekDayTotSaleCount_o = train_test.groupby(['parClass','month','dayOfWeek'],as_index=False)['saleCount'].agg({'parLastWeekDayTotSaleCount':'sum'})
    parLastWeekDayTotSaleCount = parLastWeekDayTotSaleCount_o.shift(7)
    parLastWeekDayTotSaleCount['parClass']     = parLastWeekDayTotSaleCount_o['parClass']
    parLastWeekDayTotSaleCount['dayOfWeek'] = parLastWeekDayTotSaleCount_o['dayOfWeek']
    parLastWeekDayTotSaleCount['month']     = parLastWeekDayTotSaleCount_o['month']
    parLastWeekDayTotSaleCount.fillna(1,inplace=True)    # 缺失值处理
    parLastWeekDayTotSaleCount['parLastWeekDayTotSaleCount'][parLastWeekDayTotSaleCount['parLastWeekDayTotSaleCount'] == 0.0] = 1  #把分母设为1
    parLastWeekDayTotSaleCount = pd.merge(parLastWeekDayTotSaleCount,parLastMonthTotSaleCount,on=['parClass','month'],how='left')
    parLastWeekDayTotSaleCount.loc[:,'parWeekDayRatio'] = np.round(parLastWeekDayTotSaleCount['parLastWeekDayTotSaleCount'] / (1.0 * parLastWeekDayTotSaleCount['parLastMonthTotSaleCount']) ,4)

    # #用于合并
    # parLastWeekDayTotSaleCount   # merge on parClass and dayofWeek


    # parWeekOn1WeekRatio，parWeekOn2WeekRatio
    parLast1WeekSaleCount_o = train_test.groupby(['parClass','dayOfYear'],as_index=False)['saleCount'].agg({'parLast1WeekSaleCount':'sum'})
    parLast1WeekSaleCount = parLast1WeekSaleCount_o.shift(7)
    parLast1WeekSaleCount['parClass'] = parLast1WeekSaleCount_o['parClass']
    parLast1WeekSaleCount['dayOfYear'] = parLast1WeekSaleCount_o['dayOfYear']
    parLast1WeekSaleCount['parLast1WeekSaleCount'].fillna(0,inplace=True)

    parLast2WeekSaleCount = parLast1WeekSaleCount_o.shift(14)
    parLast2WeekSaleCount.rename(columns={'parLast1WeekSaleCount':'parLast2WeekSaleCount'},inplace=True)
    parLast2WeekSaleCount['parClass'] = parLast1WeekSaleCount_o['parClass']
    parLast2WeekSaleCount['dayOfYear'] = parLast1WeekSaleCount_o['dayOfYear']
    parLast2WeekSaleCount['parLast2WeekSaleCount'].fillna(1,inplace=True)   #把分母设为1
    parLast2WeekSaleCount['parLast2WeekSaleCount'][parLast2WeekSaleCount['parLast2WeekSaleCount'] == 0.0] = 1  #把分母设为1

    parLast3WeekSaleCount = parLast1WeekSaleCount_o.shift(21)
    parLast3WeekSaleCount.rename(columns={'parLast1WeekSaleCount':'parLast3WeekSaleCount'},inplace=True)
    parLast3WeekSaleCount['parClass'] = parLast1WeekSaleCount_o['parClass']
    parLast3WeekSaleCount['dayOfYear'] = parLast1WeekSaleCount_o['dayOfYear']
    parLast3WeekSaleCount['parLast3WeekSaleCount'].fillna(0,inplace=True)   #把分母设为1
    parLast3WeekSaleCount['parLast3WeekSaleCount'][parLast3WeekSaleCount['parLast3WeekSaleCount'] == 0.0] = 1  #把分母设为1


    parWeekOnWeekRatio = pd.merge(parLast1WeekSaleCount,parLast2WeekSaleCount, on = ['parClass','dayOfYear'], how='left')
    parWeekOnWeekRatio = pd.merge(parWeekOnWeekRatio,parLast3WeekSaleCount, on = ['parClass','dayOfYear'], how='left')
    parWeekOnWeekRatio.loc[:,'parWeekOn1WeekRatio'] = np.round(parWeekOnWeekRatio['parLast1WeekSaleCount'] / (1.0 * parWeekOnWeekRatio['parLast2WeekSaleCount']) ,4)
    parWeekOnWeekRatio.loc[:,'parWeekOn2WeekRatio'] = np.round(parWeekOnWeekRatio['parLast1WeekSaleCount'] / (1.0 * parWeekOnWeekRatio['parLast3WeekSaleCount']) ,4)
    # #用于合并
    # parWeekOnWeekRatio

    # day3OoverWeek3TotRatio
    # 类别上周，上2周，上3周,上4周总销量
    lastWeekSaleCount_o = train_test.groupby(['Class','weekOfYear'],as_index=False)['saleCount'].agg({'lastWeekSaleCount':'sum'})
    last2WeekSaleCount_o = train_test.groupby(['Class','weekOfYear'],as_index=False)['saleCount'].agg({'last2WeekSaleCount':'sum'})
    last3WeekSaleCount_o = train_test.groupby(['Class','weekOfYear'],as_index=False)['saleCount'].agg({'last3WeekSaleCount':'sum'})
    last4WeekSaleCount_o = train_test.groupby(['Class','weekOfYear'],as_index=False)['saleCount'].agg({'last4WeekSaleCount':'sum'})

    lastWeekSaleCount = lastWeekSaleCount_o.shift(1)
    last2WeekSaleCount = last2WeekSaleCount_o.shift(2)
    last3WeekSaleCount = last3WeekSaleCount_o.shift(3)
    last4WeekSaleCount = last4WeekSaleCount_o.shift(4)

    lastWeekSaleCount.weekOfYear = lastWeekSaleCount_o.weekOfYear
    last2WeekSaleCount.weekOfYear = last2WeekSaleCount_o.weekOfYear
    last3WeekSaleCount.weekOfYear = last3WeekSaleCount_o.weekOfYear
    last4WeekSaleCount.weekOfYear = last4WeekSaleCount_o.weekOfYear

    lastWeekSaleCount.Class = lastWeekSaleCount_o.Class
    last2WeekSaleCount.Class = last2WeekSaleCount_o.Class
    last3WeekSaleCount.Class = last3WeekSaleCount_o.Class
    last4WeekSaleCount.Class = last4WeekSaleCount_o.Class

    lastWeekSaleCount['lastWeekSaleCount'].fillna(0,inplace=True)
    last2WeekSaleCount['last2WeekSaleCount'].fillna(1,inplace=True)
    last3WeekSaleCount['last3WeekSaleCount'].fillna(1,inplace=True)
    last4WeekSaleCount['last4WeekSaleCount'].fillna(1,inplace=True)

    day3OoverWeek3Tot = pd.merge(lastWeekSaleCount, last2WeekSaleCount, on=['Class','weekOfYear'],how='left')
    day3OoverWeek3Tot = pd.merge(day3OoverWeek3Tot, last3WeekSaleCount, on=['Class','weekOfYear'],how='left')
    day3OoverWeek3Tot = pd.merge(day3OoverWeek3Tot, last4WeekSaleCount, on=['Class','weekOfYear'],how='left')
    day3OoverWeek3Tot.loc[:,'day3OoverWeek3TotRatio'] =  np.round(day3OoverWeek3Tot['lastWeekSaleCount'] /
                                                                  (1.0 * (day3OoverWeek3Tot['last2WeekSaleCount'] + day3OoverWeek3Tot['last3WeekSaleCount'] + day3OoverWeek3Tot['last4WeekSaleCount'])) ,4)
    # #用于合并
    # day3OoverWeek3Tot

    # lastWeekDayTotSaleCount
    # weekOnWeekRatio # Class,dayOfYear
    # parLastWeekDayTotSaleCount #parClass month dayOfWeek
    # parWeekOnWeekRatio
    day3OoverWeek3Tot #Class weekOfYear
    #开始合并
    # 合并 train_test
    del lastWeekDayTotSaleCount['lastWeekDayTotSaleCount' ],lastWeekDayTotSaleCount['lastMonthTotSaleCount' ]
    del weekOnWeekRatio['last1WeekSaleCount'],weekOnWeekRatio['last2WeekSaleCount'],weekOnWeekRatio['last3WeekSaleCount']
    del parLastWeekDayTotSaleCount['parLastWeekDayTotSaleCount' ],   parLastWeekDayTotSaleCount['parLastMonthTotSaleCount' ]
    del parWeekOnWeekRatio['parLast1WeekSaleCount'],parWeekOnWeekRatio['parLast2WeekSaleCount'],parWeekOnWeekRatio['parLast3WeekSaleCount']
    del day3OoverWeek3Tot['lastWeekSaleCount'],day3OoverWeek3Tot['last2WeekSaleCount'],day3OoverWeek3Tot['last3WeekSaleCount'],day3OoverWeek3Tot['last4WeekSaleCount']
    tmp = pd.merge(train_test,lastWeekDayTotSaleCount,on=['Class','month','dayOfWeek'],how='left')
    tmp = pd.merge(tmp,weekOnWeekRatio,on=['Class','dayOfYear'],how='left')
    tmp = pd.merge(tmp,parLastWeekDayTotSaleCount,on=['parClass','month','dayOfWeek'],how='left')
    tmp = pd.merge(tmp,parWeekOnWeekRatio,on=['parClass','dayOfYear'],how='left')
    tmp = pd.merge(tmp,day3OoverWeek3Tot,on=['Class','weekOfYear'],how='left')

    # print 'new added features:',np.setdiff1d(tmp.columns, train_test.columns)
    train_test = tmp.copy()

    return train_test

def get_roll_diff_feats(train_test):
    #dayOn1DayDiff,dayOn2DayDiff,dayOn7DayDiff,dayOn14DayDiff
    last7DaysSaleCount_o = train_test.groupby(['Class','dayOfYear'],as_index=False)['saleCount'].agg({'last7DaysSaleCount':'sum'})
    last7DaysSaleCount = last7DaysSaleCount_o.shift(7)
    last7DaysSaleCount['Class'] = last7DaysSaleCount_o['Class']
    last7DaysSaleCount['dayOfYear'] = last7DaysSaleCount_o['dayOfYear']
    last7DaysSaleCount['last7DaysSaleCount'].fillna(0,inplace=True)

    last8DaysSaleCount_o = train_test.groupby(['Class','dayOfYear'],as_index=False)['saleCount'].agg({'last8DaysSaleCount':'sum'})
    last8DaysSaleCount = last8DaysSaleCount_o.shift(8)
    last8DaysSaleCount['Class'] = last8DaysSaleCount_o['Class']
    last8DaysSaleCount['dayOfYear'] = last8DaysSaleCount_o['dayOfYear']
    last8DaysSaleCount['last8DaysSaleCount'].fillna(0,inplace=True)

    last9DaysSaleCount_o = train_test.groupby(['Class','dayOfYear'],as_index=False)['saleCount'].agg({'last9DaysSaleCount':'sum'})
    last9DaysSaleCount = last9DaysSaleCount_o.shift(9)
    last9DaysSaleCount['Class'] = last9DaysSaleCount_o['Class']
    last9DaysSaleCount['dayOfYear'] = last9DaysSaleCount_o['dayOfYear']
    last9DaysSaleCount['last9DaysSaleCount'].fillna(0,inplace=True)

    last10DaysSaleCount_o = train_test.groupby(['Class','dayOfYear'],as_index=False)['saleCount'].agg({'last10DaysSaleCount':'sum'})
    last10DaysSaleCount = last10DaysSaleCount_o.shift(10)
    last10DaysSaleCount['Class'] = last10DaysSaleCount_o['Class']
    last10DaysSaleCount['dayOfYear'] = last10DaysSaleCount_o['dayOfYear']
    last10DaysSaleCount['last10DaysSaleCount'].fillna(0,inplace=True)

    last11DaysSaleCount_o = train_test.groupby(['Class','dayOfYear'],as_index=False)['saleCount'].agg({'last11DaysSaleCount':'sum'})
    last11DaysSaleCount = last11DaysSaleCount_o.shift(11)
    last11DaysSaleCount['Class'] = last11DaysSaleCount_o['Class']
    last11DaysSaleCount['dayOfYear'] = last11DaysSaleCount_o['dayOfYear']
    last11DaysSaleCount['last11DaysSaleCount'].fillna(0,inplace=True)

    last12DaysSaleCount_o = train_test.groupby(['Class','dayOfYear'],as_index=False)['saleCount'].agg({'last12DaysSaleCount':'sum'})
    last12DaysSaleCount = last12DaysSaleCount_o.shift(12)
    last12DaysSaleCount['Class'] = last12DaysSaleCount_o['Class']
    last12DaysSaleCount['dayOfYear'] = last12DaysSaleCount_o['dayOfYear']
    last12DaysSaleCount['last12DaysSaleCount'].fillna(0,inplace=True)

    last13DaysSaleCount_o = train_test.groupby(['Class','dayOfYear'],as_index=False)['saleCount'].agg({'last13DaysSaleCount':'sum'})
    last13DaysSaleCount = last13DaysSaleCount_o.shift(13)
    last13DaysSaleCount['Class'] = last13DaysSaleCount_o['Class']
    last13DaysSaleCount['dayOfYear'] = last13DaysSaleCount_o['dayOfYear']
    last13DaysSaleCount['last13DaysSaleCount'].fillna(0,inplace=True)

    last14DaysSaleCount_o = train_test.groupby(['Class','dayOfYear'],as_index=False)['saleCount'].agg({'last14DaysSaleCount':'sum'})
    last14DaysSaleCount = last14DaysSaleCount_o.shift(14)
    last14DaysSaleCount['Class'] = last14DaysSaleCount_o['Class']
    last14DaysSaleCount['dayOfYear'] = last14DaysSaleCount_o['dayOfYear']
    last14DaysSaleCount['last14DaysSaleCount'].fillna(0,inplace=True)

    last21DaysSaleCount_o = train_test.groupby(['Class','dayOfYear'],as_index=False)['saleCount'].agg({'last21DaysSaleCount':'sum'})
    last21DaysSaleCount = last21DaysSaleCount_o.shift(21)
    last21DaysSaleCount['Class'] = last21DaysSaleCount_o['Class']
    last21DaysSaleCount['dayOfYear'] = last21DaysSaleCount_o['dayOfYear']
    last21DaysSaleCount['last21DaysSaleCount'].fillna(0,inplace=True)

    diff = pd.merge(last7DaysSaleCount,last8DaysSaleCount,on = ['Class','dayOfYear'], how='left')
    diff = pd.merge(diff,last9DaysSaleCount,on = ['Class','dayOfYear'], how='left')
    diff = pd.merge(diff,last10DaysSaleCount,on = ['Class','dayOfYear'], how='left')
    diff = pd.merge(diff,last11DaysSaleCount,on = ['Class','dayOfYear'], how='left')
    diff = pd.merge(diff,last12DaysSaleCount,on = ['Class','dayOfYear'], how='left')
    diff = pd.merge(diff,last13DaysSaleCount,on = ['Class','dayOfYear'], how='left')
    diff = pd.merge(diff,last14DaysSaleCount,on = ['Class','dayOfYear'], how='left')
    diff = pd.merge(diff,last21DaysSaleCount,on = ['Class','dayOfYear'], how='left')

    #前三天均值
    diff.loc[:,'last3Days_mean'] = (diff['last7DaysSaleCount'] + diff['last8DaysSaleCount'] + diff['last9DaysSaleCount'])/3.0
    diff.loc[:,'last6Days_mean'] = (diff['last7DaysSaleCount'] + diff['last8DaysSaleCount'] + diff['last9DaysSaleCount'] + diff['last10DaysSaleCount'] + diff['last11DaysSaleCount'] + diff['last12DaysSaleCount'])/6.0
    diff.loc[:,'last3InterDays_mean'] = (diff['last10DaysSaleCount'] + diff['last11DaysSaleCount'] + diff['last12DaysSaleCount'])/3.0

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
    tmp = pd.merge(train_test,diff[['Class','dayOfYear','dayOn1DayDiff','dayOn2DayDiff','dayOn3DayDiff','dayOn4DayDiff','dayOn5DayDiff','dayOn6DayDiff','dayOn7DayDiff','dayOn14DayDiff','last3Days_mean','last6Days_mean','last3InterDays_mean']],on=['Class','dayOfYear'],how='left')
    # print 'new added features:',np.setdiff1d(tmp.columns, train_test.columns)
    train_test = tmp.copy()
    return train_test

def get_roll_feats(train_test):
    l_feat_original = train_test.columns
    print "Start extract rolling features....."
    train_test = get_roll_hot_index_feats(train_test)
    print "Roll hot index features done."
    train_test = get_roll_price_feats(train_test)
    print "Roll price features done."
    train_test = get_roll_week_sale_feats(train_test)
    print "Roll week sale features done."
    train_test = get_roll_diff_feats(train_test)
    print "Roll differentiate features done."
    print "Rolling features done. "
    l_feat_new = train_test.columns
    l_roll_feats = np.setdiff1d(l_feat_new,l_feat_original)
    return train_test,l_roll_feats
