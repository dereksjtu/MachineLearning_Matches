import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
from scipy.stats import mode
import warnings
# 根据历史的每天销量预测未来的销量，之前稍微统计，每个月的总量基本是不变的。
# 多少人曾爱慕你年轻的容颜，可是谁有能承受岁月无情的变迁
# 效果其实和求均值差不多
warnings.filterwarnings('ignore')

def mode_function(df):
    df = df.astype(int)
    counts = mode(df)
    return counts[0][0]

def score(y_test,y_pred):
    print(y_test)
    print(y_pred)
    return 1.0 / (1.0 + np.sqrt(mean_squared_error(y_test, y_pred)))

train = pd.read_csv('./训练集.csv',encoding='gbk')
print('训练集总数',train.shape)
print('字段',train.columns)
sub = pd.read_csv('./用户提交.csv',encoding='gbk')
sub['销售日期_D'] = sub['日期'].map(lambda x:str(x)[-2:])
sub['销售日期_D'] = sub['销售日期_D'] .astype(int)
result = sub.copy()
# print(sub)

sub_train = sub[['编码','销售日期_D','销量']]

train['销售日期_M'] = train['销售日期'].map(lambda x:int(str(x)[5:6]))
train['销售日期_D'] = train['销售日期'].map(lambda x:int(str(x)[6:]))
type_list = list(train['商品类型'].unique())
train['是否促销'] = train['是否促销'].map(list(train['是否促销'].unique()).index)
train['商品类型'] = train['商品类型'].map(type_list.index)
train['单位'] = train['单位'].map(list(train['单位'].unique()).index)

train.drop(['大类名称','中类名称','小类名称','销售月份','商品编码','小类编码','custid','销售日期'],axis=1,inplace=True)
print('月份总数',list(train['销售日期_M'].unique()))
# print(train)
#
print(train.describe())

# train['is_buy'] = train['销售数量'] > 0
train['is_buy'] = 1
train['is_buy'] = train['is_buy'].astype(int)

train_1 = train[train['销售日期_M'] == 1 ][['中类编码','销售日期_D','is_buy']]
train_1_count = train_1.groupby(['销售日期_D','中类编码'],as_index=False)['is_buy'].sum()
train_1_count.rename(columns = {'is_buy':'is_buy_1'},inplace=True)
train_2 = train[train['销售日期_M'] == 2 ][['中类编码','销售日期_D','is_buy']]
train_2_count = train_2.groupby(['销售日期_D','中类编码'],as_index=False)['is_buy'].sum()
train_2_count.rename(columns = {'is_buy':'is_buy_2'},inplace=True)

train_3 = train[train['销售日期_M'] == 3 ][['中类编码','销售日期_D','is_buy']]
train_3_count = train_3.groupby(['销售日期_D','中类编码'],as_index=False)['is_buy'].sum()
train_3_count.rename(columns = {'is_buy':'is_buy_3'},inplace=True)

train_4 = train[train['销售日期_M'] == 4 ][['中类编码','销售日期_D','is_buy']]
train_4_count = train_4.groupby(['销售日期_D','中类编码'],as_index=False)['is_buy'].sum()
train_4_count.rename(columns = {'is_buy':'is_buy_4'},inplace=True)

# print(train_1_count.shape)
# print(train_2_count.shape)
# print(train_3_count.shape)
# print(train_4_count.shape)

sub_train.rename(columns={'编码':'中类编码'},inplace=True)
train_median_class = pd.merge(sub_train,train_1_count,on=['销售日期_D','中类编码'],how='left')
train_median_class = pd.merge(train_median_class,train_2_count,on=['销售日期_D','中类编码'],how='left')
train_median_class = pd.merge(train_median_class,train_3_count,on=['销售日期_D','中类编码'],how='left')
train_median_class = pd.merge(train_median_class,train_4_count,on=['销售日期_D','中类编码'],how='left')

# 一些基础特征
train_1_base = train[train['销售日期_M'] == 1 ][['中类编码','销售日期_D','是否促销']]
train_1_count_base = train_1_base.groupby(['销售日期_D','中类编码'],as_index=False)['是否促销'].sum()
train_1_count_base.rename(columns = {'是否促销':'是否促销_1'},inplace=True)
train_2_base = train[train['销售日期_M'] == 2 ][['中类编码','销售日期_D','是否促销']]
train_2_count_base = train_2_base.groupby(['销售日期_D','中类编码'],as_index=False)['是否促销'].sum()
train_2_count_base.rename(columns = {'是否促销':'是否促销_2'},inplace=True)

train_3_base = train[train['销售日期_M'] == 3 ][['中类编码','销售日期_D','是否促销']]
train_3_count_base = train_3_base.groupby(['销售日期_D','中类编码'],as_index=False)['是否促销'].sum()
train_3_count_base.rename(columns = {'是否促销':'是否促销_3'},inplace=True)

train_4_base = train[train['销售日期_M'] == 4 ][['中类编码','销售日期_D','是否促销']]
train_4_count_base = train_4_base.groupby(['销售日期_D','中类编码'],as_index=False)['是否促销'].sum()
train_4_count_base.rename(columns = {'是否促销':'是否促销_4'},inplace=True)

train_median_class = pd.merge(train_median_class,train_1_count_base,on=['销售日期_D','中类编码'],how='left')
train_median_class = pd.merge(train_median_class,train_2_count_base,on=['销售日期_D','中类编码'],how='left')
train_median_class = pd.merge(train_median_class,train_3_count_base,on=['销售日期_D','中类编码'],how='left')
train_median_class = pd.merge(train_median_class,train_4_count_base,on=['销售日期_D','中类编码'],how='left')

train_median_class.rename(columns = {'中类编码':'编码'},inplace=True)

train_median_class['is_buy_1'] = train_median_class['is_buy_1'].fillna(train_median_class['is_buy_1'].dropna().median()).astype(int)
train_median_class['is_buy_2'] = train_median_class['is_buy_2'].fillna(train_median_class['is_buy_2'].dropna().mean()).astype(int)
train_median_class['is_buy_3'] = train_median_class['is_buy_3'].fillna(0).astype(int)
train_median_class['is_buy_4'] = train_median_class['is_buy_4'].fillna(0).astype(int)

train_median_class = train_median_class[train_median_class['编码'] >= 1000 ]

train_median_class = train_median_class[train_median_class['销售日期_D']<=30]
# print(train_median_class)

print(train_median_class.info())

train_1 = train[train['销售日期_M'] == 1 ][['大类编码','销售日期_D','is_buy']]
train_1_count = train_1.groupby(['销售日期_D','大类编码'],as_index=False)['is_buy'].sum()
train_1_count.rename(columns = {'is_buy':'is_buy_1'},inplace=True)
train_2 = train[train['销售日期_M'] == 2 ][['大类编码','销售日期_D','is_buy']]
train_2_count = train_2.groupby(['销售日期_D','大类编码'],as_index=False)['is_buy'].sum()
train_2_count.rename(columns = {'is_buy':'is_buy_2'},inplace=True)

train_3 = train[train['销售日期_M'] == 3 ][['大类编码','销售日期_D','is_buy']]
train_3_count = train_3.groupby(['销售日期_D','大类编码'],as_index=False)['is_buy'].sum()
train_3_count.rename(columns = {'is_buy':'is_buy_3'},inplace=True)

train_4 = train[train['销售日期_M'] == 4 ][['大类编码','销售日期_D','is_buy']]
train_4_count = train_4.groupby(['销售日期_D','大类编码'],as_index=False)['is_buy'].sum()
train_4_count.rename(columns = {'is_buy':'is_buy_4'},inplace=True)

print(train_1_count.shape)
print(train_2_count.shape)
print(train_3_count.shape)
print(train_4_count.shape)

sub_train.rename(columns={'中类编码':'大类编码'},inplace=True)

train_big_class = pd.merge(sub_train,train_1_count,on=['销售日期_D','大类编码'],how='left')
train_big_class = pd.merge(train_big_class,train_2_count,on=['销售日期_D','大类编码'],how='left')
train_big_class = pd.merge(train_big_class,train_3_count,on=['销售日期_D','大类编码'],how='left')
train_big_class = pd.merge(train_big_class,train_4_count,on=['销售日期_D','大类编码'],how='left')


# 一些基础特征
train_1_base = train[train['销售日期_M'] == 1 ][['大类编码','销售日期_D','是否促销']]
train_1_count_base = train_1_base.groupby(['销售日期_D','大类编码'],as_index=False)['是否促销'].sum()
train_1_count_base.rename(columns = {'是否促销':'是否促销_1'},inplace=True)
train_2_base = train[train['销售日期_M'] == 2 ][['大类编码','销售日期_D','是否促销']]
train_2_count_base = train_2_base.groupby(['销售日期_D','大类编码'],as_index=False)['是否促销'].sum()
train_2_count_base.rename(columns = {'是否促销':'是否促销_2'},inplace=True)

train_3_base = train[train['销售日期_M'] == 3 ][['大类编码','销售日期_D','是否促销']]
train_3_count_base = train_3_base.groupby(['销售日期_D','大类编码'],as_index=False)['是否促销'].sum()
train_3_count_base.rename(columns = {'是否促销':'是否促销_3'},inplace=True)

train_4_base = train[train['销售日期_M'] == 4 ][['大类编码','销售日期_D','是否促销']]
train_4_count_base = train_4_base.groupby(['销售日期_D','大类编码'],as_index=False)['是否促销'].sum()
train_4_count_base.rename(columns = {'是否促销':'是否促销_4'},inplace=True)

train_big_class = pd.merge(train_big_class,train_1_count_base,on=['销售日期_D','大类编码'],how='left')
train_big_class = pd.merge(train_big_class,train_2_count_base,on=['销售日期_D','大类编码'],how='left')
train_big_class = pd.merge(train_big_class,train_3_count_base,on=['销售日期_D','大类编码'],how='left')
train_big_class = pd.merge(train_big_class,train_4_count_base,on=['销售日期_D','大类编码'],how='left')

train_big_class.rename(columns = {'大类编码':'编码'},inplace=True)


train_big_class['is_buy_1'] = train_big_class['is_buy_1'].fillna(train_big_class['is_buy_1'].dropna().median()).astype(int)
train_big_class['is_buy_2'] = train_big_class['is_buy_2'].fillna(train_big_class['is_buy_2'].dropna().mean()).astype(int)
train_big_class['is_buy_3'] = train_big_class['is_buy_3'].fillna(0).astype(int)
train_big_class['is_buy_4'] = train_big_class['is_buy_4'].fillna(0).astype(int)

train_big_class = train_big_class[train_big_class['销售日期_D']<=30]
train_big_class = train_big_class[train_big_class['编码'] < 1000 ]
# print(train_big_class)

print(train_big_class.info())

train = pd.concat([train_median_class,train_big_class])
# 单独选取了中类，去掉这个就是全部需要预测的
train = train[train['编码']>=1000]
train = train.fillna(0)
train = train.astype(int)
# print(train)


import lightgbm as lgb
train_lgb = train[['编码','销售日期_D','is_buy_1','is_buy_2','is_buy_3','是否促销_1','是否促销_2']]
train_lgb['base_1'] = train_lgb['is_buy_2'] / (1 / (train_lgb['销售日期_D']))
train_lgb['mean'] = (train_lgb['is_buy_1'] + train_lgb['is_buy_2']) / 2
train_lgb['mean'] = train_lgb['mean'].astype(int)
train_lgb['diff'] = train_lgb['is_buy_2'] - train_lgb['is_buy_1']
print(train_lgb.head())
train_label = train_lgb.pop('is_buy_3')
train_label = train_label.values
train_ = train_lgb.values

test_lgb = train[['编码','销售日期_D','is_buy_2','is_buy_3','is_buy_4','是否促销_2','是否促销_3']]
test_lgb['base_1'] = test_lgb['is_buy_3'] / (1 / (test_lgb['销售日期_D']))
test_lgb['mean'] = (test_lgb['is_buy_2'] + test_lgb['is_buy_3']) / 2
test_lgb['mean'] = test_lgb['mean'].astype(int)
test_lgb['diff'] = test_lgb['is_buy_3'] - test_lgb['is_buy_2']

test_label = test_lgb.pop('is_buy_4')
test_label = test_label.values
test = test_lgb.values

print('Start training...')
# train
lgb_train = lgb.Dataset(train_, train_label)
lgb_eval = lgb.Dataset(test, test_label, reference=lgb_train)


params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 64,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

def rmse_d(y,d):
    c=d.get_label()
    result= 1.0 / (1.0 + np.sqrt(mean_squared_error(y, c)))
    return "自定义cost函数-",-result,False


gbm = lgb.train(params,
                lgb_train,
                num_boost_round=3000,
                feval=rmse_d,
                valid_sets=lgb_eval,
                early_stopping_rounds = 20)

sub_lgb = train[['编码','销售日期_D','is_buy_3','is_buy_4','是否促销_3','是否促销_4']]


sub_lgb['base_1'] = sub_lgb['is_buy_4'] / (1 / (sub_lgb['销售日期_D']))


sub_lgb['mean'] = (sub_lgb['is_buy_3'] + sub_lgb['is_buy_4']) / 2
sub_lgb['mean'] = sub_lgb['mean'].astype(int)
sub_lgb['diff'] = sub_lgb['is_buy_4'] - sub_lgb['is_buy_3']

sub = sub_lgb.values

result_lgb = gbm.predict(sub, num_iteration=gbm.best_iteration)
sub_ = pd.DataFrame({u'销量':list(result_lgb)})
sub_['销量'] = sub_['销量']
sub_['销量'] = sub_['销量'].astype(int)

print(sub_)
result = result[['编码','日期']]
# 单独看了看中类 去掉大于10000 就是全部需要预测的
result = result[result['编码']>=1000]
result = result.reset_index()
result_sub_ = pd.concat([result[['编码','日期']],sub_['销量']],axis=1)
result_sub_.to_csv('./sample_1.csv',index=False)

# [33]	valid_0's rmse: 9.09115  == 0.09 == 0.1309
# [33]	valid_0's rmse: 9.09115  == 0.101873 == 0.1317
# [33]	valid_0's rmse: 9.09115  == 0.107873 == 0.1337
