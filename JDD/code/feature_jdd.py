# -*- coding:utf-8 -*-
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Handle table like and matrices
import pandas as pd
import numpy as np

# Modeling Helper
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

# Configure visualization
mpl.style.use('ggplot')

t_login_path      = '../input/t_login.csv'
t_login_test_path = '../input/t_login_test.csv'
t_trade_path      = '../input/t_trade.csv'
t_trade_test_path = '../input/t_trade_test.csv'


login_tr = pd.read_csv(t_login_path)
login_te = pd.read_csv(t_login_test_path)
trade_tr = pd.read_csv(t_trade_path)
trade_te = pd.read_csv(t_trade_test_path)

def reshape_data(login_tr,trade_tr,login_te,trade_te):
    login_tr.rename(columns={'time':'login_time'},inplace=True)
    trade_tr.rename(columns={'time':'trade_time'},inplace=True)
    login_tr.sort_values(by=['id','login_time'],inplace=True)
    trade_tr.sort_values(by=['id','trade_time'],inplace=True)
    login_te.rename(columns={'time':'login_time'},inplace=True)
    trade_te.rename(columns={'time':'trade_time'},inplace=True)
    login_te.sort_values(by=['id','login_time'],inplace=True)
    trade_te.sort_values(by=['id','trade_time'],inplace=True)
    login_tr['login_time'] = pd.to_datetime(login_tr['login_time'])
    trade_tr['trade_time'] = pd.to_datetime(trade_tr['trade_time'])
    login_te['login_time'] = pd.to_datetime(login_te['login_time'])
    trade_te['trade_time'] = pd.to_datetime(trade_te['trade_time'])
    return login_tr,trade_tr,login_te,trade_te

def log_train_combine(login, trade):
    log_trade = pd.merge(trade[['id','trade_time','is_risk']],login, on=['id'],how='left')
    return log_trade

def get_legacy_user_info(log_trade):
    log_trade['trade_time'] = pd.to_datetime(log_trade['trade_time'])
    log_trade['login_time'] = pd.to_datetime(log_trade['login_time'])
    log_trade.loc[:,'trade_log_diff'] = log_trade['trade_time'] - log_trade['login_time']
    log_trade['trade_log_diff'] = log_trade['trade_log_diff'].map(lambda x: x.total_seconds())
    log_trade_legacy = log_trade[log_trade['trade_log_diff'] >= 0]
    # warm = log_trade_legacy.groupby(['id','trade_time'],as_index=False)['login_time'].count()
    # trade = trade_tr.copy()
    # trade.rename(columns={'time':'trade_time'},inplace=True)
    # trade['trade_time'] = pd.to_datetime(trade['trade_time'])
    # trade = pd.merge(trade,warm,on=['id','trade_time'],how='left')
    # trade_tr_cold = trade[trade['login_time'].isnull()]
    # trade_tr_cold.sort_values(by=['id','trade_time'],inplace=True)
    # trade_tr_cold
    return log_trade_legacy

def get_time_split(log_trade_legacy):
    # 1min
    log_trade_legacy.loc[:,'trade_log_diff_1min'] = log_trade_legacy['trade_log_diff'] <= 60
    log_trade_legacy['trade_log_diff_1min'] = log_trade_legacy['trade_log_diff_1min'].astype('int')
    # 2min
    log_trade_legacy.loc[:,'trade_log_diff_2min'] = log_trade_legacy['trade_log_diff'] <= 2*60
    log_trade_legacy['trade_log_diff_2min'] = log_trade_legacy['trade_log_diff_1min'].astype('int')
    #5min
    log_trade_legacy.loc[:,'trade_log_diff_5min'] = log_trade_legacy['trade_log_diff'] < 5*60
    log_trade_legacy['trade_log_diff_5min'] = log_trade_legacy['trade_log_diff_1min'].astype('int')
    #10min
    log_trade_legacy.loc[:,'trade_log_diff_10min'] = log_trade_legacy['trade_log_diff'] < 10*60
    log_trade_legacy['trade_log_diff_10min'] = log_trade_legacy['trade_log_diff_1min'].astype('int')
    #15min
    log_trade_legacy.loc[:,'trade_log_diff_15min'] = log_trade_legacy['trade_log_diff'] < 15*60
    log_trade_legacy['trade_log_diff_15min'] = log_trade_legacy['trade_log_diff_1min'].astype('int')
    # 30min
    log_trade_legacy.loc[:,'trade_log_diff_30min'] = log_trade_legacy['trade_log_diff'] < 30*60
    log_trade_legacy['trade_log_diff_30min'] = log_trade_legacy['trade_log_diff_1min'].astype('int')
    #60min
    log_trade_legacy.loc[:,'trade_log_diff_60min'] = log_trade_legacy['trade_log_diff'] < 60*60
    log_trade_legacy['trade_log_diff_60min'] = log_trade_legacy['trade_log_diff_1min'].astype('int')
    #2h
    log_trade_legacy.loc[:,'trade_log_diff_120min'] = log_trade_legacy['trade_log_diff'] < 2*60*60
    log_trade_legacy['trade_log_diff_120min'] = log_trade_legacy['trade_log_diff_1min'].astype('int')
    #5h
    log_trade_legacy.loc[:,'trade_log_diff_300min'] = log_trade_legacy['trade_log_diff'] < 5*60*60
    log_trade_legacy['trade_log_diff_300min'] = log_trade_legacy['trade_log_diff_1min'].astype('int')
    return log_trade_legacy

def get_count_feats_1min(trade,log_trade_legacy):
    tmp_legacy = log_trade_legacy[log_trade_legacy['trade_log_diff_1min'] == 1].copy()
    coord_city = tmp_legacy.groupby(['id','trade_time'])['city'].apply(lambda x: len(x.unique())).reset_index()
    coord_city.rename(columns={'city':'city_diff_count_1min'},inplace=True)

    coord_login = tmp_legacy.groupby(['id','trade_time'])['login_time'].apply(lambda x: len(x.unique())).reset_index()
    coord_login.rename(columns={'login_time':'login_diff_count_1min'},inplace=True)

    coord_device = tmp_legacy.groupby(['id','trade_time'])['device'].apply(lambda x: len(x.unique())).reset_index()
    coord_device.rename(columns={'device':'device_diff_count_1min'},inplace=True)

    coord_timelong = tmp_legacy.groupby(['id','trade_time'])['timelong'].apply(lambda x: len(x.unique())).reset_index()
    coord_timelong.rename(columns={'timelong':'timelong_diff_count_1min'},inplace=True)
    coord_timelong_std = tmp_legacy.groupby(['id','trade_time'],as_index=False)['timelong'].agg({'timelong_std_1min':'std'})
    coord_timelong_std.fillna(0,inplace=True) #有的序列只有一个登陆时长
    coord_timelong_mean = tmp_legacy.groupby(['id','trade_time'],as_index=False)['timelong'].agg({'timelong_mean_1min':'mean'})

    coord_logfrom = tmp_legacy.groupby(['id','trade_time'])['log_from'].apply(lambda x: len(x.unique())).reset_index()
    coord_logfrom.rename(columns={'log_from':'logfrom_diff_count_1min'},inplace=True)

    coord_ip = tmp_legacy.groupby(['id','trade_time'])['ip'].apply(lambda x: len(x.unique())).reset_index()
    coord_ip.rename(columns={'ip':'ip_diff_count_1min'},inplace=True)

    coord_result = tmp_legacy.groupby(['id','trade_time'])['result'].apply(lambda x: len(x.unique())).reset_index()
    coord_result.rename(columns={'result':'result_diff_count_1min'},inplace=True)

    coord_type = tmp_legacy.groupby(['id','trade_time'])['type'].apply(lambda x: len(x.unique())).reset_index()
    coord_type.rename(columns={'type':'type_diff_count_1min'},inplace=True)

    # combine to trade data
    trade =  pd.merge(trade,coord_city,on=['id','trade_time'],how='left')
    trade['city_diff_count_1min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_login,on=['id','trade_time'],how='left')
    trade['login_diff_count_1min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_device,on=['id','trade_time'],how='left')
    trade['device_diff_count_1min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_timelong,on=['id','trade_time'],how='left')
    trade['timelong_diff_count_1min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_timelong_std,on=['id','trade_time'],how='left')
    trade['timelong_std_1min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_timelong_mean,on=['id','trade_time'],how='left')
    trade['timelong_mean_1min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_logfrom,on=['id','trade_time'],how='left')
    trade['logfrom_diff_count_1min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_ip,on=['id','trade_time'],how='left')
    trade['ip_diff_count_1min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_result,on=['id','trade_time'],how='left')
    trade['result_diff_count_1min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_type,on=['id','trade_time'],how='left')
    trade['type_diff_count_1min'].fillna(0,inplace=True)
    return trade

def get_count_feats_2min(trade,log_trade_legacy):
    tmp_legacy = log_trade_legacy[log_trade_legacy['trade_log_diff_2min'] == 1].copy()
    coord_city = tmp_legacy.groupby(['id','trade_time'])['city'].apply(lambda x: len(x.unique())).reset_index()
    coord_city.rename(columns={'city':'city_diff_count_2min'},inplace=True)

    coord_login = tmp_legacy.groupby(['id','trade_time'])['login_time'].apply(lambda x: len(x.unique())).reset_index()
    coord_login.rename(columns={'login_time':'login_diff_count_2min'},inplace=True)

    coord_device = tmp_legacy.groupby(['id','trade_time'])['device'].apply(lambda x: len(x.unique())).reset_index()
    coord_device.rename(columns={'device':'device_diff_count_2min'},inplace=True)

    coord_timelong = tmp_legacy.groupby(['id','trade_time'])['timelong'].apply(lambda x: len(x.unique())).reset_index()
    coord_timelong.rename(columns={'timelong':'timelong_diff_count_2min'},inplace=True)
    coord_timelong_std = tmp_legacy.groupby(['id','trade_time'],as_index=False)['timelong'].agg({'timelong_std_2min':'std'})
    coord_timelong_std.fillna(0,inplace=True) #有的序列只有一个登陆时长
    coord_timelong_mean = tmp_legacy.groupby(['id','trade_time'],as_index=False)['timelong'].agg({'timelong_mean_2min':'mean'})

    coord_logfrom = tmp_legacy.groupby(['id','trade_time'])['log_from'].apply(lambda x: len(x.unique())).reset_index()
    coord_logfrom.rename(columns={'log_from':'logfrom_diff_count_2min'},inplace=True)

    coord_ip = tmp_legacy.groupby(['id','trade_time'])['ip'].apply(lambda x: len(x.unique())).reset_index()
    coord_ip.rename(columns={'ip':'ip_diff_count_2min'},inplace=True)

    coord_result = tmp_legacy.groupby(['id','trade_time'])['result'].apply(lambda x: len(x.unique())).reset_index()
    coord_result.rename(columns={'result':'result_diff_count_2min'},inplace=True)

    coord_type = tmp_legacy.groupby(['id','trade_time'])['type'].apply(lambda x: len(x.unique())).reset_index()
    coord_type.rename(columns={'type':'type_diff_count_2min'},inplace=True)

    # combine to trade data
    trade =  pd.merge(trade,coord_city,on=['id','trade_time'],how='left')
    trade['city_diff_count_2min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_login,on=['id','trade_time'],how='left')
    trade['login_diff_count_2min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_device,on=['id','trade_time'],how='left')
    trade['device_diff_count_2min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_timelong,on=['id','trade_time'],how='left')
    trade['timelong_diff_count_2min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_timelong_std,on=['id','trade_time'],how='left')
    trade['timelong_std_2min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_timelong_mean,on=['id','trade_time'],how='left')
    trade['timelong_mean_2min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_logfrom,on=['id','trade_time'],how='left')
    trade['logfrom_diff_count_2min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_ip,on=['id','trade_time'],how='left')
    trade['ip_diff_count_2min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_result,on=['id','trade_time'],how='left')
    trade['result_diff_count_2min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_type,on=['id','trade_time'],how='left')
    trade['type_diff_count_2min'].fillna(0,inplace=True)
    return trade

def get_count_feats_5min(trade,log_trade_legacy):
    tmp_legacy = log_trade_legacy[log_trade_legacy['trade_log_diff_5min'] == 1].copy()
    coord_city = tmp_legacy.groupby(['id','trade_time'])['city'].apply(lambda x: len(x.unique())).reset_index()
    coord_city.rename(columns={'city':'city_diff_count_5min'},inplace=True)

    coord_login = tmp_legacy.groupby(['id','trade_time'])['login_time'].apply(lambda x: len(x.unique())).reset_index()
    coord_login.rename(columns={'login_time':'login_diff_count_5min'},inplace=True)

    coord_device = tmp_legacy.groupby(['id','trade_time'])['device'].apply(lambda x: len(x.unique())).reset_index()
    coord_device.rename(columns={'device':'device_diff_count_5min'},inplace=True)

    coord_timelong = tmp_legacy.groupby(['id','trade_time'])['timelong'].apply(lambda x: len(x.unique())).reset_index()
    coord_timelong.rename(columns={'timelong':'timelong_diff_count_5min'},inplace=True)
    coord_timelong_std = tmp_legacy.groupby(['id','trade_time'],as_index=False)['timelong'].agg({'timelong_std_5min':'std'})
    coord_timelong_std.fillna(0,inplace=True) #有的序列只有一个登陆时长
    coord_timelong_mean = tmp_legacy.groupby(['id','trade_time'],as_index=False)['timelong'].agg({'timelong_mean_5min':'mean'})

    coord_logfrom = tmp_legacy.groupby(['id','trade_time'])['log_from'].apply(lambda x: len(x.unique())).reset_index()
    coord_logfrom.rename(columns={'log_from':'logfrom_diff_count_5min'},inplace=True)

    coord_ip = tmp_legacy.groupby(['id','trade_time'])['ip'].apply(lambda x: len(x.unique())).reset_index()
    coord_ip.rename(columns={'ip':'ip_diff_count_5min'},inplace=True)

    coord_result = tmp_legacy.groupby(['id','trade_time'])['result'].apply(lambda x: len(x.unique())).reset_index()
    coord_result.rename(columns={'result':'result_diff_count_5min'},inplace=True)

    coord_type = tmp_legacy.groupby(['id','trade_time'])['type'].apply(lambda x: len(x.unique())).reset_index()
    coord_type.rename(columns={'type':'type_diff_count_5min'},inplace=True)

    # combine to trade data
    trade =  pd.merge(trade,coord_city,on=['id','trade_time'],how='left')
    trade['city_diff_count_5min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_login,on=['id','trade_time'],how='left')
    trade['login_diff_count_5min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_device,on=['id','trade_time'],how='left')
    trade['device_diff_count_5min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_timelong,on=['id','trade_time'],how='left')
    trade['timelong_diff_count_5min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_timelong_std,on=['id','trade_time'],how='left')
    trade['timelong_std_5min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_timelong_mean,on=['id','trade_time'],how='left')
    trade['timelong_mean_5min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_logfrom,on=['id','trade_time'],how='left')
    trade['logfrom_diff_count_5min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_ip,on=['id','trade_time'],how='left')
    trade['ip_diff_count_5min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_result,on=['id','trade_time'],how='left')
    trade['result_diff_count_5min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_type,on=['id','trade_time'],how='left')
    trade['type_diff_count_5min'].fillna(0,inplace=True)
    return trade

def get_count_feats_10min(trade,log_trade_legacy):
    tmp_legacy = log_trade_legacy[log_trade_legacy['trade_log_diff_10min'] == 1].copy()
    coord_city = tmp_legacy.groupby(['id','trade_time'])['city'].apply(lambda x: len(x.unique())).reset_index()
    coord_city.rename(columns={'city':'city_diff_count_10min'},inplace=True)

    coord_login = tmp_legacy.groupby(['id','trade_time'])['login_time'].apply(lambda x: len(x.unique())).reset_index()
    coord_login.rename(columns={'login_time':'login_diff_count_10min'},inplace=True)

    coord_device = tmp_legacy.groupby(['id','trade_time'])['device'].apply(lambda x: len(x.unique())).reset_index()
    coord_device.rename(columns={'device':'device_diff_count_10min'},inplace=True)

    coord_timelong = tmp_legacy.groupby(['id','trade_time'])['timelong'].apply(lambda x: len(x.unique())).reset_index()
    coord_timelong.rename(columns={'timelong':'timelong_diff_count_10min'},inplace=True)
    coord_timelong_std = tmp_legacy.groupby(['id','trade_time'],as_index=False)['timelong'].agg({'timelong_std_10min':'std'})
    coord_timelong_std.fillna(0,inplace=True) #有的序列只有一个登陆时长
    coord_timelong_mean = tmp_legacy.groupby(['id','trade_time'],as_index=False)['timelong'].agg({'timelong_mean_10min':'mean'})

    coord_logfrom = tmp_legacy.groupby(['id','trade_time'])['log_from'].apply(lambda x: len(x.unique())).reset_index()
    coord_logfrom.rename(columns={'log_from':'logfrom_diff_count_10min'},inplace=True)

    coord_ip = tmp_legacy.groupby(['id','trade_time'])['ip'].apply(lambda x: len(x.unique())).reset_index()
    coord_ip.rename(columns={'ip':'ip_diff_count_10min'},inplace=True)

    coord_result = tmp_legacy.groupby(['id','trade_time'])['result'].apply(lambda x: len(x.unique())).reset_index()
    coord_result.rename(columns={'result':'result_diff_count_10min'},inplace=True)

    coord_type = tmp_legacy.groupby(['id','trade_time'])['type'].apply(lambda x: len(x.unique())).reset_index()
    coord_type.rename(columns={'type':'type_diff_count_10min'},inplace=True)

    # combine to trade data
    trade =  pd.merge(trade,coord_city,on=['id','trade_time'],how='left')
    trade['city_diff_count_10min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_login,on=['id','trade_time'],how='left')
    trade['login_diff_count_10min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_device,on=['id','trade_time'],how='left')
    trade['device_diff_count_10min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_timelong,on=['id','trade_time'],how='left')
    trade['timelong_diff_count_10min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_timelong_std,on=['id','trade_time'],how='left')
    trade['timelong_std_10min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_timelong_mean,on=['id','trade_time'],how='left')
    trade['timelong_mean_10min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_logfrom,on=['id','trade_time'],how='left')
    trade['logfrom_diff_count_10min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_ip,on=['id','trade_time'],how='left')
    trade['ip_diff_count_10min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_result,on=['id','trade_time'],how='left')
    trade['result_diff_count_10min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_type,on=['id','trade_time'],how='left')
    trade['type_diff_count_10min'].fillna(0,inplace=True)
    return trade

def get_count_feats_15min(trade,log_trade_legacy):
    tmp_legacy = log_trade_legacy[log_trade_legacy['trade_log_diff_15min'] == 1].copy()
    coord_city = tmp_legacy.groupby(['id','trade_time'])['city'].apply(lambda x: len(x.unique())).reset_index()
    coord_city.rename(columns={'city':'city_diff_count_15min'},inplace=True)

    coord_login = tmp_legacy.groupby(['id','trade_time'])['login_time'].apply(lambda x: len(x.unique())).reset_index()
    coord_login.rename(columns={'login_time':'login_diff_count_15min'},inplace=True)

    coord_device = tmp_legacy.groupby(['id','trade_time'])['device'].apply(lambda x: len(x.unique())).reset_index()
    coord_device.rename(columns={'device':'device_diff_count_15min'},inplace=True)

    coord_timelong = tmp_legacy.groupby(['id','trade_time'])['timelong'].apply(lambda x: len(x.unique())).reset_index()
    coord_timelong.rename(columns={'timelong':'timelong_diff_count_15min'},inplace=True)
    coord_timelong_std = tmp_legacy.groupby(['id','trade_time'],as_index=False)['timelong'].agg({'timelong_std_15min':'std'})
    coord_timelong_std.fillna(0,inplace=True) #有的序列只有一个登陆时长
    coord_timelong_mean = tmp_legacy.groupby(['id','trade_time'],as_index=False)['timelong'].agg({'timelong_mean_15min':'mean'})

    coord_logfrom = tmp_legacy.groupby(['id','trade_time'])['log_from'].apply(lambda x: len(x.unique())).reset_index()
    coord_logfrom.rename(columns={'log_from':'logfrom_diff_count_15min'},inplace=True)

    coord_ip = tmp_legacy.groupby(['id','trade_time'])['ip'].apply(lambda x: len(x.unique())).reset_index()
    coord_ip.rename(columns={'ip':'ip_diff_count_15min'},inplace=True)

    coord_result = tmp_legacy.groupby(['id','trade_time'])['result'].apply(lambda x: len(x.unique())).reset_index()
    coord_result.rename(columns={'result':'result_diff_count_15min'},inplace=True)

    coord_type = tmp_legacy.groupby(['id','trade_time'])['type'].apply(lambda x: len(x.unique())).reset_index()
    coord_type.rename(columns={'type':'type_diff_count_15min'},inplace=True)

    # combine to trade data
    trade =  pd.merge(trade,coord_city,on=['id','trade_time'],how='left')
    trade['city_diff_count_15min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_login,on=['id','trade_time'],how='left')
    trade['login_diff_count_15min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_device,on=['id','trade_time'],how='left')
    trade['device_diff_count_15min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_timelong,on=['id','trade_time'],how='left')
    trade['timelong_diff_count_15min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_timelong_std,on=['id','trade_time'],how='left')
    trade['timelong_std_15min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_timelong_mean,on=['id','trade_time'],how='left')
    trade['timelong_mean_15min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_logfrom,on=['id','trade_time'],how='left')
    trade['logfrom_diff_count_15min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_ip,on=['id','trade_time'],how='left')
    trade['ip_diff_count_15min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_result,on=['id','trade_time'],how='left')
    trade['result_diff_count_15min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_type,on=['id','trade_time'],how='left')
    trade['type_diff_count_15min'].fillna(0,inplace=True)
    return trade

def get_count_feats_30min(trade,log_trade_legacy):
    tmp_legacy = log_trade_legacy[log_trade_legacy['trade_log_diff_30min'] == 1].copy()
    coord_city = tmp_legacy.groupby(['id','trade_time'])['city'].apply(lambda x: len(x.unique())).reset_index()
    coord_city.rename(columns={'city':'city_diff_count_30min'},inplace=True)

    coord_login = tmp_legacy.groupby(['id','trade_time'])['login_time'].apply(lambda x: len(x.unique())).reset_index()
    coord_login.rename(columns={'login_time':'login_diff_count_30min'},inplace=True)

    coord_device = tmp_legacy.groupby(['id','trade_time'])['device'].apply(lambda x: len(x.unique())).reset_index()
    coord_device.rename(columns={'device':'device_diff_count_30min'},inplace=True)

    coord_timelong = tmp_legacy.groupby(['id','trade_time'])['timelong'].apply(lambda x: len(x.unique())).reset_index()
    coord_timelong.rename(columns={'timelong':'timelong_diff_count_30min'},inplace=True)
    coord_timelong_std = tmp_legacy.groupby(['id','trade_time'],as_index=False)['timelong'].agg({'timelong_std_30min':'std'})
    coord_timelong_std.fillna(0,inplace=True) #有的序列只有一个登陆时长
    coord_timelong_mean = tmp_legacy.groupby(['id','trade_time'],as_index=False)['timelong'].agg({'timelong_mean_30min':'mean'})

    coord_logfrom = tmp_legacy.groupby(['id','trade_time'])['log_from'].apply(lambda x: len(x.unique())).reset_index()
    coord_logfrom.rename(columns={'log_from':'logfrom_diff_count_30min'},inplace=True)

    coord_ip = tmp_legacy.groupby(['id','trade_time'])['ip'].apply(lambda x: len(x.unique())).reset_index()
    coord_ip.rename(columns={'ip':'ip_diff_count_30min'},inplace=True)

    coord_result = tmp_legacy.groupby(['id','trade_time'])['result'].apply(lambda x: len(x.unique())).reset_index()
    coord_result.rename(columns={'result':'result_diff_count_30min'},inplace=True)

    coord_type = tmp_legacy.groupby(['id','trade_time'])['type'].apply(lambda x: len(x.unique())).reset_index()
    coord_type.rename(columns={'type':'type_diff_count_30min'},inplace=True)

    # combine to trade data
    trade =  pd.merge(trade,coord_city,on=['id','trade_time'],how='left')
    trade['city_diff_count_30min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_login,on=['id','trade_time'],how='left')
    trade['login_diff_count_30min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_device,on=['id','trade_time'],how='left')
    trade['device_diff_count_30min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_timelong,on=['id','trade_time'],how='left')
    trade['timelong_diff_count_30min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_timelong_std,on=['id','trade_time'],how='left')
    trade['timelong_std_30min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_timelong_mean,on=['id','trade_time'],how='left')
    trade['timelong_mean_30min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_logfrom,on=['id','trade_time'],how='left')
    trade['logfrom_diff_count_30min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_ip,on=['id','trade_time'],how='left')
    trade['ip_diff_count_30min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_result,on=['id','trade_time'],how='left')
    trade['result_diff_count_30min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_type,on=['id','trade_time'],how='left')
    trade['type_diff_count_30min'].fillna(0,inplace=True)
    return trade

def get_count_feats_60min(trade,log_trade_legacy):
    tmp_legacy = log_trade_legacy[log_trade_legacy['trade_log_diff_60min'] == 1].copy()
    coord_city = tmp_legacy.groupby(['id','trade_time'])['city'].apply(lambda x: len(x.unique())).reset_index()
    coord_city.rename(columns={'city':'city_diff_count_60min'},inplace=True)

    coord_login = tmp_legacy.groupby(['id','trade_time'])['login_time'].apply(lambda x: len(x.unique())).reset_index()
    coord_login.rename(columns={'login_time':'login_diff_count_60min'},inplace=True)

    coord_device = tmp_legacy.groupby(['id','trade_time'])['device'].apply(lambda x: len(x.unique())).reset_index()
    coord_device.rename(columns={'device':'device_diff_count_60min'},inplace=True)

    coord_timelong = tmp_legacy.groupby(['id','trade_time'])['timelong'].apply(lambda x: len(x.unique())).reset_index()
    coord_timelong.rename(columns={'timelong':'timelong_diff_count_60min'},inplace=True)
    coord_timelong_std = tmp_legacy.groupby(['id','trade_time'],as_index=False)['timelong'].agg({'timelong_std_60min':'std'})
    coord_timelong_std.fillna(0,inplace=True) #有的序列只有一个登陆时长
    coord_timelong_mean = tmp_legacy.groupby(['id','trade_time'],as_index=False)['timelong'].agg({'timelong_mean_60min':'mean'})

    coord_logfrom = tmp_legacy.groupby(['id','trade_time'])['log_from'].apply(lambda x: len(x.unique())).reset_index()
    coord_logfrom.rename(columns={'log_from':'logfrom_diff_count_60min'},inplace=True)

    coord_ip = tmp_legacy.groupby(['id','trade_time'])['ip'].apply(lambda x: len(x.unique())).reset_index()
    coord_ip.rename(columns={'ip':'ip_diff_count_60min'},inplace=True)

    coord_result = tmp_legacy.groupby(['id','trade_time'])['result'].apply(lambda x: len(x.unique())).reset_index()
    coord_result.rename(columns={'result':'result_diff_count_60min'},inplace=True)

    coord_type = tmp_legacy.groupby(['id','trade_time'])['type'].apply(lambda x: len(x.unique())).reset_index()
    coord_type.rename(columns={'type':'type_diff_count_60min'},inplace=True)

    # combine to trade data
    trade =  pd.merge(trade,coord_city,on=['id','trade_time'],how='left')
    trade['city_diff_count_60min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_login,on=['id','trade_time'],how='left')
    trade['login_diff_count_60min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_device,on=['id','trade_time'],how='left')
    trade['device_diff_count_60min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_timelong,on=['id','trade_time'],how='left')
    trade['timelong_diff_count_60min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_timelong_std,on=['id','trade_time'],how='left')
    trade['timelong_std_60min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_timelong_mean,on=['id','trade_time'],how='left')
    trade['timelong_mean_60min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_logfrom,on=['id','trade_time'],how='left')
    trade['logfrom_diff_count_60min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_ip,on=['id','trade_time'],how='left')
    trade['ip_diff_count_60min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_result,on=['id','trade_time'],how='left')
    trade['result_diff_count_60min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_type,on=['id','trade_time'],how='left')
    trade['type_diff_count_60min'].fillna(0,inplace=True)
    return trade

def get_count_feats_120min(trade,log_trade_legacy):
    tmp_legacy = log_trade_legacy[log_trade_legacy['trade_log_diff_120min'] == 1].copy()
    coord_city = tmp_legacy.groupby(['id','trade_time'])['city'].apply(lambda x: len(x.unique())).reset_index()
    coord_city.rename(columns={'city':'city_diff_count_120min'},inplace=True)

    coord_login = tmp_legacy.groupby(['id','trade_time'])['login_time'].apply(lambda x: len(x.unique())).reset_index()
    coord_login.rename(columns={'login_time':'login_diff_count_120min'},inplace=True)

    coord_device = tmp_legacy.groupby(['id','trade_time'])['device'].apply(lambda x: len(x.unique())).reset_index()
    coord_device.rename(columns={'device':'device_diff_count_120min'},inplace=True)

    coord_timelong = tmp_legacy.groupby(['id','trade_time'])['timelong'].apply(lambda x: len(x.unique())).reset_index()
    coord_timelong.rename(columns={'timelong':'timelong_diff_count_120min'},inplace=True)
    coord_timelong_std = tmp_legacy.groupby(['id','trade_time'],as_index=False)['timelong'].agg({'timelong_std_120min':'std'})
    coord_timelong_std.fillna(0,inplace=True) #有的序列只有一个登陆时长
    coord_timelong_mean = tmp_legacy.groupby(['id','trade_time'],as_index=False)['timelong'].agg({'timelong_mean_120min':'mean'})

    coord_logfrom = tmp_legacy.groupby(['id','trade_time'])['log_from'].apply(lambda x: len(x.unique())).reset_index()
    coord_logfrom.rename(columns={'log_from':'logfrom_diff_count_120min'},inplace=True)

    coord_ip = tmp_legacy.groupby(['id','trade_time'])['ip'].apply(lambda x: len(x.unique())).reset_index()
    coord_ip.rename(columns={'ip':'ip_diff_count_120min'},inplace=True)

    coord_result = tmp_legacy.groupby(['id','trade_time'])['result'].apply(lambda x: len(x.unique())).reset_index()
    coord_result.rename(columns={'result':'result_diff_count_120min'},inplace=True)

    coord_type = tmp_legacy.groupby(['id','trade_time'])['type'].apply(lambda x: len(x.unique())).reset_index()
    coord_type.rename(columns={'type':'type_diff_count_120min'},inplace=True)

    # combine to trade data
    trade =  pd.merge(trade,coord_city,on=['id','trade_time'],how='left')
    trade['city_diff_count_120min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_login,on=['id','trade_time'],how='left')
    trade['login_diff_count_120min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_device,on=['id','trade_time'],how='left')
    trade['device_diff_count_120min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_timelong,on=['id','trade_time'],how='left')
    trade['timelong_diff_count_120min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_timelong_std,on=['id','trade_time'],how='left')
    trade['timelong_std_120min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_timelong_mean,on=['id','trade_time'],how='left')
    trade['timelong_mean_120min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_logfrom,on=['id','trade_time'],how='left')
    trade['logfrom_diff_count_120min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_ip,on=['id','trade_time'],how='left')
    trade['ip_diff_count_120min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_result,on=['id','trade_time'],how='left')
    trade['result_diff_count_120min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_type,on=['id','trade_time'],how='left')
    trade['type_diff_count_120min'].fillna(0,inplace=True)
    return trade

def get_count_feats_300min(trade,log_trade_legacy):
    tmp_legacy = log_trade_legacy[log_trade_legacy['trade_log_diff_300min'] == 1].copy()
    coord_city = tmp_legacy.groupby(['id','trade_time'])['city'].apply(lambda x: len(x.unique())).reset_index()
    coord_city.rename(columns={'city':'city_diff_count_300min'},inplace=True)

    coord_login = tmp_legacy.groupby(['id','trade_time'])['login_time'].apply(lambda x: len(x.unique())).reset_index()
    coord_login.rename(columns={'login_time':'login_diff_count_300min'},inplace=True)

    coord_device = tmp_legacy.groupby(['id','trade_time'])['device'].apply(lambda x: len(x.unique())).reset_index()
    coord_device.rename(columns={'device':'device_diff_count_300min'},inplace=True)

    coord_timelong = tmp_legacy.groupby(['id','trade_time'])['timelong'].apply(lambda x: len(x.unique())).reset_index()
    coord_timelong.rename(columns={'timelong':'timelong_diff_count_300min'},inplace=True)
    coord_timelong_std = tmp_legacy.groupby(['id','trade_time'],as_index=False)['timelong'].agg({'timelong_std_300min':'std'})
    coord_timelong_std.fillna(0,inplace=True) #有的序列只有一个登陆时长
    coord_timelong_mean = tmp_legacy.groupby(['id','trade_time'],as_index=False)['timelong'].agg({'timelong_mean_300min':'mean'})

    coord_logfrom = tmp_legacy.groupby(['id','trade_time'])['log_from'].apply(lambda x: len(x.unique())).reset_index()
    coord_logfrom.rename(columns={'log_from':'logfrom_diff_count_300min'},inplace=True)

    coord_ip = tmp_legacy.groupby(['id','trade_time'])['ip'].apply(lambda x: len(x.unique())).reset_index()
    coord_ip.rename(columns={'ip':'ip_diff_count_300min'},inplace=True)

    coord_result = tmp_legacy.groupby(['id','trade_time'])['result'].apply(lambda x: len(x.unique())).reset_index()
    coord_result.rename(columns={'result':'result_diff_count_300min'},inplace=True)

    coord_type = tmp_legacy.groupby(['id','trade_time'])['type'].apply(lambda x: len(x.unique())).reset_index()
    coord_type.rename(columns={'type':'type_diff_count_300min'},inplace=True)

    # combine to trade data
    trade =  pd.merge(trade,coord_city,on=['id','trade_time'],how='left')
    trade['city_diff_count_300min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_login,on=['id','trade_time'],how='left')
    trade['login_diff_count_300min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_device,on=['id','trade_time'],how='left')
    trade['device_diff_count_300min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_timelong,on=['id','trade_time'],how='left')
    trade['timelong_diff_count_300min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_timelong_std,on=['id','trade_time'],how='left')
    trade['timelong_std_300min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_timelong_mean,on=['id','trade_time'],how='left')
    trade['timelong_mean_300min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_logfrom,on=['id','trade_time'],how='left')
    trade['logfrom_diff_count_300min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_ip,on=['id','trade_time'],how='left')
    trade['ip_diff_count_300min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_result,on=['id','trade_time'],how='left')
    trade['result_diff_count_300min'].fillna(0,inplace=True)
    trade =  pd.merge(trade,coord_type,on=['id','trade_time'],how='left')
    trade['type_diff_count_300min'].fillna(0,inplace=True)
    return trade

def get_train_feats(login_tr,trade_tr):
    log_trade = log_train_combine(login_tr, trade_tr)
    print 'log trade data combine done.'
    log_trade_legacy = get_legacy_user_info(log_trade)
    print 'get log trade legacy user information done.'
    log_trade_legacy = get_time_split(log_trade_legacy)
    print 'time split done.'
    trade_tr = get_count_feats_1min(trade_tr,log_trade_legacy)
    trade_tr = get_count_feats_2min(trade_tr,log_trade_legacy)
    trade_tr = get_count_feats_5min(trade_tr,log_trade_legacy)
    trade_tr = get_count_feats_10min(trade_tr,log_trade_legacy)
    trade_tr = get_count_feats_15min(trade_tr,log_trade_legacy)
    trade_tr = get_count_feats_30min(trade_tr,log_trade_legacy)
    trade_tr = get_count_feats_60min(trade_tr,log_trade_legacy)
    trade_tr = get_count_feats_120min(trade_tr,log_trade_legacy)
    trade_tr = get_count_feats_300min(trade_tr,log_trade_legacy)
    print 'all split time frame features done.'
    return trade_tr


def get_test_feats(login_te,trade_te):
    log_trade = log_train_combine(login_te, trade_te)
    print 'log trade data combine done.'
    log_trade_legacy = get_legacy_user_info(log_trade)
    print 'get log trade legacy user information done.'
    log_trade_legacy = get_time_split(log_trade_legacy)
    print 'time split done.'
    trade_te = get_count_feats_1min(trade_te,log_trade_legacy)
    trade_te = get_count_feats_2min(trade_te,log_trade_legacy)
    trade_te = get_count_feats_5min(trade_te,log_trade_legacy)
    trade_te = get_count_feats_10min(trade_te,log_trade_legacy)
    trade_te = get_count_feats_15min(trade_te,log_trade_legacy)
    trade_te = get_count_feats_30min(trade_te,log_trade_legacy)
    trade_te = get_count_feats_60min(trade_te,log_trade_legacy)
    trade_te = get_count_feats_120min(trade_te,log_trade_legacy)
    trade_te = get_count_feats_300min(trade_te,log_trade_legacy)
    print 'all split time frame features done.'
    return trade_te


