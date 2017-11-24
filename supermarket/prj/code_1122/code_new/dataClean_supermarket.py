# -*- coding:utf-8 -*-
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

import time
# Handle table like and matrices
import pandas as pd
import numpy as np

def exclude_class(train_new_o, train_o, do_not_use_class):
    # train_o = train_o[~train_o['Class'].isin(do_not_use_class)]
    # train_new_o = train_new_o[~train_new_o['Class'].isin(do_not_use_class)]
    train_o = train_o[train_o['Class'].isin(do_not_use_class)]
    train_new_o = train_new_o[train_new_o['Class'].isin(do_not_use_class)]
    return train_o, train_new_o

# def exclude_abnormal_value(train):
#     # best 2 mean 1.5 std
#     step = 14   # past one week
#     train.loc[:,'count'] = range(0,train.shape[0])
#     train['count'] = train['count'] % (train[train['Class'] == 10].shape[0])
#     train['count'] = (train['count'] / step)
#     # train['count'] = (train['count'] / step).astype('int')
#     coord = train.groupby('Class',as_index=False)['saleCount'].agg({'mean':'mean'})
#     train = pd.merge(train, coord, on=['Class'],how='left')
#     coord = train.groupby('Class',as_index=False)['saleCount'].agg({'std':'std'})
#     train = pd.merge(train, coord, on='Class',how='left')
#     train['mean'] = train.groupby('Class')['mean'].shift(step)
#     train['std'] = train.groupby('Class')['std'].shift(step)
#     train['mean'].fillna(method='bfill',inplace=True)
#     train['std'].fillna(method='bfill',inplace=True)
#     train.loc[:,'saleCount_min'] = train['mean'] - 2 * train['std']
#     train.loc[:,'saleCount_max'] = train['mean'] + 1.3 * train['std']
#     train['saleCount_max'] = np.ceil(train['saleCount_max'])
#     max_bool = train['saleCount_max'] < train['saleCount']
#     train['saleCount'][max_bool] = train['saleCount_max'][max_bool]
#     neg_bool = train['saleCount_min'] < 0
#     train['saleCount_min'][neg_bool] = 0
#     min_bool = train['saleCount_min'] > train['saleCount']
#     train['saleCount'][min_bool] = train['saleCount_min'][min_bool]
#     train['saleCount'] = np.ceil(train['saleCount'])
#     del train['saleCount_min'],train['saleCount_max'],train['std'],train['mean'],train['count']
#
#     # coord  = train.groupby('Class',as_index=False)['saleCount'].agg({'saleCount_mean':'mean'})
#     # train = pd.merge(train, coord, on='Class', how = 'left')
#     # coord  = train.groupby('Class',as_index=False)['saleCount'].agg({'saleCount_std':'std'})
#     # train = pd.merge(train, coord, on='Class', how = 'left')
#     # train.loc[:,'saleCount_min'] = train['saleCount_mean'] - 2 * train['saleCount_std']
#     # train.loc[:,'saleCount_max'] = train['saleCount_mean'] + 2 * train['saleCount_std']
#     # train['saleCount_max'] = np.ceil(train['saleCount_max'])
#     # max_bool = train['saleCount_max'] < train['saleCount']
#     # train['saleCount'][max_bool] = train['saleCount_max'][max_bool]
#     # neg_bool = train['saleCount_min'] < 0
#     # train['saleCount_min'][neg_bool] = 0
#     # min_bool = train['saleCount_min'] > train['saleCount']
#     # train['saleCount'][min_bool] = train['saleCount_min'][min_bool]
#     # train['saleCount'] = np.ceil(train['saleCount'])
#     # del train['saleCount_min'],train['saleCount_max'],train['saleCount_std'],train['saleCount_mean']
#     return train

def exclude_abnormal_value(train):
    step = 14   # past two week
    train.loc[:,'count'] = range(0,train.shape[0])
    train['count'] = train['count'] % (train[train['Class'] == 20].shape[0])
    train['count'] = (train['count'] / step).astype('int')
    coord = train.groupby(['Class','count'],as_index=False)['saleCount'].agg({'mean':'mean'})
    train = pd.merge(train, coord, on=['Class','count'],how='left')
    coord = train.groupby(['Class','count'],as_index=False)['saleCount'].agg({'std':'std'})
    train = pd.merge(train, coord, on=['Class','count'],how='left')
    train['mean'] = train.groupby('Class')['mean'].shift(step)
    train['std'] = train.groupby('Class')['std'].shift(step)
    train['mean'].fillna(method='bfill',inplace=True)
    train['std'].fillna(method='bfill',inplace=True)
    train.loc[:,'saleCount_min'] = train['mean'] - 2 * train['std']
    train.loc[:,'saleCount_max'] = train['mean'] + 2 * train['std']
    train['saleCount_max'] = np.ceil(train['saleCount_max'])
    max_bool = train['saleCount_max'] < train['saleCount']
    train['saleCount'][max_bool] = train['saleCount_max'][max_bool]
    neg_bool = train['saleCount_min'] < 0
    train['saleCount_min'][neg_bool] = 0
    min_bool = train['saleCount_min'] > train['saleCount']
    train['saleCount'][min_bool] = train['saleCount_min'][min_bool]
    train['saleCount'] = np.ceil(train['saleCount'])
    del train['saleCount_min'],train['saleCount_max'],train['std'],train['mean'],train['count']
    return train

def exclude_abnormal_value_coupon(train):
    step = 14   # past two week
    train.loc[:,'count'] = range(0,train.shape[0])
    train['count'] = train['count'] % (train[train['Class'] == 20].shape[0])
    # print train['count']
    train['count'] = (train['count'] / step).astype('int')
    coord = train.groupby(['Class','count'],as_index=False)['saleCount'].agg({'mean':'mean'})
    train = pd.merge(train, coord, on=['Class','count'],how='left')
    coord = train.groupby(['Class','count'],as_index=False)['saleCount'].agg({'std':'std'})
    train = pd.merge(train, coord, on=['Class','count'],how='left')
    train['mean'] = train.groupby('Class')['mean'].shift(step)
    train['std'] = train.groupby('Class')['std'].shift(step)
    train['mean'].fillna(method='bfill',inplace=True)
    train['std'].fillna(method='bfill',inplace=True)
    train.loc[:,'saleCount_min'] = train['mean'] - 1.5 * train['std']
    train.loc[:,'saleCount_max'] = train['mean'] + 1.5 * train['std']
    train['saleCount_max'] = np.ceil(train['saleCount_max'])
    max_bool = train['saleCount_max'] < train['saleCount']
    train['saleCount'][max_bool] = train['saleCount_max'][max_bool]
    neg_bool = train['saleCount_min'] < 0
    train['saleCount_min'][neg_bool] = 0
    min_bool = train['saleCount_min'] > train['saleCount']
    train['saleCount'][min_bool] = train['saleCount_min'][min_bool]
    train['saleCount'] = np.ceil(train['saleCount'])
    del train['saleCount_min'],train['std'],train['mean'],train['count']
    return train