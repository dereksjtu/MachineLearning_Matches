# -*- coding:utf-8 -*-

# genneral
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from feature_jdd import *

# machine learning tools
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import fbeta_score
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import learning_curve

# gbdt tools
import xgboost as xgb
import lightgbm as lgb

if __name__ ==  "__main__":
    pipe_lr=getPipe()
    X_train,y_train=getTrainData(isUndersample=False)
    #记录程序运行时间
    start_time = time.time()
    scores = cross_val_score(estimator=pipe_lr, X=X_train, y=y_train, cv=10, n_jobs=2,scoring=rocJdScore)
    print(scores)
    #整体预测
    X_train,y_train=getTrainData(isUndersample=False)
    pipe_lr
    #输出运行时长
    cost_time = time.time()-start_time
    print("交叉验证 success!",'\n',"cost time:",cost_time,"(s)")

    #网格搜索实验
    parameters = {
    #     'rf__n_estimators': (5, 10, 20, 50),
    #     'rf__max_depth': (50, 150, 250),
    #     'rf__min_samples_split': [10, 2, 3],
    #     'rf__min_samples_leaf': (1, 2, 3),
        #xgb的参数
        'xgb__max_depth':(4,6),
        'xgb__learning_rate':(0.3,0.5)

    }
    pipe_lr=getPipe()
    X_train,y_train=getTrainData()


    #网格搜索
    grid_search = GridSearchCV(pipe_lr, parameters, n_jobs=-1, verbose=1, scoring=rocJdScore)
    grid_search.fit(X_train, y_train)

    #获取最优参数
    print('最佳效果：%0.3f' % grid_search.best_score_)
    print('最优参数：')
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print('\t%s= %r' % (param_name, best_parameters[param_name]))

    # #预测以及分类器参数报告
    # predictions = grid_search.predict(X_test)
    # print(classification_report(y_test, predictions))

    #学习曲线
    pipe_lr = getPipe()
    X_train,y_train=getTrainData()
    # train_sizes参数指定用于生成学习曲线的训练集数量，如果是分数指的是相对数量，整数指的是绝对数量
    train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr, X=X_train, y=y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=10,
                                                            n_jobs=2,scoring=rocJdScore)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    plt.grid()
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.0, 1.0])
    plt.show()

    pipe=getPipe()
    pipe=jdPipeFited(pipe)
    preData=transferData(loginTestData,tradeTestData)
    x_pred=preData.iloc[:,2:].values
    y_pred=pipe.predict(x_pred)
    np.sum(y_pred)

    p=pd.DataFrame(y_pred)
    subData=pd.DataFrame(preData['rowkey'])
    subData['is_risk']=p
    #之前用很多inner join，很多数据没有，都默认处理为没有风险
    subData=pd.merge(tradeTestData[['rowkey']],subData,on='rowkey',how='left')
    subData=subData.fillna(0)
    subData['is_risk']=subData['is_risk'].astype('int')
    subData.to_csv('./sub.csv',header=False,index=False)
