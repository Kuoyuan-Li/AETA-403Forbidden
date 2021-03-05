import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold,KFold
import os 
import re

import warnings
warnings.filterwarnings('ignore')

def get_area_data(area_id, flag, path='./area_feature/'):
    file_name = path+'area_' + str(area_id) + '_' + flag + '.csv'
    return pd.read_csv(file_name,index_col=0)  # read the csv (area_feature/area_<id>_train/valid.csv)

if __name__ == "__main__":
    for area in (0,1,2,3,4,5,6,7):  # one area for a time
        print(f"Area_{area}:")
        train_data = get_area_data(area, 'train')  # get the train data of an area
        valid_data = get_area_data(area, 'valid')  # get the valid data of an area

    #训练数据
        print("Trainning......")
        #数据平衡，升采样
        long_data = train_data[train_data['label_M']==0]  # data with no earthquake (usually long)
        short_data = train_data[train_data['label_M']!=0]  # data with earthquake (usually short)
        if len(long_data)<len(short_data) :
            long_data,short_data = short_data,long_data  # if long data too short, exchange
        short_data = short_data.sample(len(long_data), replace=True)  # return random sample with length equal to the long_data, allow sampling same row more than once
        train_data = pd.concat([long_data,short_data])  # concatenate short data (eq) and long data (no eq)
        train_data = train_data.sample(frac=1).reset_index(drop=True)  # reset index
        del long_data
        del short_data

        target_M = train_data['label_M']  # magnitude
        train_feature = train_data.drop(['label_M','label_long','label_lati','Day'],axis=1)  # train features: no eq related columns

        #震级类别化
        for i, ss in enumerate(target_M):
            if(ss<3.5):
                target_M.iloc[i] = 0
            elif(ss<4.0):
                target_M.iloc[i] = 1
            elif(ss<4.5):
                target_M.iloc[i] = 2
            elif(ss<5.0):
                target_M.iloc[i] = 3
            else:
                target_M.iloc[i] = 4
        #震级赋权 equal weight for now
        train_data['weight'] = None
        train_data['label_M'] = target_M  #magnitude in 震级类别化 (instead of discrete number)
        train_data['weight'][train_data['label_M']==0] = 1
        train_data['weight'][train_data['label_M']==1] = 1
        train_data['weight'][train_data['label_M']==2] = 1
        train_data['weight'][train_data['label_M']==3] = 1
        train_data['weight'][train_data['label_M']==4] = 1
        weight_T = train_data['weight'].values
    #验证数据
        valid_M = valid_data['label_M']
        valid_feature = valid_data.drop(['label_M','label_long','label_lati','Day'],axis=1)
        for i, ss in enumerate(valid_M): # 震级类别化 classify earthquake level
            if(ss<3.5):
                valid_M.iloc[i] = 0
            elif(ss<4.0):
                valid_M.iloc[i] = 1
            elif(ss<4.5):
                valid_M.iloc[i] = 2
            elif(ss<5.0):
                valid_M.iloc[i] = 3
            else:
                valid_M.iloc[i] = 4
        # weight valid_Data (equal now)
        valid_data['weight'] = None
        valid_data['label_M'] = valid_M
        valid_data['weight'][valid_data['label_M']==0] = 1
        valid_data['weight'][valid_data['label_M']==1] = 1
        valid_data['weight'][valid_data['label_M']==2] = 1
        valid_data['weight'][valid_data['label_M']==3] = 1
        valid_data['weight'][valid_data['label_M']==4] = 1
        weight_V = valid_data['weight'].values

        params = {
        'num_leaves': 48,
        'learning_rate': 0.05,
        "boosting": "rf",
        'objective': 'multiclass', #转为分类问题
        'num_class': 5,
        # 'objective': 'regression',
        "feature_fraction": 0.6,
        "bagging_fraction": 0.6,
        "bagging_freq": 2,
        "lambda_l1": 0.05,
        "lambda_l2": 0.05,
        "nthread": -1,
        'min_child_samples': 10,
        'max_bin': 200,
        'verbose' : -1
        }
        num_round = 5000
        trn_data = lgb.Dataset(train_feature, label=target_M,weight=weight_T)
        val_data = lgb.Dataset(valid_feature, label=valid_M, weight=weight_V)
        clf = lgb.train(params, trn_data, num_round, valid_sets=[trn_data,val_data],verbose_eval=50,
                        early_stopping_rounds=500)  # train

        #评估
        oof_lgb = np.matrix(clf.predict(valid_feature, num_iteration=clf.best_iteration))
        cpf = open(str(area)+'_sm.txt','w+')
        ccc = oof_lgb.argmax(axis=1)
        for i in range(len(ccc)):
             print(f"{i}, pre:{ccc[i]}, origin:{valid_M[i]}", file=cpf)
        print((oof_lgb.argmax(axis=1)==(np.matrix(valid_M).T)).sum(),len(oof_lgb))

        #保存区域模型
        clf.save_model('./model/'+str(area)+'_mag_model.txt')
