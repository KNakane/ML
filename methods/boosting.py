import os, sys
import sklearn
import xgboost as xgb
import lightgbm as lgb

class XGBoost():
    def __init__(self, n_class=0):
        if n_class:
            self.params = {
                'objective': 'multiclass', # 多値分類問題
                'num_class': 3
                }
        else:
            self.params = {
                'objective': 'multiclass', # 多値分類問題
                'num_class': 3
                }
    
    def train(self, x, y):
        dtrain = xgb.DMatrix(x, label=y)
        self.model = xgb.train(self.params, 
                               dtrain, 
                               early_stopping_rounds=10,
                               num_boost_round=100)

    def predict(self, x, y):
        dtest = xgb.DMatrix(x, label=y)
        return self.model.predict(dtest)



class LightGBM():
    def __init__(self, n_class=0):
        if n_class:
            self.params = {
                        'task': 'train',
                        'boosting_type': 'gbdt',
                        'objective': 'multiclass',
                        'metric': {'multi_logloss'},
                        'num_class': 3,
                        'learning_rate': 0.1,
                        'num_leaves': 23,
                        'min_data_in_leaf': 1,
                        'num_iteration': 100,
                        'verbose': 0
                        }
        else:
            self.params = {
                        'task' : 'train',
                        'boosting_type' : 'gbdt',
                        'objective' : 'regression',
                        'metric' : {'l2'},
                        'num_leaves' : 31,
                        'learning_rate' : 0.1,
                        'feature_fraction' : 0.9,
                        'bagging_fraction' : 0.8,
                        'bagging_freq': 5,
                        'verbose' : 0
                        }
    
    def train(self, x, y):
        self.dtrain = lgb.Dataset(x, y)
        self.model = lgb.train(self.params, 
                               self.dtrain, 
                               early_stopping_rounds=10,
                               num_boost_round=100)

    def predict(self, x, y):
        dtest = lgb.Dataset(x, y, reference=self.dtrain)
        return self.model.predict(dtest)


class catboost():
    def __init__(self):
        pass