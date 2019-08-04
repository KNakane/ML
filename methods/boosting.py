import os, sys
import numpy as np
import sklearn
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt

class XGBoost():
    def __init__(self, n_class=0):
        if n_class:
            self.params = {
                'objective': 'multiclass', # 多値分類問題
                'num_class': 3
                }
        else:
            self.params = {
                "objective":"reg:linear",
                'colsample_bytree': 0.3,
                'learning_rate': 0.1,
                'max_depth': 5,
                'alpha': 10
                }
    
    def train(self, x, y):
        #dtrain = xgb.DMatrix(x, label=y)
        self.model = xgb.XGBRegressor()
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def loss(self, pred, grand_truth):
        return np.sqrt(sklearn.metrics.mean_squared_error(grand_truth, pred))

    def plot_tree(self):
        xgb.plot_tree(self.model, num_trees=1)
        plt.show()


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
        #self.dtrain = lgb.Dataset(x, y)
        self.model = lgb.LGBMRegressor()
        self.model.fit(x, y)

    def predict(self, x):
        #dtest = lgb.Dataset(x, y, reference=self.dtrain)
        return self.model.predict(x)

    def loss(self, pred, grand_truth):
        return np.sqrt(sklearn.metrics.mean_squared_error(grand_truth, pred))


class catboost():
    def __init__(self):
        pass