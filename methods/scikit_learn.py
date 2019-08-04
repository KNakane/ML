import os, sys
import numpy as np
import sklearn
import sklearn.svm as sk
from sklearn import linear_model

class Linear():
    def __init__(self, kernel=None, C=None, gamma=None):
        self.model = linear_model.LinearRegression()

    def train(self, x, y):
        self.model = self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def loss(self, pred, grand_truth):
        return np.sqrt(sklearn.metrics.mean_squared_error(grand_truth, pred))

class Ridge():
    def __init__(self, kernel=None, C=None, gamma=None):
        self.model = linear_model.Ridge()

    def train(self, x, y):
        self.model = self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def loss(self, pred, grand_truth):
        return np.sqrt(sklearn.metrics.mean_squared_error(grand_truth, pred))


class SVR():
    def __init__(self, kernel='rbf', C=1e3, gamma=0.1):
        self.model = sk.SVR(kernel=kernel, C=C, gamma=gamma)

    def train(self, x, y):
        self.model = self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def loss(self, pred, grand_truth):
        return np.sqrt(sklearn.metrics.mean_squared_error(grand_truth, pred))