import os, sys
import numpy as np
import sklearn.svm as sk
from sklearn import linear_model

class Linear():
    def __init__(self, kernel='rbf', C=1e3, gamma=0.1):
        self.model = linear_model.LinearRegression()

    def train(self, x, y):
        self.model = self.model.fit(x, y)

    def predict(self, x, y):
        return self.model.predict(x)


class SVR():
    def __init__(self, kernel='rbf', C=1e3, gamma=0.1):
        self.model = sk.SVR(kernel=kernel, C=C, gamma=gamma)

    def train(self, x, y):
        self.model = self.model.fit(x, y)

    def predict(self, x, y):
        return self.model.predict(x)
