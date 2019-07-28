import os, sys
from sklearn import datasets
from sklearn.model_selection import train_test_split

class Dataset():
    def __init__(self, name):
        self.name = name

    def load(self):
        pass