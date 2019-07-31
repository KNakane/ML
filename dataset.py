import os, sys
from sklearn import datasets
from sklearn.model_selection import train_test_split

class Dataset():
    def __init__(self, name):
        self.name = name
        if self.name == 'boston':
            dataset = datasets.load_boston()
            self.x = dataset['data']
            self.y = dataset['target']
            self.out_dim = 1
        elif self.name == 'iris':
            dataset = datasets.load_iris()
            self.x = dataset['data']
            self.y = dataset['target']
            self.out_dim = 3
        elif self.name == 'diabetes':
            dataset = datasets.load_diabetes()
            self.x = dataset['data']
            self.y = dataset['target']
            self.out_dim = 1
        elif self.name == 'digits':
            dataset = datasets.load_digits(n_class=10)
            self.x = dataset['data']
            self.y = dataset['target']
            self.out_dim = 10
        elif self.name == 'linnerud':
            dataset = datasets.load_linnerud()
            self.x = dataset['data']
            self.y = dataset['target']
            self.out_dim = 3
        else:
            NotImplementedError()

    def load(self, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size=test_size, random_state=0)

if __name__ == "__main__":
    data = Dataset('boston')
    data.load()