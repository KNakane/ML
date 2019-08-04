import os, sys
import argparse
import sklearn
import matplotlib.pyplot as plt
from dataset import Dataset
from methods.boosting import *
from methods.scikit_learn import *


def main(args):
    # Load dataset
    data = Dataset(args.data)
    (train_x, train_y), (test_x, test_y) = data.load()

    # Load model
    model = eval(args.r)()
    
    # Train model
    model.train(train_x, train_y)

    # Predict test data
    train_logits = model.predict(train_x)
    predict = model.predict(test_x)

    # calculate rmse
    train_rmse = model.loss(train_logits, train_y)
    test_rmse = model.loss(predict, test_y)

    print("Train RMSE : {}  Test RMSE : {}".format(train_rmse, test_rmse))

    plt.figure(figsize=[10, 5])
    plt.scatter(train_y, train_logits, c='r', label='train_data')
    plt.scatter(test_y, predict, c='g', label='test_data')
    plt.xlabel('grand truth')
    plt.ylabel('predict')
    plt.title('Regression')
    plt.legend()
    plt.show()

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='boston', choices=['boston', 'iris', 'diabetes', 'digits', 'linnerud'])
    parser.add_argument('--r', default='Linear', choices=['Linear', 'SVR', 'XGBoost', 'LightGBM'])
    args = parser.parse_args()
    main(args)