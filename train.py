import os, sys
import argparse
import sklearn
from methods.boosting import *


def main(args):
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data', default='boston', choices=['boston', 'iris', 'diabetes', 'digits', 'linnerud'])
    parser.add_argument('r', default='XGBoost', choices=['XGBoost', 'LightGBM'])
    args = parser.parse_args()
    main(args)