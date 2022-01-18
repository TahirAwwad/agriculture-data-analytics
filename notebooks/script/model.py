#!/usr/bin/env python
# coding: utf-8

#!pip install sklearn
#!pip install sklearn
#from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyRegressor
from sklearn.utils import all_estimators
from sklearn import datasets
from sklearn.utils import shuffle
import numpy as np


boston = datasets.load_boston()
X, y = shuffle(boston.data, boston.target, random_state=13)
X = X.astype(np.float32)
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]
reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
print(models)




