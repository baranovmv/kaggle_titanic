#!env python3

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
from catboost import CatBoostClassifier
import pandas as pd
import sys
from optparse import OptionParser
import argparse

FILL_MEDIAN = True

#######################################################################################################################

def prep_df(X_in):
    sexnum = {'male':0, 'female':1}
    embarked = {'S':0, 'C':1, 'Q':2, np.nan: np.nan}

    X_out = X_in.loc[:,['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]
    X_out['EmbarkedNum'] = X_in['Embarked'].replace(embarked)
    X_out['SexNum'] = X_in['Sex'].replace(sexnum)

    X_out['Age'] = X_out['Age'].fillna(X_out['Age'].mean() if FILL_MEDIAN else -1)
    X_out['EmbarkedNum'] = X_out['EmbarkedNum'].fillna(X_out['EmbarkedNum'].mean() if FILL_MEDIAN else -1)
    X_out['SexNum'] = X_out['SexNum'].fillna(X_out['SexNum'].mean() if FILL_MEDIAN else -1)
    X_out['Pclass'] = X_out['Pclass'].fillna(X_out['Pclass'].mean() if FILL_MEDIAN else -1)

    return X_out

#######################################################################################################################
train = pd.read_csv('train.csv', index_col=0)
Y = train.loc[:,['Survived']]

X = prep_df(train)

train_ind = np.random.rand(len(X)) < 0.8
test_ind = ~train_ind

train_x = X[train_ind].as_matrix()
train_y = Y[train_ind]

test_x = X[test_ind].as_matrix()
test_y = Y[test_ind]

# You can experiment with many other options here, using the same .fit() and .predict()
# methods; see http://scikit-learn.org
# This example uses the current build of XGBoost, from https://github.com/dmlc/xgboost
# gbm = xgb.XGBClassifier(max_depth=4, n_estimators=200, learning_rate=0.003).fit(train_x, np.ravel(train_y.values))
# pred_train_y = gbm.predict(train_x)
# pred_test_y = gbm.predict(test_x)

# clf = svm.SVC(gamma=0.0005, C=100., kernel='rbf')
# clf.fit(train_x, np.ravel(train_y.values))

# pred_train_y = clf.predict(train_x)
# pred_test_y = clf.predict(test_x)

model = CatBoostClassifier(iterations=30, depth=4, learning_rate=0.01, loss_function='Logloss', verbose=False)
model.fit(train_x, np.ravel(train_y.values), verbose=True)
pred_train_y = model.predict(train_x)
pred_test_y = model.predict(test_x)

accuracy_train = accuracy_score(train_y, pred_train_y)
accuracy_test = accuracy_score(test_y, pred_test_y)
f1_test = f1_score(test_y, pred_test_y)
print("Training set accuracy: {:.2%}".format(accuracy_train))
print("Test set accuracy: {:.2%}".format(accuracy_test))
print("Test set F1: {:.2%}".format(f1_test))

#######################################################################################################################
test = pd.read_csv('test.csv', index_col=0)

