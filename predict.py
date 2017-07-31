#!env python3

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
import re

FILL_MEDIAN = True

#######################################################################################################################

def prep_df(X_in):
    sexnum = {'male':0, 'female':1}
    embarked = {'S':0, 'C':1, 'Q':2, np.nan: np.nan}
    X_out = X_in.loc[:,['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]
    X_out['EmbarkedNum'] = X_in['Embarked'].replace(embarked)
    X_out['SexNum'] = X_in['Sex'].replace(sexnum)
    X_out['Age'] = X_out['Age'].fillna(X_out['Age'].median() if FILL_MEDIAN else -1)
    X_out['EmbarkedNum'] = X_out['EmbarkedNum'].fillna(X_out['EmbarkedNum'].median() if FILL_MEDIAN else -1)
    X_out['SexNum'] = X_out['SexNum'].fillna(X_out['SexNum'].median() if FILL_MEDIAN else -1)
    X_out['Pclass'] = X_out['Pclass'].fillna(X_out['Pclass'].median() if FILL_MEDIAN else -1)

    nms = [r'.*\sMr\.\s.*',r'.*\sMrs\.\s.*', r'.*\sMiss\.\s.*', r'.*\sMaster\.\s.*',\
    r'.*\sDon\.\s.*', r'.*\sRev\.\s.*', r'.*\sDr\.\s.*', r'.*\sMme\.\s.*', r'.*\sMs\.\s.*',\
    r'.*\sMajor\.\s.*', r'.*\sLady\.\s.*', r'.*\sSir\.\s.*', r'.*\sMlle\.\s.*', r'.*\sCol\.\s.*',\
    r'.*\sCapt\.\s.*']

    l = [[re.match(r, x) for r in nms] for x in X_in['Name']] 
    l = [[x != None for x in lt] for lt in l]
    l = [[x for x,v in enumerate(li) if v] for li in l]
    l = [x[0] if x != [] else len(nms) for x in l]  
    X_out['Name'] = l

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
# gbm = xgb.XGBClassifier(max_depth=16, n_estimators=60, learning_rate=0.05).fit(train_x, np.ravel(train_y.values))
# pred_train_y = gbm.predict(train_x)
# pred_test_y = gbm.predict(test_x)

# clf = svm.SVC(gamma=0.0005, C=100., kernel='rbf')
# clf.fit(train_x, np.ravel(train_y.values))

# pred_train_y = clf.predict(train_x)
# pred_test_y = clf.predict(test_x)

model = CatBoostClassifier(iterations=120, depth=16,l2_leaf_reg=4, learning_rate=0.015, loss_function='Logloss',\
    eval_metric='AUC', verbose=True)
model.fit(train_x, np.ravel(train_y.values), eval_set=(test_x,np.ravel(test_y)), use_best_model=False, verbose=True)
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
test_x = prep_df(test)

pred_test_y = model.predict(test_x)
pred_test_y_df = pd.DataFrame(data=pred_test_y, index=test.index, columns=['Survived'])
pred_test_y_df['Survived'] = pred_test_y_df['Survived'].astype('int')
pred_test_y_df.to_csv('submission.csv')
