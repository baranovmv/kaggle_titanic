#!env python3

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
import pandas as pd
import sys
from optparse import OptionParser
import argparse

FILL_MEDIAN = True

#######################################################################################################################
train = pd.read_csv('train.csv', index_col=0)
Y = train.loc[:,['Survived']]

sexnum = {'male':0, 'female':1}
embarked = {'S':0, 'C':1, 'Q':2, np.nan: np.nan}

X = train.loc[:,['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]
X['EmbarkedNum'] = train['Embarked'].replace(embarked)
X['SexNum'] = train['Sex'].replace(sexnum)

X['Age'] = X['Age'].fillna(X['Age'].median() if FILL_MEDIAN else -1)
X['EmbarkedNum'] = X['EmbarkedNum'].fillna(X['EmbarkedNum'].median() if FILL_MEDIAN else -1)
X['SexNum'] = X['SexNum'].fillna(X['SexNum'].median() if FILL_MEDIAN else -1)
X['Pclass'] = X['Pclass'].fillna(X['Pclass'].median() if FILL_MEDIAN else -1)

train_ind = np.random.rand(len(X)) < 0.8
test_ind = ~train_ind

train_x = X[train_ind]
train_y = Y[train_ind]

test_x = X[test_ind]
test_y = Y[test_ind]

clf = svm.SVC(gamma=0.0005, C=100., kernel='rbf')
clf.fit(train_x, np.ravel(train_y.values))

pred_train_y = clf.predict(train_x)
pred_test_y = clf.predict(test_x)
accuracy_train = accuracy_score(train_y, pred_train_y)
accuracy_test = accuracy_score(test_y, pred_test_y)
print("Training set accuracy: {:.2%}".format(accuracy_train))
print("Test set accuracy: {:.2%}".format(accuracy_test))
