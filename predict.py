#!env python3

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
import pandas as pd
import sys
from optparse import OptionParser
import argparse

#######################################################################################################################
train = pd.read_csv('train.csv', index_col=0)
train_y = train.loc[:,['Survived']]

sexnum = {'male':0, 'female':1}
embarked = {'S':0, 'C':1, 'Q':2, np.nan: 3}

train_x = train.loc[:,['Pclass', 'Age', 'Fare']]
train_x['EmbarkedNum'] = train['Embarked'].replace(embarked)
train_x['SexNum'] = train['Sex'].replace(sexnum)

# print(train_y['Survived'].unique())
# plt.figure()
train.hist()
# plt.axhline(0, color='k')

# clf = svm.SVC(gamma=0.001, C=100.)
# clf.fit(train_x, train_y)