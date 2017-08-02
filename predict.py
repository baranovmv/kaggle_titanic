#!env python3

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn import ensemble
from sklearn import neural_network
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, ShuffleSplit
import xgboost as xgb
# from catboost import CatBoostClassifier
import pandas as pd
import sys
from optparse import OptionParser
import argparse
import re

FILL_MEDIAN = True

#######################################################################################################################

def feature_importance(X, clf):
    features = pd.DataFrame()
    features['feature'] = X.columns
    features['importance'] = clf.feature_importances_
    features.set_index('feature', inplace=True)
    features.sort_values(by=['importance'], inplace=True)
    features.plot(kind='barh')
    plt.show()

def prep_df(X_in):
    sexnum = {'male':0, 'female':1}
    embarked = {'S':0, 'C':1, 'Q':2, np.nan: np.nan}
    nms = [r'.*\sMr\.\s.*',r'.*\sMrs\.\s.*', r'.*\sMiss\.\s.*', r'.*\sMaster\.\s.*',\
    r'.*\sDon\.\s.*', r'.*\sRev\.\s.*', r'.*\sDr\.\s.*', r'.*\sMme\.\s.*', r'.*\sMs\.\s.*',\
    r'.*\sMajor\.\s.*', r'.*\sLady\.\s.*', r'.*\sSir\.\s.*', r'.*\sMlle\.\s.*', r'.*\sCol\.\s.*',\
    r'.*\sCapt\.\s.*']
    X_out = X_in.loc[:,['Pclass','SibSp', 'Parch', 'Age', 'Fare']]
    # X_out['FamSize'] = X_in['SibSp'] + X_in['Parch']
    # X_out['EmbarkedNum'] = X_in['Embarked'].replace(embarked)
    X_out['male'] = X_in['Sex'] == 'male'
    X_out['male'].replace({True:1, False:0}, inplace=True)
    X_out['female'] = X_in['Sex'] == 'female'
    X_out['female'].replace({True:1, False:0}, inplace=True)
    X_out['Fare'] = X_out['Fare'].fillna(X_out['Fare'].median() if FILL_MEDIAN else -1)
    # X_out['SexNum'] = X_in['Sex'].replace(sexnum)
    # X_out['Age'] = X_out['Age'].fillna(X_out['Age'].median() if FILL_MEDIAN else -1)
    # X_out['EmbarkedNum'] = X_out['EmbarkedNum'].fillna(X_out['EmbarkedNum'].median() if FILL_MEDIAN else -1)
    # X_out['SexNum'] = X_out['SexNum'].fillna(X_out['SexNum'].median() if FILL_MEDIAN else -1)
    X_out['Pclass'] = X_out['Pclass'].fillna(X_out['Pclass'].median() if FILL_MEDIAN else -1)
    l = [[re.match(r, x) for r in nms] for x in X_in['Name']] 
    l = [[x != None for x in lt] for lt in l]
    l = [[x for x,v in enumerate(li) if v] for li in l]
    l = [x[0] if x != [] else len(nms) for x in l]  
    X_out['Name'] = l
    return X_out

def make_ages(X, age_regr=None):
    # Work out missing age values
    x_age = X.loc[X['Age'].notnull(),['Pclass','SibSp', 'Parch', 'Fare','male', 'female', 'Name']]
    x_age_ = X.loc[X['Age'].isnull(),['Pclass','SibSp', 'Parch', 'Fare','male', 'female', 'Name']]
    y_age = np.ravel(X.loc[X['Age'].notnull(),['Age']].values)
    if not age_regr:
        age_regr = ensemble.RandomForestRegressor(n_estimators=200)
        # age_regr = neural_network.MLPRegressor(hidden_layer_sizes=(8,64,64), activation='relu', learning_rate_init=0.01)
        age_regr.fit(x_age,y_age)
    y_err = age_regr.predict(x_age)-y_age
    print('Age regression MSE: {:.2}'.format(np.sqrt(((y_err) ** 2).mean())))
    # plt.hist(y_err)
    # plt.show()
    # sys.exit(0)
    Xret = X
    Xret.loc[X['Age'].isnull(),['Age']] = age_regr.predict(x_age_)
    return Xret, age_regr

#######################################################################################################################
train = pd.read_csv('train.csv', index_col=0)

Y = train.loc[:,['Survived']]
X = prep_df(train)
X, age_regr = make_ages(X)

#######################################################################################################################

n = 10
accuracy_test = np.zeros(n)
gbm = xgb.XGBClassifier(max_depth=2, n_estimators=100, learning_rate=0.1)

for i in range(n):
    train_ind = np.random.rand(len(X)) < 0.8
    test_ind = ~train_ind

    train_x = X[train_ind].as_matrix()
    train_y = Y[train_ind]

    test_x = X[test_ind].as_matrix()
    test_y = Y[test_ind]

    # You can experiment with many other options here, using the same .fit() and .predict()
    # methods; see http://scikit-learn.org
    # This example uses the current build of XGBoost, from https://github.com/dmlc/xgboost
    gbm.fit(train_x, np.ravel(train_y.values))

    pred_train_y = gbm.predict(train_x)
    pred_test_y = gbm.predict(test_x)

    accuracy_train = accuracy_score(train_y, pred_train_y)
    accuracy_test[i] = accuracy_score(test_y, pred_test_y)
    f1_test = f1_score(test_y, pred_test_y)
    print("Training set accuracy: {:.2%}".format(accuracy_train))
    print("Test set accuracy: {:.2%}".format(accuracy_test[i]))
    print("Test set F1: {:.2%}".format(f1_test))
# clf = ensemble.GradientBoostingClassifier(learning_rate=0.05, n_estimators=200, max_depth=2)
# cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
# scores = cross_val_score(clf, train_x, np.ravel(train_y.values), cv=cv)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# clf.fit(train_x,np.ravel(train_y.values))
# feature_importance(X,clf)

print('Mean test set accuracy: {:.2%}'.format(accuracy_test.mean()))

#######################################################################################################################
test = pd.read_csv('test.csv', index_col=0)
test_x = prep_df(test)
test_x, _ = make_ages(test_x, age_regr)

pred_test_y = gbm.predict(test_x.as_matrix())
# pred_test_y = clf.predict(test_x.as_matrix())
pred_test_y_df = pd.DataFrame(data=pred_test_y, index=test.index, columns=['Survived'])
pred_test_y_df['Survived'] = pred_test_y_df['Survived'].astype('int')
pred_test_y_df.to_csv('submission.csv')
