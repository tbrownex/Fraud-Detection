import pandas as pd
import numpy as np
import time
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
import xgboost as xgb
from scipy import stats

from getConfig  import getConfig
from getData     import getData

def splitLabel(df, config):
    Y = df[config["labelColumn"]]
    del df[config["labelColumn"]]
    X = df
    return X, Y

def printStats(Y):
    ratio = int(Y.shape[0]/Y.sum())
    print("Positive ratio of {}:1".format(ratio))
    return ratio

def formatData(df, config):
    X, Y = splitLabel(df, config)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def getClassifierparms(ratio):
    d = {}
    d['n_estimators'] = stats.randint(100, 200)
    d['learning_rate'] = stats.uniform(5e-2, 2e-2)
    d['subsample'] = [.8, 1.0]
    d['max_depth'] = [3]
    d['colsample_bytree'] = [0.5, 0.6, 0.7, 0.8, 0.9]
    d['min_child_weight'] = [3,5,7,9]
    d['scale_pos_weight'] = [ratio]
    return d

def getSearchparms(parms):
    parms["n_iter"] = 20
    parms["scoring"] = 'recall'
    parms["error_score"] = 0
    parms["verbose"] = 1
    parms["n_jobs"] = -1
    return parms

def process(X, Y, config, ratio):
    xgbClf = xgb.XGBClassifier(objective = 'binary:logistic')
    parms = {}
    parms["estimator"] = xgbClf
    clfParms = getClassifierparms(ratio)
    parms["param_distributions"] = clfParms
    parms = getSearchparms(parms)
    clf = RandomizedSearchCV(**parms)
    
    numFolds = 5
    folds = KFold(n_splits = numFolds, shuffle = True)
    
    estimators = []
    results = np.zeros(len(X))
    score = 0.0
    for train_index, test_index in folds.split(X):
        xTrain, xTest = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        clf.fit(xTrain, y_train)
        
        estimators.append(clf.best_estimator_)
    
    print("{:>6}{:>6}{:>12}{:>6}{:>11}{:>10}".\
          format("Trees", "LR","Subsample", "Depth", "ColSample", "MinChild" ))
    for est in estimators:
        print("{:>6}{:>7.3f}{:>10.2f}{:>6}{:>13.2f}{:>8}".\
          format(est.n_estimators, est.learning_rate, est.subsample, est.max_depth, est.colsample_bytree, est.min_child_weight ))

if __name__ == "__main__":
    config = getConfig()
    randSeed = None
    
    df = getData(config)
    df = df.sample(frac=0.5, random_state =randSeed)
    X,Y = formatData(df, config)
    ratio = printStats(Y)
    
    start = time.time()
    process(X, Y, config, ratio)
    elapsed = (time.time() - start)/60
    print("Elapsed time: {:.1f} minutes".format(elapsed))