import pandas as pd
import numpy as np
import time
import pickle
from sklearn.model_selection import KFold
import xgboost as xgb

from getConfig  import getConfig
from getArgs import getArgs
from setLogging import setLogging
from getData     import getData
from getModelParms  import getParms
from fitXGB import fitXGB
import getXGBpreds as XGB
from getClassScores import getClassScores

def splitLabel(df, config):
    Y = df[config["labelColumn"]]
    del df[config["labelColumn"]]
    X = df
    return X, Y

def formatData(df, config):
    X, Y = splitLabel(df, config)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def printStats(Y):
    ratio = int(Y.shape[0]/Y.sum())
    print("Positive ratio of {}:1".format(ratio))
    return ratio

def loadParms(p, ratio):
    params = {'n_estimators': p[0],
              'learning_rate': p[1],
              'max_depth': p[2],
              'min_child_weight': p[3],
              'colsample_bytree': p[4],
              'subsample': p[5],
              'scale_pos_weight': ratio,
              'gamma': p[6]}
    return params
    
def saveModel(model, config):
    pickle.dump(model, open(config["modelDir"] + "XGBmodel", 'wb'))

def process(X, Y, parmList, config, args, ratio):
    bestScore = np.inf
    dfList = []
    count=1
    
    folds = KFold(n_splits = config["numFolds"], shuffle = True)
    clf = xgb.XGBClassifier(objective = 'binary:logistic')
    
    for p in parmList:
        parms = loadParms(p, ratio)
        totalScore = 0
        for trainIdx, testIdx in folds.split(X):
            trainX, testX = X[trainIdx], X[testIdx]
            trainY, testY = Y[trainIdx], Y[testIdx]
            clf.fit(trainX, trainY)
            preds = clf.predict(testX)
            score = getClassScores(testY, preds)
            totalScore += score["recall"]
        parms["Score"] = round(totalScore/config["numFolds"],3)
        
        tmp = pd.DataFrame.from_records([parms])
        dfList.append(tmp)
        
        if args.save:
            if totalScore < bestScore:
                bestScore = totalScore
                saveModel(model, config)

        print("{} of {}".format(count, len(parmList)))
        count+=1
    return  pd.concat(dfList)

if __name__ == "__main__":
    args = getArgs()
    config = getConfig()
    setLogging(config)
    
    df = getData(config)
    if args.testInd == "test":
        df = df.sample(frac=0.4)
        print(" - Using a fraction of the full data")
    else:
        print(" - Using full dataset")
        
    randSeed = None
    
    X,Y = formatData(df, config)
    ratio = printStats(Y)
    
    parmList = getParms("XGB")
    
    start = time.time()
    results = process(X,Y, parmList, config, args, ratio)
    results.to_csv("/home/tbrownex/XGBresults.csv", index=False)
    elapsed = (time.time() - start)/60
    print("Elapsed time: {:.1f} minutes".format(elapsed))