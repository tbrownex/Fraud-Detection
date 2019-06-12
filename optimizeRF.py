import pandas as pd
import numpy  as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

from getModelParms  import getParms
from oneHot         import oneHot
from getClassErrors import getClassErrors

def saveModel(model, config):
    pickle.dump(model, open(config["modelDir"] + "RFmodel", 'wb'))

def loadParms(p):
    params = {'n_estimators': p[0],\
              'min_samples_split': p[1],\
              'max_depth': p[2],\
              "min_samples_leaf": p[3],\
              "max_features": p[4]}
    return params

def process(dataDict, parms, config):
    best = np.inf
    bestErrors = None
    bestPreds = None
    
    for p in parms:
        params = loadParms(p)
        model  = RandomForestClassifier(**params)
        model.fit(dataDict["trainX"], dataDict["trainY"])
        preds = model.predict(dataDict["testX"])
        errors = getClassErrors(dataDict["testY"], preds)
        ll = errors["ll"]
        if ll < best:
            best = ll
            bestErrors = errors
            bestPreds = preds
            saveModel(model, config)
    return bestErrors, bestPreds
    
def buildRF(dataDict, config):
    parms = getParms("RF")
    ll, bestPreds = process(dataDict, parms, config)
    return ll, bestPreds