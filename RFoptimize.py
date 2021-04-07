import pandas as pd
import numpy as np
import time
import pickle

from getConfig  import getConfig
from getArgs import getArgs
from setLogging import setLogging
from getData     import getData
from splitData   import splitData
from splitLabel  import splitLabel
from getModelParms  import getParms
import getRFpreds as RF
from getClassErrors import getClassErrors

def loadParms(p):
    params = {'n_estimators': p[0],\
              'min_samples_split': p[1],\
              'max_depth': p[2],\
              "min_samples_leaf": p[3],\
              "max_features": p[4]}
    return params

def saveModel(model, config):
    pickle.dump(model, open(config["modelDir"] + "RFmodel", 'wb'))
    
def process(dataDict, parmList, config, args):
    bestError = np.inf
    bestParms = None
    dfList = []
    count=1
    
    for p in parmList:
        parms = loadParms(p)
        model, preds = RF.predict(parms, dataDict)
        errors = getClassErrors(dataDict["testY"], preds)
        error = round(errors["recall"],3)
        parms["score"] = error
        tmp = pd.DataFrame.from_records([parms])
        dfList.append(tmp)
        
        if args.save:
            if error < bestError:
                bestParms = parms
                bestError = error
                saveModel(model, config)

        print("{} of {}".format(count, len(parmList)))
        count+=1
    return  pd.concat(dfList)

if __name__ == "__main__":
    args = getArgs()
    config = getConfig()
    setLogging(config)
    
    df = getData(config)
    dataDict = preProcess(df, config, args)
    ratio = printStats(dataDict)
    input()
    
    parmList = getParms("RF")
    
    start = time.time()
    results = process(dataDict, parmList, config, args, ratio)
    #results.to_csv("/home/tbrownex/RFresults.csv", index=False)
    elapsed = (time.time() - start)/60
    print("Elapsed time: {:.1f} minutes".format(elapsed))