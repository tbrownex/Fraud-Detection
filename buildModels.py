import pandas as pd
#from imblearn.over_sampling import SMOTE

from getConfig   import getConfig
from getData     import getData
from splitData   import splitData
from splitLabel  import splitLabel
from RFoptimize  import buildRF
#from optimizeNN  import buildNN
#from optimizeXGB import buildXGB

def getOptimizers():
    optimizers = {}
    optimizers["RF"] = buildRF
    #optimizers["NN"] = buildNN
    #optimizers["XGB"] = buildXGB
    return optimizers

def getModelPreds(dataDict, config):
    '''
    For each entry in "optimizers" call the associated algo and get its predictions
    For each algo, store the best error score and its predictions in a numpy array
    Then compute the average of all the predictions, the "ensemble"
    '''
    optimizers = getOptimizers()
    for typ, module in optimizers.items():
        error, parms = module(dataDict, config)
        #errors, preds = module(SMOTEdata, config)
    return df

'''def calcErrors(df):
DataFrame has a column for each Model; the column holds the predictions
    DataFrame also has a column of Actuals, which are used to compute the error (rmse and mape)
    
    df.set_index("unit", inplace=True)
    return evaluate(df)'''

def formatResults(results):
    import operator
    import collections
    '''
    "results" is a dictionary of dictionaries: d["RF"] has keys "mape" and "rmse"
    First convert the dict of dicts to just a dict
    Then sort by rmse
    '''
    d = {}
    for x in results.keys():
        d[x] = results[x]["rmse"]   # ignore MAPE for now
    l = sorted(d.items(), key=operator.itemgetter(1))   # sort by rmse
    # "l" is a list of tuples (model, rmse) e.g. ("RF", 24.2)
    results = []
    for k, val in l:
        d = {}
        d[k] = val
        results.append(d)
    return results

def writeOutput(df, results, config):
    df.to_csv("predictions.csv")
    
    results = formatResults(results)

if __name__ == "__main__":
    '''
    Run the optimizer routine for each model type (Random Forest, NN, XGB)
    Get the error and predictions of the best model
    '''
    config = getConfig()
    df       = getData(config)
    
    df = df.sample(frac=0.4)
    
    dataDict = splitData(df, config)
    dataDict = splitLabel(dataDict, config)

    df = getModelPreds(dataDict, config)
    #results = calcErrors(df)
    writeOutput(df, results, config)