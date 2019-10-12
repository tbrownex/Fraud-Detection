import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.mixture import GaussianMixture as GMM

from getConfig  import getConfig
from getArgs     import getArgs
from getData     import getData
from preProcess import preProcess
from getModelParms  import getParms
from removeClass import removeClass

def process(dataDict, config, args):
    #from sklearn.decomposition import PCA
    #pca = PCA(0.99, whiten=True)
    #data = pca.fit_transform(dataDict["trainX"])
    print("{:<20}{}".format("NumComponents", "AIC"))
    numComponents = np.arange(start=20, stop=60, step=20)
    
    models = [GMM(n,covariance_type='full', random_state=0).fit(dataDict["trainX"]) for n in numComponents]
    aics = [model.aic(dataDict["trainX"]) for model in models]
    for x in zip(numComponents, aics):
        print("{:<25}{}".format(x[0], x[1]))
    input()
    
if __name__ == "__main__":
    args = getArgs()
    config = getConfig()
    df = getData(config)
    ''' This preprocessing is common to all the different algos I might try, e.g. XGBoost or standard NN'''
    dataDict = preProcess(df, config, args)
    ''' Train on Negatives only (the '1' is the column for Positives) '''
    dataDict = removeClass(dataDict, 1)
    
    #parmList = getParms("GMM")
    
    start = time.time()
    #results = process(dataDict, parmList, config, args)
    results = process(dataDict, config, args)
    resultsDF = pd.concat(results)
    resultsDF.to_csv("/home/tbrownex/GMMresults.csv", index=False, float_format='%.4f')
    elapsed = (time.time() - start)/60
    print("Elapsed time: {:.1f} minutes".format(elapsed))