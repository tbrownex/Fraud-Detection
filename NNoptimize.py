import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import pandas as pd
import numpy as np
import random
import time
import tensorflow as tf

from tensorflow import keras
import logging

from getConfig  import getConfig
from getArgs import getArgs
from setLogging import setLogging
from getData     import getData
from preProcess import preProcess
from getModelParms  import getParms
from generateHyperParms import generateHyperParms
import metrics
from nn import Model

#x = random.randint(1, 1e3)
x = 4455
np.random.seed(x)
tf.random.set_seed(x)

EPOCHS = 100

def printStats(dataDict):
    ratio = int(dataDict["trainY"].shape[0]/dataDict["trainY"].sum())
    print(" - Positive ratio of {}:1".format(ratio))
    return ratio
    
def printInitialMetrics(nn, dataDict):
    results = nn.model.evaluate(dataDict["trainX"], dataDict["trainY"], batch_size=4096, verbose=0)
    print(" - Initial metrics before training:")
    for metric in zip(nn.model.metrics_names, results):
        print(metric)
    '''def saveModel(model, config):
    pickle.dump(model, open(config["modelDir"] + "XGBmodel", 'wb'))'''

def runModel(dataDict, parmDict,  config, ratio, l):
    # Create the network
    nn = Model(parmDict, config, ratio)
    ''' When you are happy with the initial weights (low initial loss) save them off 
    printInitialMetrics(nn, dataDict)
    nn.model.save_weights("initializedWeights")
    input("weights saved. <Enter> to continue")'''
    
    clsWeights = {0: 1, 1: parmDict["clsWeight"]}
    clsWeights = {0: 0.5, 1: 289}
    
    TB = keras.callbacks.TensorBoard(log_dir="/home/tbrownex/TF/Tensorboard/nn/" + "loop_" + str(l))
    earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=10,
                                              verbose=0,
                                              mode='auto',
                                              restore_best_weights=True)
    
    
    results = nn.model.fit(
        dataDict["trainX"], dataDict["trainY"],
        batch_size=parmDict["batchSize"],
        epochs=EPOCHS,
        validation_data=(dataDict["valX"], dataDict["valY"]),
        verbose=0,
        class_weight=clsWeights,
        callbacks=[earlyStop])
    return results

def evalModel(nn, dataDict, parmDict):
    ''' Run Test data through the model and get the evaluation metrics '''
    results = nn.model.evaluate(dataDict["testX"], dataDict["testY"])
    resultDict = dict(zip(nn.model.metrics_names, results))
    # G-score is an alternative metric for evaluation of imbalanced datasets
    resultDict["G-score"] = metrics.calcG(resultDict["precision"], resultDict["recall"])
    resultDict["F1-score"] = metrics.calcF1(resultDict["precision"], resultDict["recall"])
    
    keys = ["auc", "G-score",   "F1-score", "precision", "recall"]
    for k in keys:
        resultDict[k] = round(resultDict[k], 3)
    return {**parmDict, **resultDict}

def process(dataDict, parms, numLoops, config):
    ratio = printStats(dataDict)
    dfList = []   # This will store the result for each set of parms
    
    for l in range(numLoops):
        l += 1
        if l %10==0: print(l)
        parmDict = generateHyperParms(parms)
        logging.info(parmDict)
        nn = runModel(dataDict, parmDict, config, ratio, l)
        results = evalModel(nn, dataDict, parmDict)
        dfList.append(pd.DataFrame([results]))
    return pd.concat(dfList)

if __name__ == "__main__":
    args = getArgs()
    config = getConfig()
    setLogging(config)
    
    df = getData(config)
    dataDict = preProcess(df, config, args)
    
    parms = getParms("NN")
    
    start = time.time()
    numLoops = args.numLoops
    results = process(dataDict, parms, numLoops, config)
    results.to_csv("/home/tbrownex/NNresults.csv", index=False)
    elapsed = (time.time() - start)/60
    print("Elapsed time: {:.1f} minutes".format(elapsed))