import pandas as pd
import numpy as np
import time
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from getConfig  import getConfig
from getArgs import getArgs
from getData     import getData
from preProcess import preProcess
from getModelParms  import getParms
from nnKeras import Model as kerasModel
from nnNative import Model as native
from removeLabels import removeLabels
from removeClass import removeClass

EPOCHS = 1

def loadParms(p, parmDict):
    parmDict['l1Size'] = p[0]
    parmDict['l2Size'] = p[1]
    parmDict['l3Size'] = p[2]
    parmDict['l4Size'] = parmDict['l2Size']
    parmDict['l5Size'] = parmDict['l1Size']
    parmDict['activation'] = p[3]
    parmDict['learningRate'] = p[4]
    parmDict['dropout'] = p[5]
    parmDict['optimizer'] = p[6]
    return parmDict

def printHistory(history):
    print("Epoch     ValidationLoss")
    for e,l in enumerate(history.history["val_loss"]):
        print("{:<10}{:.3f}".format(e,l))
              
def save(actuals, preds):
    actuals.to_csv("/home/tbrownex/actuals.csv", index=False)
    preds = pd.DataFrame(preds)
    preds.to_csv("/home/tbrownex/preds.csv", index=False)

def updateTB(tbWriter, tag, val, step):
    summary = tf.Summary(value=[
        tf.Summary.Value(
            tag=tag,
            simple_value=val)])
    tbWriter.add_summary(summary, step)
    
def saveModel(model, config):
    pickle.dump(model, open(config["modelDir"] + "NNnative", 'wb'))

def processNative(dataDict, parmList, config, args):
    parmDict = {}
    parmDict["featureCount"] = dataDict["trainX"].shape[1]
    
    BATCH = config['batchSize']
    
    numBatches = int(dataDict["trainX"].shape[0]/config["batchSize"])
    
    results = []               # Holds the final cost (against Test) for each set of parameters
    for p in parmList:
        parmDict = loadParms(p, parmDict)
        
        # Build the network and get ops for Train and Cost functions
        nn = native(parmDict)
        cost = nn.cost
        train = nn.train
        
        # for Tensorboard
        tbCounter = 0
        now = datetime.now()
        
        with tf.compat.v1.Session() as sess:
            saver = tf.compat.v1.train.Saver()
            tbWriter = tf.summary.FileWriter(config["TBdir"]+"/nn/"+ now.strftime("%Y%m%d-%H%M"), sess.graph)
            sess.run(tf.compat.v1.global_variables_initializer())
            for e in range(EPOCHS):
                a, b = shuffle(dataDict['trainX'],dataDict['trainX'])
                for j in range(numBatches):
                    x_mini = a[j*BATCH:j*BATCH+BATCH]
                    y_mini = b[j*BATCH:j*BATCH+BATCH]
                    _ = sess.run(train, feed_dict = {nn.X: x_mini, nn.y_: y_mini})
                    if j % 50 ==0:
                        valCost = sess.run(cost, feed_dict = {nn.X: dataDict["valX"], nn.y_: dataDict["valX"]})
                        updateTB(tbWriter, "ValidationCost", valCost, tbCounter)
                        tbCounter +=1
                # end of epoch
                
            # end of all epochs
            finalCost = sess.run(cost, feed_dict = {nn.X: dataDict["testX"], nn.y_: dataDict["testX"]})
            parmDict["finalCost"] = round(finalCost,4)
            tmp = pd.DataFrame.from_dict([parmDict])
            results.append(tmp)
            saver.save(sess, config["modelDir"]+"NNmodel")
        tbWriter.close()
    return results
    
def processKeras(dataDict, parmList, config, args):    
    parmDict = {}
    parmDict["featureCount"] = dataDict["trainX"].shape[1]
    
    tbCallback = keras.callbacks.TensorBoard(log_dir=config["TBdir"]+"/keras")
        
    results = []               # Holds the final cost (against Test) for each set of parameters    
    for p in parmList:
        parmDict = loadParms(p, parmDict)
        nn = kerasModel(parmDict)
        
        history = nn.model.fit(
            dataDict["trainX"],
            dataDict["trainX"],
            batch_size=config["batchSize"],
            epochs=EPOCHS,
            validation_data=(dataDict["valX"], dataDict["valX"]),
            callbacks=[tbCallback],
            verbose=0
        )
        #printHistory(history)
        #preds = nn.model.predict(dataDict["testX"])
        #save(dataDict["testX"], preds)
        
        mse = nn.model.evaluate(dataDict["testX"], dataDict["testX"])
        parmDict["finalCost"] = round(mse, 4)
        tmp = pd.DataFrame.from_dict([parmDict])
        results.append(tmp)
        
    return results

if __name__ == "__main__":
    args = getArgs()
    config = getConfig()
    df = getData(config)
    ''' This preprocessing is common to all the different algos I might try, e.g. XGBoost or standard NN'''
    dataDict = preProcess(df, config, args)
    ''' This is for autoencoder only '''
    dataDict = removeClass(dataDict)
    dataDict = removeLabels(dataDict)
    
    parmList = getParms("AE")
    
    start = time.time()
    if args.networkType == "keras":
        results = processKeras(dataDict, parmList, config, args)
        resultsDF = pd.concat(results)
        resultsDF.to_csv("/home/tbrownex/KERASresults.csv", index=False)
    else:
        results = processNative(dataDict, parmList, config, args)
        resultsDF = pd.concat(results)
        resultsDF.to_csv("/home/tbrownex/NNresults.csv", index=False, float_format='%.4f')
    elapsed = (time.time() - start)/60
    print("Elapsed time: {:.1f} minutes".format(elapsed))