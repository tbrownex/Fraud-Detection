import pandas as pd
import numpy as np
import time
import tensorflow as tf
from tensorflow import keras

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from getConfig  import getConfig
from getArgs import getArgs
from getData     import getData
from preProcess import preProcess
from createDataset import createDataset
from getModelParms  import getParms
from nnKeras import Model as kerasModel
from nnNative import Model as native

EPOCHS = 10

def removePositives(dataDict):
    ''' Train the encoder on Negatives only '''
    before = dataDict["trainX"].shape[0]
    idx = dataDict["trainY"][:,0]==1        # Column 0 =1 are the negatives
    dataDict["trainX"] = dataDict["trainX"].loc[idx]
    print(" - {} rows removed (positives) from 'train'".format(before - dataDict["trainX"].shape[0]))
    idx = dataDict["valY"][:,0]==1
    dataDict["valX"] = dataDict["valX"].loc[idx]
    idx = dataDict["testY"][:,0]==1
    dataDict["testX"] = dataDict["testX"].loc[idx]
    return dataDict

def removeLabels(dataDict):
    ''' No labels for an Autoencoder; train on features only '''
    del dataDict["trainY"]
    del dataDict["valY"]
    del dataDict["testY"]
    return dataDict

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
    pickle.dump(model, open(config["modelDir"] + "XGBmodel", 'wb'))

def processNative(dataDict, parmList, config, args):
    parmDict = {}
    parmDict["featureCount"] = dataDict["trainX"].shape[1]
    
    with tf.name_scope("inputPipeline"):
        trainDS = createDataset(dataDict, config, "train")
        valDS = createDataset(dataDict, config, "val")
        testDS = createDataset(dataDict, config, "test")
        
        iter = tf.compat.v1.data.Iterator.from_structure(trainDS.output_types, tf.compat.v1.data.get_output_shapes(trainDS))
        # "labels" here is just the features repeated (we're trying to recreate the input)
        features, labels = iter.get_next()
        
        trainInit = iter.make_initializer(trainDS)
        valInit = iter.make_initializer(valDS)
        testInit = iter.make_initializer(testDS)
    
    results = []               # Holds the final cost (against Test) for each set of parameters    
    for p in parmList:
        parmDict = loadParms(p, parmDict)
        
        trainBatches = int(dataDict["trainX"].shape[0]/config["batchSize"])
        valBatches = int(dataDict["valX"].shape[0]/config["batchSize"])
        testBatches = int(dataDict["testX"].shape[0]/config["batchSize"])
        
        with tf.compat.v1.Session() as sess:
            nn = native(parmDict, features, labels)
            cost = nn.cost
            train = nn.train
            
            tbCounter = 0
            tbWriter = tf.summary.FileWriter(config["TBdir"]+"/nn", sess.graph)
            sess.run(tf.compat.v1.global_variables_initializer())
            for e in range(EPOCHS):
                sess.run(trainInit)
                for z in range(trainBatches):
                    _ = sess.run(train)
                    
                sess.run(valInit)    # end of epoch
                total = 0
                for _ in range(valBatches):
                    total += sess.run(cost)
                updateTB(tbWriter, "ValidationCost", total/valBatches, tbCounter)
                tbCounter +=1
                sess.run(trainInit)
                
            sess.run(testInit)     # end of all epochs
            total = 0
            for _ in range(testBatches):
                total += sess.run(cost)
            parmDict["finalCost"] = round(total/testBatches, 3)
            tmp = pd.DataFrame.from_dict([parmDict])
            results.append(tmp)
        tbWriter.close()
    return results
    
def processKeras(dataDict, parmList, config, args):    
    parmDict = {}
    parmDict["featureCount"] = dataDict["trainX"].shape[1]
    
    tbCallback = keras.callbacks.TensorBoard(log_dir=config["TBdir"]+"/keras")
        
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
        preds = nn.model.predict(dataDict["testX"])
        
        save(dataDict["testX"], preds)
        results = nn.model.evaluate(dataDict["testX"], dataDict["testX"])
        print("Final: {:.3f}".format(results))
        
        
        
    '''bestScore = np.inf
    dfList = []
    count=1
    
    for p in parmList:
        parmDict = loadParms(p, ratio)
        
        lift = run(dataDict, parmDict, config)
        
        tup = (count, parm_dict, lift)
        results.append(tup)
        count +=1
        
        tmp = pd.DataFrame.from_records([parms])
        dfList.append(tmp)
        
        if args.save:
            if totalScore < bestScore:
                bestScore = totalScore
                saveModel(model, config)

        print("{} of {}".format(count, len(parmList)))
        count+=1'''
    return
    #return  pd.concat(dfList)

if __name__ == "__main__":
    args = getArgs()
    config = getConfig()
    
    df = getData(config)
    
    ''' This preprocessing is common to all the different algos I might try, e.g. XGBoost or standard NN'''
    dataDict = preProcess(df, config, args)
    ''' This is for autoencoder only '''
    dataDict = removePositives(dataDict)
    dataDict = removeLabels(dataDict)
    
    parmList = getParms("AE")
    
    start = time.time()
    if args.networkType == "keras":
        results = processKeras(dataDict, parmList, config, args)
    else:
        results = processNative(dataDict, parmList, config, args)
    resultsDF = pd.concat(results)
    resultsDF.to_csv("/home/tbrownex/NNresults.csv", index=False)
    elapsed = (time.time() - start)/60
    print("Elapsed time: {:.1f} minutes".format(elapsed))