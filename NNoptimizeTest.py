import pandas as pd
import numpy as np
import time
import tensorflow as tf

from getConfig  import getConfig
from getArgs import getArgs
from setLogging import setLogging
from getData     import getData
from preProcess import preProcess
from createDataset import createDataset
from getModelParms  import getParms
from NNtest import Model
from getClassScores import getClassScores

def printStats(dataDict):
    ratio = int(dataDict["trainY"].shape[0]/dataDict["trainY"][:,1].sum())
    print("Positive ratio of {}:1".format(ratio))
    input()
    return ratio

def loadParms(p, ratio):
    parmDict = {}
    parmDict['l1Size'] = p[0]
    parmDict['activation'] = p[1]
    parmDict['learningRate'] = p[2]
    parmDict["Lambda"] = p[3]
    parmDict['dropout'] = p[4]
    parmDict['optimizer'] = p[5]
    parmDict["weight"] = ratio
    parmDict["featureCount"] = 2
    return parmDict
    
def saveModel(model, config):
    pickle.dump(model, open(config["modelDir"] + "XGBmodel", 'wb'))

def process(dataDict, parmList, config, args, ratio):
    with tf.name_scope("inputPipeline"):
        trainDS = createDataset(dataDict, config, "train")
        valDS = createDataset(dataDict, config, "val")
        
        iter = tf.data.Iterator.from_structure(trainDS.output_types, tf.compat.v1.data.get_output_shapes(trainDS))
        features, labels = iter.get_next()
        
        trainInit = iter.make_initializer(trainDS)
        valInit = iter.make_initializer(valDS)
    
    for p in parmList:
        parmDict = loadParms(p, ratio)
        nn = Model(parmDict, features, labels)
        
        cost = nn.cost
        train = nn.train
        epochs=10
        
        trainBatches = int(dataDict["trainX"].shape[0]/config["batchSize"])
        valBatches = int(dataDict["valX"].shape[0]/config["batchSize"])
            
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            for e in range(epochs):
                sess.run(trainInit)
                for _ in range(trainBatches):
                    _ = sess.run(train)
                sess.run(valInit)
                actualPositives = 0
                for _ in range(valBatches):
                    l , preds = sess.run([labels, nn._output])
                    scores = getClassScores(l, preds)
                    print(scores)
                    input()
                #print("epoch ", e, " cost: ", valCost/valBatches)
                #np.savetxt("/home/tbrownex/labels.csv", l, delimiter=",")
                #np.savetxt("/home/tbrownex/entropy.csv", c, delimiter=",")
        
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
    setLogging(config)
    
    df = getData(config)
    dataDict = preProcess(df, config, args)
    ratio = printStats(dataDict)
    
    parmList = getParms("NN")
    
    start = time.time()
    results = process(dataDict, parmList, config, args, ratio)
    #results.to_csv("/home/tbrownex/NNresults.csv", index=False)
    elapsed = (time.time() - start)/60
    print("Elapsed time: {:.1f} minutes".format(elapsed))