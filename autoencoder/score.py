import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from getConfig  import getConfig
from getArgs import getArgs
from getData     import getData
from preProcess import preProcess

def getModel(config):
    saver = tf.compat.v1.train.import_meta_graph(config["modelDir"]+"NNmodel.meta")
    return saver

def prepData(df):
    ''' Can't use the regular "preProcessor" module because of some tedious differences so wrote this mini-version '''
    del df["Time"]
    
    svClass =df[config["labelColumn"]]       # Need these values for evaluation later
    del df[config["labelColumn"]]
    
    cols = df.columns
    scaler = MinMaxScaler()
    scaler.fit(df)
    arr = scaler.transform(df)
    df = pd.DataFrame(arr, columns=cols)
    return df, svClass

def computeScore(actuals, preds):
    ''' Compute the error for the recreation of each row. Presumably the higher error are the Positives '''
    diff = actuals-preds
    diffSq = np.array(diff*diff)
    return diffSq.mean(axis=1)

def getTop(scores, num):
    ''' Get the index of the top X highest scores. Highest means most error: means most likely to be a Positive '''
    idx = np.argsort(scores)
    return idx[::-1][:num]    # reverse the sort, then get the top 10

def getMetrics(actualPos, predPos):
    '''
    posIdx: the index of actual positives
    allPos: the index of predicted positives
    '''
    TP = [x for x in predPos if x in actualPos]
    precision = len(TP)/len(predPos)
    recall = len(TP)/len(actualPos)
    return precision, recall
    
def process(nn, dataDict):
    with tf.compat.v1.Session() as sess:
        nn.restore(sess, config["modelDir"]+"NNmodel")
        graph=tf.compat.v1.get_default_graph()
        return sess.run("prediction:0", feed_dict={"input:0": df})

if __name__ == "__main__":
    args = getArgs()
    config = getConfig()
    df = getData(config)
    df, svClass = prepData(df)
    print("\nThere are {} positives in this data\n".format(svClass.sum()))
    
    posIdx = svClass.loc[svClass==1].index
        
    nn = getModel(config)
    
    preds = process(nn, df)
    scores = computeScore(df, preds)
    num = 100
    predPos = getTop(scores, num)
    p,r = getMetrics(posIdx, predPos)
    print("Precision: {:.1%}".format(p))
    print("Recall: {:.1%}".format(r))