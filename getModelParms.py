import itertools
import tensorflow as tf

''' This is the set of parameters we're "gridsearching" when optimizing the algos '''

def getParms(typ):
    if typ == "RF":
        nEstimators      = [20,60]
        min_samples_split = [25,50]
        max_depth         = [5,15]
        min_samples_leaf  = [4,12]
        max_features      = [ 0.6, 0.9]
        return list(itertools.product(nEstimators,
                                      min_samples_split,
                                      max_depth,
                                      min_samples_leaf,
                                      max_features))
    elif typ == "XGB":
        nEstimators = [100]
        learningRate = [0.05]
        maxDepth = [3]
        min_child_weight = [7]
        colsample_bytree  = [ 0.7]
        subsample = [1.0]
        gamma  = [0]
        return list(itertools.product(nEstimators,
                                      learningRate,
                                      maxDepth,
                                      min_child_weight,
                                      colsample_bytree,
                                      subsample,
                                      gamma))
    elif typ == "NN":
        L1Size = [256]
        activation = [tf.nn.relu]
        learningRate = [1e-2]
        Lambda = [0.]
        dropout = [0.3]
        optimizer = ["Adam"]
        return list(itertools.product(L1Size,
                                      activation,
                                      learningRate,
                                      Lambda,
                                      dropout,
                                      optimizer))
    elif typ == "AE":
        L1Size      = [48]
        L2Size      = [24]
        L3Size      = [12]
        activation   = ["relu"]
        learningRate = [5e-4]
        dropout      = [0.0]
        optimizer    = ["adam"]
        return list(itertools.product(L1Size,
                                      L2Size,
                                      L3Size,
                                      activation,
                                      learningRate,
                                      dropout,
                                      optimizer))