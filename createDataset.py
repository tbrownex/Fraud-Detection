import tensorflow as tf

def createTrainDS(dataDict, config):
    ds = tf.data.Dataset.from_tensor_slices((dataDict["trainX"], dataDict["trainY"]))
    ds = ds.shuffle(buffer_size=100000, reshuffle_each_iteration=True)
    return ds.batch(config["batchSize"])

def createValDS(dataDict, config):
    ''' You need the Validation set in batches due to sharing the iterator: iterator requires fixed shape (batchsize in 1st dimension)'''
    ds = tf.data.Dataset.from_tensor_slices((dataDict["valX"], dataDict["valY"]))
    return ds.batch(config["batchSize"])

def createDataset(dataDict, config, typ):
    assert typ in ["train", "val"], "invalid dataset typ (train or val)"
    if typ == "train":
        return createTrainDS(dataDict, config)
    else:
        return createValDS(dataDict, config)