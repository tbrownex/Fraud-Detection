def removeClass(dataDict, val):
    ''' The idea here is that the calling module needs only 1 class. For instance the GNN is trained on Negatives only so remove all the positives '''
    idx = dataDict["trainY"][:,val] ==1        # "val" is the column (which correspondes to a class)
    dataDict["trainX"] = dataDict["trainX"][~idx]
    return dataDict