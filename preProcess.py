''' Prepare the data for modeling:
    - Identify any static columns (single value) and remove them
    - (optional) Select a smaller portion of the full data
    - Split data into Train & Test, which also shuffles
    - Split the features from the label
    - (optional) Normalize the data
    - (optional) Remove outliers
    - Set the data type to float32 for Tensorflow
    '''

import numpy as np

from normalizeData import normalize
from analyzeCols   import analyzeCols
from splitData     import splitData
from splitLabel    import splitLabel
#from convertOrdinals import convertOrdinals
#from removeOutliers import removeOutliers
#from genFeatures import genFeatures

def removeCols(df):
    ''' if any column is constant (same value for all rows) remove it '''
    cols   = df.columns
    remove = analyzeCols(df)
    keep = [col for col in cols if col not in remove]
    df = df[keep]
    return df

def preProcess(df, config, args):
    colCount = df.shape[1]
    df = removeCols(df)
    print(" - {} static columns removed".format(colCount - df.shape[1]))
    
    if args.testInd == "test":
        df = df.sample(frac=0.4)
        print(" - Using a fraction of the full data")
    else:
        print(" - Using full dataset")
    
    '''if args.genFeatures == "Y":
        print("\nGenerating features")
        df = genFeatures(train, test)
    '''
    dataDict = splitData(df, config)
    dataDict = splitLabel(dataDict, config)
    
    if config["normalize"]:
        print(" - Normalizing the data")
        dataDict = normalize(dataDict, "Std")
    else:
        print(" - Not normalizing the data")
        
    '''if args.Outliers == "Y":
        print(" - Removing outliers")
        dataDict = removeOutliers(dataDict)    
    else:
        print(" - Not removing outliers")'''
    dataDict["trainX"] = dataDict["trainX"].astype(np.float32)
    dataDict["valX"] = dataDict["valX"].astype(np.float32)
    return dataDict