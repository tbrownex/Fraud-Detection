''' A dictionary object that holds key parameters '''

__author__ = "Tom Browne"

def getConfig():
    d = {}
    d["dataLoc"]     = "/home/tbrownex/data/CreditCardFraud/"
    d["fileName"]    = "full.csv"
    d["labelColumn"] = "Class"
    d["labelType"] = "Categorical"
    d["numFeatures"]   = 29
    d["oneHot"] = False
    d["normalize"] = True
    d["numFolds"]   = 5
    d["numClasses"]  = 2
    d["evaluationMethod"] = "--"
    d["logLoc"]     = "/home/tbrownex/"
    d["logFile"]    = "fraud.log"
    d["logDefault"] = "info"
    d["valPct"]     = 0.2
    d["testPct"]    = 0.2     # There is a separate file with Test data
    d["weightsFileName"] = 'initializedWeights'
    d["weightsFileName"] = None
    d["TBdir"] = '/home/tbrownex/TF/Tensorboard'         # where to store Tensorboard data
    d["modelDir"] = "/home/tbrownex/TF/models/"  # where to save models
    return d