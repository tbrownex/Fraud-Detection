''' A dictionary object holds key parameters such as:
    - the location and name of the input data file
    - the location and name of the log file
    - the default logging level
    - an indicator allowing execution in Test mode'''

__author__ = "Tom Browne"

def getConfig():
    d = {}
    d["dataLoc"]     = "/home/tbrownex/data/CreditCardFraud/"
    d["fileName"]    = "holdout.csv"
    d["labelColumn"] = "Class"
    d["labelType"] = "Categorical"
    d["oneHot"] = True
    d["normalize"] = True
    d["numFolds"]   = 5
    d["batchSize"] = 32
    d["numClasses"]  = 2
    d["evaluationMethod"] = "--"
    d["logLoc"]     = "/home/tbrownex/"
    d["logFile"]    = "fraud.log"
    d["logDefault"] = "info"
    d["valPct"]     = 0.2
    d["testPct"]    = 0.2     # There is a separate file with Test data
    d["TBdir"] = '/home/tbrownex/TF/Tensorboard'         # where to store Tensorboard data
    d["modelDir"] = "/home/tbrownex/TF/models/"  # where to save models
    return d