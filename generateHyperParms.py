import random

def getClass(parm):
    return parm.__class__.__name__

def getL1(d):
    # Neuron count in Layer 1 is powers of 2
    typ = getClass(d["L1Size"])
    assert (typ in ["list", "int"]), "invalid L1 size; check 'modelParms'"
    if typ == "list":
        start = d["L1Size"][0]
        end = d["L1Size"][1]
        return 2 ** random.randint(start, end)
    else:
        return d["L1Size"]

def getActivation(d):
    typ = getClass(d["activation"])
    assert (typ in ["list", "str"]), "invalid activation; check 'modelParms'"
    if typ == "list":
        return random.sample(d["activation"], 1)[0]
    else:
        return d["activation"]
    
def getBatchSize(d):
    typ = getClass(d["batchSize"])
    assert (typ in ["list", "int"]), "invalid batch size; check 'modelParms'"
    # Batch size is powers of 2
    if typ == "list":
        return 2 ** random.sample(d["batchSize"], 1)[0]
    else:
        return d["batchSize"]

def getLR(d):
    typ = getClass(d["learningRate"])
    assert (typ in ["list", "float"]), "invalid learningRate; check 'modelParms'"
    if typ == "list":
        start = d["learningRate"][0]
        end = d["learningRate"][1]
        return round(random.uniform(start, end), 3)
    else:
        return d["learningRate"]

def getDropout(d):
    typ = getClass(d["dropout"])
    assert (typ in ["list", "float"]), "invalid dropout; check 'modelParms'"
    if typ == "list":
        start = d["dropout"][0]
        end = d["dropout"][1]
        return random.uniform(start, end)
    else:
        return d["dropout"]

def getClassWeight(d):
    # How much to penalize false negatives
    typ = getClass(d["clsWeight"])
    assert (typ in ["list", "int"]), "invalid classWeight; check 'modelParms'"
    if typ == "list":
        start = d["clsWeight"][0]
        end = d["clsWeight"][1]
        return random.randint(start, end)
    else:
        return d["clsWeight"]
    
def generateHyperParms(d):
    hyperParms = {}
    hyperParms["L1Size"] = getL1(d)
    hyperParms["activation"] = getActivation(d)
    hyperParms["batchSize"] = getBatchSize(d)
    hyperParms["learningRate"] = getLR(d)
    hyperParms["dropout"] = getDropout(d)
    hyperParms["clsWeight"] = getClassWeight(d)
    return hyperParms