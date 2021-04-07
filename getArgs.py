''' Get the command line arguments:
    Mandatory
    - none
    
    Optional arguments
    - save: whether to save the model with best score'''

import argparse

def editLoops(value):
    ''' Make sure the value is positive and within a rational range '''
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    if ivalue > 500:
        raise argparse.ArgumentTypeError("%s is too big" % value)
    return ivalue

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save',
                        required=False,
                        action='store_true',
                        help="Saves the best model")
    parser.add_argument("testInd", \
                        choices=['test','full'], \
                        help="Run against the full data or a test portion")
    parser.add_argument("numLoops",\
                        type=editLoops,\
                        help="Specify how many hyper-parameter combinations to run")
    '''parser.add_argument('--outliers',
                        required=False,
                        action='store_true',
                        help="Whether to remove outliers or not")'''
    return parser.parse_args()