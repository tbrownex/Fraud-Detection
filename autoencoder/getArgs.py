''' Get the command line arguments:
    Mandatory
    - Whether to use the full dataset or a sample
    - Whether to use a keras-defined network or native code
    
    Optional arguments
    - save: whether to save the model with best score'''

import argparse

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save',
                        required=False,
                        action='store_true',
                        help="Saves the best model")
    parser.add_argument("testInd", \
                        choices=['test','full'], \
                        help="Run against the full data or a test portion")
    parser.add_argument("networkType", \
                        choices=['keras','native'], \
                        help="Use keras or native")
    '''parser.add_argument('--outliers',
                        required=False,
                        action='store_true',
                        help="Whether to remove outliers or not")'''
    return parser.parse_args()