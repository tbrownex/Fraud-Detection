import tensorflow as tf
from tensorflow import keras
import numpy as np

class Model:
    '''
    Create a Seq model:
     - Set the evaluation metrics
     - Set the initial output bias
     - Define the loss function
     - Set the optimizer
     - Initialize weights (optional)
    '''
    def __init__(self, parmDict, config, ratio):
        initialBias = tf.keras.initializers.Constant(np.log(1/ratio))
        ''' This is TF tutorial code '''
        self.model = keras.Sequential([
            keras.layers.Dense(
                parmDict["L1Size"],
                activation=parmDict["activation"],
                input_shape=(config["numFeatures"],)),
            keras.layers.Dropout(parmDict["dropout"]),
            keras.layers.Dense(1, activation='sigmoid',
                               bias_initializer=initialBias),
        ])
        metrics = [
            keras.metrics.TruePositives(name='TruePos'),
            keras.metrics.FalsePositives(name='FalsePos'),
            keras.metrics.FalseNegatives(name='FalseNeg'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
        self.model.compile(
            optimizer=keras.optimizers.Adam(lr=parmDict["learningRate"]),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=metrics)
        # if you have saved off initialized weights, this is the filename 
        if config["weightsFileName"]:
            #print(" - Loading initialized weights")
            self.model.load_weights(config["weightsFileName"])

        ''' This is my code
        self.model = keras.Sequential([
            keras.layers.Dense(parmDict["l1Size"],
                                        activation=parmDict["activation"],
                                        input_shape=(config["numFeatures"],)),
            #keras.layers.Dense(256, activation=parmDict["activation"]),
            keras.layers.Dropout(parmDict["dropout"]),
            keras.layers.Dense(128, activation=parmDict["activation"]),
            keras.layers.Dropout(parmDict["dropout"]),
            keras.layers.Dense(1,
                               activation='sigmoid',
                               bias_initializer=initialBias),
        ])
        
        metrics = [
            keras.metrics.TruePositives(name='TruePos'),
            keras.metrics.FalsePositives(name='FalsePos'),
            keras.metrics.FalseNegatives(name='FalseNeg'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
        self.model.compile(
            optimizer=keras.optimizers.Adam(lr=parmDict["learningRate"]),
            loss='binary_crossentropy',
            metrics=metrics) '''