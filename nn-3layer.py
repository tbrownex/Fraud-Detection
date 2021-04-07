import tensorflow as tf
from tensorflow import keras
import numpy as np

class Model:
    def __init__(self, parmDict, config, ratio):
        initial_bias = np.log(1/ratio)
        self.model = keras.Sequential([
            keras.layers.Dense(parmDict["l1Size"],
                                        activation=parmDict["activation"],
                                        input_shape=(config["numFeatures"],)),
            keras.layers.Dense(256, activation=parmDict["activation"]),
            keras.layers.Dropout(parmDict["dropout"]),
            keras.layers.Dense(128, activation=parmDict["activation"]),
            keras.layers.Dropout(parmDict["dropout"]),
            keras.layers.Dense(1,
                               activation='sigmoid')
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
            metrics=metrics)