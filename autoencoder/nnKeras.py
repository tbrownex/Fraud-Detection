import tensorflow as tf
from tensorflow import keras

class Model:
    def __init__(self, parmDict):
        self.model = keras.Sequential([
            keras.layers.Dense(parmDict["l1Size"],
                               activation=parmDict["activation"],
                               input_shape=(parmDict["featureCount"],)),
            keras.layers.Dropout(parmDict["dropout"]),
            keras.layers.Dense(parmDict["l2Size"],
                               activation=parmDict["activation"]),
            keras.layers.Dropout(parmDict["dropout"]),
            keras.layers.Dense(parmDict["l3Size"],
                               activation=parmDict["activation"]),
            keras.layers.Dropout(parmDict["dropout"]),
            keras.layers.Dense(parmDict["l4Size"],
                               activation=parmDict["activation"]),
            keras.layers.Dropout(parmDict["dropout"]),
            keras.layers.Dense(parmDict["l5Size"],
                               activation=parmDict["activation"]),
            keras.layers.Dropout(parmDict["dropout"]),
            keras.layers.Dense(parmDict["featureCount"],
                               activation='linear'),
        ])
        
        if parmDict["optimizer"] == "adam":
            opt = tf.keras.optimizers.Adam(learning_rate=parmDict["learningRate"])
        else:
            raise ValueError('Invalid optimizer')
                
        self.model.compile(
            optimizer=opt,
            loss='mse')