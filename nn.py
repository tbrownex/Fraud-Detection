import tensorflow as tf
from tensorflow import keras

class Model:
    def __init__(self, numFeatures):
        self.model = keras.Sequential([
            keras.layers.Dense(256, activation='relu',
                               input_shape=(numFeatures,)),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(1, activation='sigmoid'),
        ])
        
        metrics = [
            keras.metrics.Accuracy(name='accuracy'),
            keras.metrics.TruePositives(name='tp'),
            keras.metrics.FalsePositives(name='fp'),
            keras.metrics.TrueNegatives(name='tn'),
            keras.metrics.FalseNegatives(name='fn'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
        
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=metrics)