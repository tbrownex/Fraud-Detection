import tensorflow as tf
import numpy as np

class Model:
    def __init__(self, parmDict):
        self._cost = None
        self._train = None
        
        # For Batch Normalization
        #training = True
        self.X = tf.placeholder("float", shape=[None, parmDict["featureCount"]], name="input")
        self.y_ = tf.placeholder("float", shape=[None, parmDict["featureCount"]], name="output")
        
        init = tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32)
        self.l1w = tf.Variable(init((parmDict["featureCount"], parmDict["l1Size"])), name="L1w")
        self.l2w = tf.Variable(init((parmDict["l1Size"], parmDict["l2Size"])), name="L2w")
        self.l3w = tf.Variable(init((parmDict["l2Size"], parmDict["l3Size"])), name="L3w")
        self.l4w = tf.Variable(init((parmDict["l3Size"], parmDict["l4Size"])), name="L4w")
        self.l5w = tf.Variable(init((parmDict["l4Size"], parmDict["l5Size"])), name="L5w")
        self.l6w = tf.Variable(init((parmDict["l5Size"], parmDict["featureCount"])), name="L6w")
        
        self.l1b = tf.Variable(tf.zeros(parmDict["l1Size"], dtype=tf.float32), name="L1b")
        self.l2b = tf.Variable(tf.zeros(parmDict["l2Size"], dtype=tf.float32), name="L2b")
        self.l3b = tf.Variable(tf.zeros(parmDict["l3Size"], dtype=tf.float32), name="L3b")
        self.l4b = tf.Variable(tf.zeros(parmDict["l4Size"], dtype=tf.float32), name="L4b")
        self.l5b = tf.Variable(tf.zeros(parmDict["l5Size"], dtype=tf.float32), name="L5b")
        self.l6b = tf.Variable(tf.zeros(parmDict["featureCount"], dtype=tf.float32), name="L6b")
        
        if parmDict["activation"] == "relu":
            actf = tf.nn.relu
        elif parmDict["activation"] == "tanh":
            actf = tf.nn.tanh
        else:
            raise ValueError('Invalid activiation function')
            
        self.lr = parmDict["learningRate"]
        
        l1_out = actf(tf.matmul(self.X, self.l1w)+self.l1b, name="L1act")
        self.drop1 = tf.nn.dropout(l1_out, rate=parmDict["dropout"], name="L1drop")
        l2_out = actf(tf.matmul(self.drop1, self.l2w)+self.l2b, name="L2act")
        self.drop2 = tf.nn.dropout(l2_out, rate=parmDict["dropout"])
        l3_out = actf(tf.matmul(self.drop2, self.l3w)+self.l3b, name="L3act")
        self.drop3 = tf.nn.dropout(l3_out, rate=parmDict["dropout"])
        l4_out = actf(tf.matmul(self.drop3, self.l4w)+self.l4b, name="L4act")
        self.drop4 = tf.nn.dropout(l4_out, rate=parmDict["dropout"])
        l5_out = actf(tf.matmul(self.drop4, self.l5w)+self.l5b, name="L5act")
        self.drop5 = tf.nn.dropout(l5_out, rate=parmDict["dropout"])
        self.output = tf.math.add(tf.matmul(self.drop5, self.l6w), self.l6b, name="prediction")
        #bn1 = batch_norm_wrapper(l1, training)
        #l1_out = actf(bn1)
        self.opt = parmDict["optimizer"]
        
    @property
    def cost(self):
        if self._cost is None:
            self._cost = tf.losses.mean_squared_error(self.y_, self.output)
        return self._cost

    @property
    def train(self):
        if self._train is None:
            if self.opt == "adam":
                opt = tf.train.AdamOptimizer(self.lr)
            else:
                raise ValueError('Invalid optimizer')
            self._train = opt.minimize(self.cost, name="min")
        return self._train