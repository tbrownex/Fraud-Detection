import tensorflow as tf

class Model:
    def __init__(self, parms, features, labels):
        self._cost = None
        self._train = None
        self.labels = labels
        shape=features.get_shape().as_list()
        numFeatures = shape[1]
        shape=labels.get_shape().as_list()
        numClasses=shape[1]
        
        init = tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32)
        self.l1w = tf.compat.v1.get_variable(name="L1", shape=[numFeatures, parms["l1Size"]], initializer=init)
        self.l2w = tf.compat.v1.get_variable(name="L2", shape=[parms["l1Size"], 256], initializer=init)
        self.l3w = tf.compat.v1.get_variable(name="L3", shape=[256, numClasses], initializer=init)

        self.l1b = tf.Variable(tf.zeros(parms["l1Size"], dtype=tf.float32))
        self.l2b = tf.Variable(tf.zeros(256, dtype=tf.float32))
        self.l3b = tf.Variable(tf.zeros(numClasses))
        
        actf = parms["activation"]
        self.lr = parms["learningRate"]
        
        l1Out = actf(tf.matmul(features,self.l1w)+self.l1b)
        drop1 = tf.nn.dropout(l1Out, rate=parms["dropout"])
        l2Out = actf(tf.matmul(drop1,self.l2w)+self.l2b)
        drop2 = tf.nn.dropout(l2Out, rate=parms["dropout"])
        self._output = tf.math.sigmoid(tf.matmul(drop2,self.l3w)+self.l3b)
        
    @property
    def cost(self):
        if self._cost is None:
            self._entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self._output)
            self._cost = tf.math.reduce_mean(self._entropy)
        return self._cost

    @property
    def train(self):
        if self._train is None:
            optimizer = tf.train.AdamOptimizer(self.lr)
            self._train = optimizer.minimize(self.cost)
        return self._train