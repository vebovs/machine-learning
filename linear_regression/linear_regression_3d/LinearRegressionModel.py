import tensorflow as tf

class LinearRegressionModel:
    def __init__(self):
        
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)
        self.z = tf.placeholder(tf.float32)

        self.W = tf.Variable([[0.0]])
        self.M = tf.Variable([[0.0]])
        self.b = tf.Variable([[0.0]])

        f = tf.matmul(self.x, self.W) + tf.matmul(self.y, self.M) + self.b

        self.loss = tf.reduce_mean(tf.square(f - self.z))

