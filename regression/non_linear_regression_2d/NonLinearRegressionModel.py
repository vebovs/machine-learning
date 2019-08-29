import tensorflow as tf
import math

class NonLinearRegressionModel:
    def __init__(self):
        
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)

        self.W = tf.Variable([[0.0]])
        self.b = tf.Variable([[0.0]])

        f = 20 * tf.sigmoid(tf.matmul(self.x, self.W) + self.b) + 31

        self.loss = tf.reduce_mean(tf.square(f - self.y))

