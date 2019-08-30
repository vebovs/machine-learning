import tensorflow as tf

class XOROperatorModel:
    
    def __init__(self):
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)

        self.W1 = tf.Variable(tf.random_uniform([2,2], -1, 1))
        self.W2 = tf.Variable(tf.random_uniform([2,1], -1, 1))

        self.b1 = tf.Variable([[0.0, 0.0]])
        self.b2 = tf.Variable([[0.0]])

        h = tf.sigmoid(tf.matmul(self.x, self.W1) + self.b1)
        f = tf.sigmoid(tf.matmul(h, self.W2) + self.b2)

        self.loss = tf.reduce_mean(( (self.y * tf.log(f)) + ((1 - self.y) * tf.log(1.0 - f)) ) * -1)