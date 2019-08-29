import tensorflow as tf

class NotOperatorModel:

    def __init__(self):
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)

        self.W = tf.Variable([[0.0]])
        self.b = tf.Variable([[0.0]])

        logits = tf.multiply(self.x, self.W) + self.b

        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = self.y, logits = logits)