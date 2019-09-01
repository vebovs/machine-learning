import tensorflow as tf

class classifier:

    def __init__(self):
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)

        self.W = tf.Variable(tf.random_uniform([784, 10], 0))
        self.b = tf.Variable(tf.random_uniform([10], 0))

        logits = tf.matmul(self.x, self.W) + self.b

        f = tf.nn.softmax(logits = logits)

        self.loss = tf.reduce_mean(( (self.y * tf.log(f)) + ((1 - self.y) * tf.log(1.0 - f)) ) * -1)

        #self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(f, 1), tf.argmax(self.y, 1)), tf.float32))