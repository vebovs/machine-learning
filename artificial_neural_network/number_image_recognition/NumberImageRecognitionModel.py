import tensorflow as tf

class NumberImageRecognitionModel:

    def __init__(self):
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)

        self.W = tf.Variable([tf.random_uniform([28,1], 0, 1)])
        self.b = tf.Variable([[0.0]])

        print(self.W)

        logits = tf.matmul(self.x, self.W) + self.b

        f = tf.nn.softmax(logits)

        self.loss = tf.losses.softmax_cross_entropy(self.y, logits)

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(f, 1), tf.argmax(self.y, 1)), tf.float32))
