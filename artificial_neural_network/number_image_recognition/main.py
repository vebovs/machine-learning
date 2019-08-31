import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from NumberImageRecognitionModel import NumberImageRecognitionModel as NIRM

def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    model = NIRM()

    minimize_operation = tf.train.AdamOptimizer(0.01).minimize(model.loss)

    session = tf.Session()

    session.run(tf.global_variables_initializer())

    for epoch in range(10):
        session.run(minimize_operation, {model.x: x_train, model.y: y_train})

    W, b, loss = session.run([model.W, model.b, model.loss], {model.x: x_train, model.y: y_train})
    print("W = %s, b = %s, loss = \n %s" % (W, b, loss))

    session.close()

if __name__ == "__main__":
    main()
