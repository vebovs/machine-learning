import numpy as np
from NANDOperatorModel import NANDOperatorModel as NAND
import tensorflow as tf
from visualize import visualize as vis

def main():
    x_train = np.mat([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.mat([[1], [1], [1], [0]])
    
    model = NAND()

    minimize_operation = tf.train.AdamOptimizer(0.1).minimize(model.loss)

    session = tf.Session()

    session.run(tf.global_variables_initializer())

    for epoch in range(10000):
        session.run(minimize_operation, {model.x: x_train, model.y: y_train})

    W, b, loss = session.run([model.W, model.b, model.loss], {model.x: x_train, model.y: y_train})
    print("W = %s, b = %s, \n loss = \n %s" % (W, b, loss))

    session.close()

    graph = vis(W, b)
    graph.plot(x_train, y_train)

if __name__ == "__main__":
    main()
