import numpy as np
from XOROperatorModel import XOROperatorModel as XOR
import tensorflow as tf
from visualize import visualize as vis

def main():
    x_train = np.mat([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.mat([[0], [1], [1], [0]])
    
    model = XOR()

    minimize_operation = tf.train.AdamOptimizer(0.01).minimize(model.loss)

    session = tf.Session()

    session.run(tf.global_variables_initializer())

    for epoch in range(10000):
        session.run(minimize_operation, {model.x: x_train, model.y: y_train})

    W1, W2, b1, b2, loss = session.run([model.W1, model.W2, model.b1, model.b2, model.loss], {model.x: x_train, model.y: y_train})
    print("W1 = %s, W2 = %s, b1 = %s, b2 = %s, \n loss = \n %s" % (W1, W2, b1, b2, loss))

    session.close()
    
    graph = vis(W1, W2, b1, b2)
    graph.plot(x_train, y_train)

if __name__ == "__main__":
    main()