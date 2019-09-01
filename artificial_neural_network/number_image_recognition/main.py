import numpy as np
from classifier import classifier as classifier
import matplotlib.pyplot as plt
import tensorflow as tf

def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    max_examples = 100
    x_train = x_train[:max_examples]
    y_train = y_train[:max_examples]

    x = []
    for i in range(len(x_train)):
        x.append(x_train[i].flatten())
    x = np.asarray(x)

    t = []
    for i in range(len(x)):
        t.append([x[i]])
    t = np.asarray(t)
   
    x_train = t

    model = classifier()

    for i in range(len(x_train)):

        minimize_operation = tf.train.AdamOptimizer(0.01).minimize(model.loss)

        session = tf.Session()

        session.run(tf.global_variables_initializer())

        for epoch in range(10):
            session.run(minimize_operation, {model.x: x_train[i], model.y: y_train[i]})

        W, b, loss = session.run([model.W, model.b, model.loss], {model.x: x_train[i], model.y: y_train[i]})

        session.close()

    print("W = %s \n b = %s \n loss = %s" % (W, b, loss))

    f, axes = plt.subplots(2, 5, figsize=(10,4))
    axes = axes.reshape(-1)
    for i in range(len(axes)):
        a = axes[i]
        a.imshow(W.T[i].reshape(28, 28), cmap = 'seismic')
        a.set_title(i)
        a.set_xticks(())
        a.set_yticks(())
    plt.show()

if __name__ == "__main__":
    main()
