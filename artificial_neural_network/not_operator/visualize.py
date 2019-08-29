import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

class visualize:

    def __init__(self, W, b):
        self.W = W
        self.b = b

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def f(self, x):
        return self.sigmoid(x * self.W + self.b)

    def loss(self, x, y):
        return -np.mean(np.multiply(y, np.log(self.f(x))) + np.multiply((1 - y), np.log(1 - self.f(x))))

    def unpack(self, x_arr, y_arr):
        x = []
        y = []
        for row in x_arr:
            for col in row:
                x.append(col)
        for row in y_arr:
            for col in row:
                y.append(col)

        return x, y

    def scatterplot(self, x, y):
        points = self.unpack(x, y)
        ax.scatter(points[0], points[1])
    
    def plot(self, x, y, xlab = 'Input', ylab = 'Output', title = 'Logical NOT Operator'):
        self.scatterplot(x, y)
        ax.plot(x, self.f(x))

        print('loss: ', self.loss(x, y))

        ax.legend()
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        plt.title(title)
        plt.show()

