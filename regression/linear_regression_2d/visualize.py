import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

class visualize:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    def f(self, x):
        return x * self.W + self.b
    
    def loss(self, x, y):
        return np.mean(np.square(self.f(x) - y))
    
    def scatterplot(self, x_train, y_train, xlab, ylab):
        ax.plot(x_train, y_train, 'o', label = '$(\\hat x^{(i)},\\hat y^{(i)})$')
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)


    def plot(self, x_train, y_train, xlab, ylab, title = 'linear_regression_2d'):
        self.scatterplot(x_train, y_train, xlab, ylab)
        x_plot = np.mat([[np.min(x_train)], [np.max(x_train)]])
        ax.plot(x_plot, self.f(x_plot), label = '$y = f(x) = xW + b$')
        
        print('loss: ', self.loss(x_train, y_train))

        ax.legend()
        plt.title(title)
        plt.show()

