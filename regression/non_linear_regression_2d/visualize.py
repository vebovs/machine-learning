import numpy as np
import matplotlib.pyplot as plt
import math

fig, ax = plt.subplots()

class visualize:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    def f(self, x):
        return 20 * self.sigma(x * self.W + self.b) + 31

    def sigma(self, x):
        return np.divide(1 , 1 + np.exp(-x))
    
    def loss(self, x, y):
        return np.mean(np.square(self.f(x) - y))
    
    def scatterplot(self, x_train, y_train, xlab, ylab):
        ax.plot(x_train, y_train, 'o', label = '$(\\hat x^{(i)},\\hat y^{(i)})$')
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)


    def plot(self, x_arr, x_train, y_train, xlab, ylab, title = 'non_linear_regression_2d'):
        self.scatterplot(x_train, y_train, xlab, ylab)
        x_plot = np.sort(x_arr, axis = 0)
        ax.plot(x_plot, self.f(x_plot), label = '$y = f(x) = 20\u03C3(xW + b) + 31$')
        
        print('loss: ', self.loss(x_arr, y_train))

        ax.legend()
        plt.title(title)
        plt.show()

