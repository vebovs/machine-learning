import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

class visualize:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    def f(self, x):
        #return 20 * np.multiply(self.sigma(x), (np.multiply(x, self.W) + self.b)) + 31
        return 3 * np.log(self.W * x) + self.b + 52

    def sigma(self, x):
        arr = np.arange(-5, 5, 0.01)
        matrix = []
        for x in arr:
            matrix.append([float(x)])
        mat = np.mat(matrix)
        p = np.divide(1, 1 + np.exp(-mat))
        #ax.plot(mat, p)
        return np.divide(1, 1 + np.exp(-x))
    
    def loss(self, x, y):
        return np.mean(np.square(self.f(x) - y))
    
    def scatterplot(self, x_train, y_train, xlab, ylab):
        ax.plot(x_train, y_train, 'o', label = '$(\\hat x^{(i)},\\hat y^{(i)})$')
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)


    def plot(self, x, x_train, y_train, xlab, ylab):
        self.scatterplot(x_train, y_train, xlab, ylab)

        ax.plot(x, self.f(x), label = '$y = f(x) = 20\u03C3(xW + b) + 31$')
        
        print('loss: ', self.loss(x, y_train))

        ax.legend()
        plt.title('non_linear_regression_2d')
        plt.show()

