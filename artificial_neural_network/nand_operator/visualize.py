import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

class visualize:

    def __init__(self, W, b):
        self.W = np.asmatrix(W)
        self.b = b

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def f(self, x):
        return self.sigmoid(x * self.W + self.b)

    def loss(self, x, y):
        return -np.mean(np.multiply(y, np.log(self.f(x))) + np.multiply((1 - y), np.log(1 - self.f(x))))
    
    def plot(self, x, y, xlab = 'Input', ylab = 'Input', zlab = 'Output', title = 'Logical NAND Operator'):
        print('loss: ', self.loss(x, y))

        x1_grid, x2_grid = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
        y_grid = np.empty([10, 10])

        for i in range(0, x1_grid.shape[0]):
            for j in range(0, x1_grid.shape[1]):
                y_grid[i, j] = self.f([[x1_grid[i, j], x2_grid[i, j]]])

        
        ax.plot_wireframe(x1_grid, x2_grid, y_grid, color="green")
        ax.legend()
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_zlabel(zlab)
        plt.title(title)
        plt.show()