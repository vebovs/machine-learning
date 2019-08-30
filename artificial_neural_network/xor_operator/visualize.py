import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

fig = plt.figure()
ax = fig.add_subplot(224, projection = '3d')

class visualize:
    def __init__(self, W1, W2, b1, b2):
        self.W1 = np.asmatrix(W1)
        self.W2 = np.asmatrix(W2)
        self.b1 = b1
        self.b2 = b2

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def f1(self, x):
        return self.sigmoid(x * self.W1 + self.b1)

    def f2(self, h):
        return self.sigmoid(h * self.W2 + self.b2)

    def f(self, x):
        return self.f2(self.f1(x))

    def loss(self, x, y):
        return -np.mean(np.multiply(y, np.log(self.f(x))) + np.multiply((1 - y), np.log(1 - self.f(x))))

    def plot(self, x, y, xlab = 'Input', ylab = 'Input', zlab = 'Output', title = 'Logical XOR Operator'):
        print('loss: ', self.loss(x, y))

        x1_grid, x2_grid = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
        f_grid = np.empty([10, 10])
        for i in range(0, x1_grid.shape[0]):
            for j in range(0, x1_grid.shape[1]):
                f_grid[i, j] = self.f([[x1_grid[i, j], x2_grid[i, j]]])

        ax.plot_wireframe(x1_grid, x2_grid, f_grid, color="lightgreen")

        ax.legend()
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_zlabel(zlab)
        plt.title(title)
        plt.show()
