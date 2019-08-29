import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

class visualize:
    def __init__(self, W, M, b):
        self.W = W
        self.M = M
        self.b = b

    def f(self, x, y):
        return x * self.W + y * self.M + self.b
    
    def loss(self, x, y, z):
        return np.mean(np.square(self.f(x, y) - z))

    def unpack(self, x_arr, y_arr, z_arr):
        x = []
        y = []
        z = []
        for row in x_arr:
            for col in row:
                x.append(col)
        for row in y_arr:
            for col in row:
                y.append(col)
        for row in z_arr:
            for col in row:
                z.append(col)

        return x, y, z
    
    def scatterplot(self, x_arr, y_arr, z_arr, xlab, ylab, zlab):
        points = self.unpack(x_arr, y_arr, z_arr)
        ax.scatter(points[0] ,points[1], points[2])
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_zlabel(zlab)

    def plot(self, x_arr, y_arr, z_arr, x_train, y_train, z_train, xlab, ylab, zlab, title = 'linear_regression_3d'):
        self.scatterplot(x_arr, y_arr, z_arr, xlab, ylab, zlab)
        points = self.unpack(x_arr, y_arr, z_arr)
        points[0].sort()
        points[1].sort()
        points[2].sort()
        ax.plot([min(points[0]), max(points[0])], [min(points[1]), max(points[1])], [min(points[2]), max(points[2])])
        
        print('loss: ', self.loss(x_train, y_train, z_train))

        ax.legend()
        plt.title(title)
        plt.show()

