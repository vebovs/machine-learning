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
    
    def scatterplot(self, x_arr, y_arr, z_arr):
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

        #ax.plot(x_train, y_train, 'o', label = '$(\\hat x^{(i)},\\hat y^{(i)})$')
        ax.scatter(x, y, z)
        ax.legend()

    def plot(self, x_arr, y_arr, z_arr, x_train, y_train, z_train):
        self.scatterplot(x_arr, y_arr, z_arr)

        x = []
        y = []
        z = []
        res = self.f(x_train, y_train)
        for row in x_arr:
            for col in row:
                x.append(col)
        for row in y_arr:
            for col in row:
                y.append(col)
        for row in res.tolist():
            for col in row:
                z.append(col)

        x.sort()
        y.sort()
        z.sort()
        ax.plot([min(x), max(x)], [min(y), max(y)], [min(z), max(z)])
        
        print('loss: ', self.loss(x_train, y_train, z_train))

        ax.legend()
        plt.title('linear_regression_3d')
        plt.show()

