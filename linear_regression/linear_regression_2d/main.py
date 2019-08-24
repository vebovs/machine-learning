import csv
from LinearRegressionModel import LinearRegressionModel as lgm
import numpy as np
import tensorflow as tf
from visualize import visualize as vis

def main():
    x_arr = []
    y_arr = []
    with open('/home/vebovs/Desktop/machine-learning/linear_regression/linear_regression_2d/length_weight.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter = ',')
        for row in readCSV:
            x = [float(row[0])]
            y = [float(row[1])]
            
            x_arr.append(x)
            y_arr.append(y)
    
    x_train = np.mat(x_arr)
    y_train = np.mat(y_arr)
  
    model = lgm()

    learning_rate = 0.0001

    minimize_operation = tf.train.GradientDescentOptimizer(learning_rate).minimize(model.loss)

    session = tf.Session()

    session.run(tf.global_variables_initializer())

    for epoch in range(1000):
        session.run(minimize_operation, {model.x: x_train, model.y: y_train})

    W, b, loss = session.run([model.W, model.b, model.loss], {model.x: x_train, model.y: y_train})
    print("W = %s, b = %s, loss = %s" % (W, b, loss))

    session.close()

    graph = vis(W, b)
    x_plot = np.mat([[np.min(x_train)], [np.max(x_train)]])
    graph.plot(x_plot, x_train, y_train, 'length', 'weight')


if __name__ == "__main__":
    main()