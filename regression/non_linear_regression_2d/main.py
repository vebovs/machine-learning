import csv
import numpy as np
from NonLinearRegressionModel import NonLinearRegressionModel as lgm
import tensorflow as tf
from visualize import visualize as vis

def main():
    x_arr = []
    y_arr = []
    with open('/home/vebovs/Desktop/machine-learning/regression/non_linear_regression_2d/day_head_circumference.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter = ',')
        for row in readCSV:
            x = [float(row[0])]
            y = [float(row[1])]
            
            x_arr.append(x)
            y_arr.append(y)
    
    x_train = np.mat(x_arr) # day
    y_train = np.mat(y_arr) # head_circumference
    
    model = lgm()

    learning_rate = 0.00001

    minimize_operation = tf.train.AdamOptimizer(learning_rate).minimize(model.loss)

    session = tf.Session()

    session.run(tf.global_variables_initializer())

    for epoch in range(10000):
        session.run(minimize_operation, {model.x: x_train, model.y: y_train})

    W, b, loss = session.run([model.W, model.b, model.loss], {model.x: x_train, model.y: y_train})
    print("W = %s, b = %s, loss = %s" % (W, b, loss))

    session.close()

    graph = vis(W, b)
    x_plot = np.sort(x_arr, axis = 0)
    graph.plot(x_plot, x_train, y_train, 'day', 'head_circumference')

if __name__ == "__main__":
    main()