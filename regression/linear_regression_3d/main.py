import csv
import numpy as np
from LinearRegressionModel import LinearRegressionModel as lgm
import tensorflow as tf
from visualize import visualize as vis

def main():
    x_arr = []
    y_arr = []
    z_arr = []
    with open('/home/vebovs/Desktop/machine-learning/regression/linear_regression_3d/day_length_weight.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter = ',')
        for row in readCSV:
            x = [float(row[2])]
            y = [float(row[1])]
            z = [float(row[0])]
            
            x_arr.append(x)
            y_arr.append(y)
            z_arr.append(z)
    
    x_train = np.mat(x_arr) # weight
    y_train = np.mat(y_arr) # length
    z_train = np.mat(z_arr) # day

    model = lgm()

    learning_rate = 0.0000001

    minimize_operation = tf.train.GradientDescentOptimizer(learning_rate).minimize(model.loss)

    session = tf.Session()

    session.run(tf.global_variables_initializer())

    for epoch in range(10000):
        session.run(minimize_operation, {model.x: x_train, model.y: y_train, model.z: z_train})

    W, M, b, loss = session.run([model.W, model.M, model.b, model.loss], {model.x: x_train, model.y: y_train, model.z: z_train})
    print("W = %s, M = %s, b = %s, loss = %s" % (W, M, b, loss))

    session.close()

    graph = vis(W, M ,b)
    graph.plot(x_arr, y_arr, z_arr, x_train, y_train, z_train, 'weight', 'length', 'day')

if __name__ == "__main__":
    main()