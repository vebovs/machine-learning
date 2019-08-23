import csv
from LinearRegressionModel import LinearRegressionModel as lgm
import numpy as np
import tensorflow as tf

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

    # Training: adjust the model so that its loss is minimized
    minimize_operation = tf.train.GradientDescentOptimizer(0.01).minimize(model.loss)

    # Create session object for running TensorFlow operations
    session = tf.Session()

    # Initialize tf.Variable objects
    session.run(tf.global_variables_initializer())

    for epoch in range(1000):
        session.run(minimize_operation, {model.x: x_train, model.y: y_train})

    # Evaluate training accuracy
    W, b, loss = session.run([model.W, model.b, model.loss], {model.x: x_train, model.y: y_train})
    print("W = %s, b = %s, loss = %s" % (W, b, loss))

    session.close()


if __name__ == "__main__":
    main()