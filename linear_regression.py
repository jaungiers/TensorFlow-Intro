from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
rng = np.random

# Different parameters for learning
learning_rate = 0.01
training_epochs = 1000
display_step = 100

#Population, Percent with <$5000 income, Percent Unemployed, Murders per annum per 1mil population
data = np.genfromtxt('murder_rates_data.csv', delimiter=',', skip_header=1)

# Training Data
train_test_split = int(len(data)*0.7) #70% training : 30% testing
train_X = data[:, 2][:train_test_split] #Percent unemployed
train_Y = data[:, 3][:train_test_split] #Murders per 1 million population per year
n_samples = train_X.shape[0]

# Create placeholder for providing inputs
X = tf.placeholder("float")
Y = tf.placeholder("float")

# create weights and bias and initialize with random number
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

with tf.name_scope('WX_b') as scope:
    # Construct a linear model using Y=WX+b
    pred = tf.add(tf.multiply(X, W), b)

w_h = tf.summary.histogram('weights', W)
b_h = tf.summary.histogram('biases', b)

with tf.name_scope('cost_function') as scope:
    # Calculate Mean squared error
    cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
    tf.summary.scalar('cost_function', cost)

with tf.name_scope('train') as scope:
    # Gradient descent to minimize mean sequare error
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

marge_summary_op = tf.summary.merge_all()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    summary_writer = tf.summary.FileWriter('/logs', graph_def=sess.graph_def)

    print(">>> Training started")

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            #create small batch of trining and testing data and feed it to model
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Display training information after each N step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            
            print(">>> Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), "W=", sess.run(W), "b=", sess.run(b))
            summary_str = sess.run(marge_summary_op, feed_dict={X: train_X, Y:train_Y})
            summary_writer.add_summary(summary_str, epoch)

    print(">>> Training completed")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print(">>> Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # Testing 
    print(">>> Testing started")
    test_X = data[:, 2][train_test_split:] #Percent unemployed
    test_Y = data[:, 3][train_test_split:] #Murders per 1 million population per year

    #Calculate Mean square error
    print(">>> Calculate Mean square error")
    testing_cost = sess.run(
    	tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y}
    ) #same function as cost above

    print(">>> Testing cost=", testing_cost)
    print(">>> Absolute mean square loss difference:", abs(training_cost - testing_cost))

    fig = plt.figure(1)
    plt_train = fig.add_subplot(2,1,1)
    plt_test = fig.add_subplot(2,1,2, sharex=plt_train, sharey=plt_train)

    plt_train.plot(train_X, train_Y, 'ro', label='Original data')
    plt_train.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt_train.legend()

    plt_test.plot(test_X, test_Y, 'bo', label='Testing data')
    plt_test.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt_test.legend()
    
    plt.show()