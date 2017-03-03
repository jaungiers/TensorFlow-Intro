from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax for output probablity

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print(">>> Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print(">>> Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(">>> Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

    # For fun show a few visual test cases
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    test1_index = 0
    test1_x = mnist.test.images[test1_index].reshape(1, 784)
    test1_img = mnist.test.images[test1_index].reshape((28,28))
    test1_y = mnist.test.labels[test1_index].reshape(1, 10)
    test1_pred = sess.run(pred, feed_dict={x: test1_x, y: test1_y})
    
    ax1.imshow(test1_img, cmap='gray')
    ax2.bar(list(range(0,10)), test1_pred[0])

    test2_index = 1
    test2_x = mnist.test.images[test2_index].reshape(1, 784)
    test2_img = mnist.test.images[test2_index].reshape((28,28))
    test2_y = mnist.test.labels[test2_index].reshape(1, 10)
    test2_pred = sess.run(pred, feed_dict={x: test2_x, y: test2_y})
    
    ax3.imshow(test2_img, cmap='gray')
    ax4.bar(list(range(0,10)), test2_pred[0])

    plt.show()