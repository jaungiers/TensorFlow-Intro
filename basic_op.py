#print() is only available in python 3+ so we explicitly include print function
from __future__ import print_function

import tensorflow as tf
import os

# Hide TensorFlows warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#Use constant function to define constanst node in tensorflow computation graph
a = tf.constant(1)
b = tf.constant(2)

# Launch the default graph using Session() funuction.
with tf.Session() as sess:
    print("a=1, b=2")
    print("Addition: %i" % sess.run(a+b))
    print("Multiplication: %i" % sess.run(a*b))


# Basic Operations with variable as graph input
# The value returned by the constructor represents the output
# of the Variable op. (define as input when running session)
# tf Graph input
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

# Define some operations
add = tf.add(a, b)
mul = tf.multiply(a, b)

# Launch the default graph.
with tf.Session() as sess:
    # Run every operation with variable input
    print("Addition with variables: %i" % sess.run(add, feed_dict={a: 4, b: 8}))
    print("Multiplication with variables: %i" % sess.run(mul, feed_dict={a: 4, b: 8}))

# Matrix multiplaction


# Create a Constant op that produces a 1x2 matrix.  The op is
# added as a node to the default graph.
#
# The value returned by the constructor represents the output
# of the Constant op.
#Create matrix with 1x2 dim
matrix1 = tf.constant([[1., 2.]])

# create matrix with 2x1 dim
matrix2 = tf.constant([[3.],[4.]])

#create node "product" for result
product = tf.matmul(matrix1, matrix2)

# to compute the resultwe need to run the graph.
# for that we create object of session and run it
with tf.Session() as sess:
    result = sess.run(product)
    print(result)
