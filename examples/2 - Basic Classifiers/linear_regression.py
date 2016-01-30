'''
A linear regression learning algorithm example using TensorFlow library.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random

# Parameters
learning_rate = 0.01
training_epochs = 5000
display_step = 50

# Training Data
train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]

# Testing example, as requested (Issue #2)
test_X = numpy.asarray([6.83,4.668,8.9,7.91,5.7,8.7,3.1,2.1])
test_Y = numpy.asarray([1.84,2.273,3.2,2.831,2.92,3.24,1.35,1.03])

# tf Graph Input
x = tf.placeholder("float", name='x')
y_ = tf.placeholder("float", name='y_')

# Create Model

# Set model weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model
with tf.name_scope('Wx_b') as scope:
    y = tf.add(tf.mul(x, W), b)

# Add summary ops to collect data
_ = tf.histogram_summary('weights', W)
_ = tf.histogram_summary('biases', b)
_ = tf.histogram_summary('y', y)

#_ = tf.scalar_summary('biases', b)
#_ = tf.scalar_summary('weights', W)

# Minimize the squared errors
with tf.name_scope('cost') as scope:
    cost = tf.reduce_sum(tf.pow(y-y_, 2))/(2*n_samples) #L2 loss
    _ = tf.scalar_summary('cost', cost)

with tf.name_scope('train') as scope:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) #Gradient descent

with tf.name_scope('test') as scope:
    accuracy = tf.reduce_sum(tf.pow(y-y_, 2))/(2*n_samples) #L2 loss
    _ = tf.scalar_summary('accuracy', accuracy)

# Initializing the variables
merged = tf.merge_all_summaries()
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    writer = tf.train.SummaryWriter('/tmp/tf_logs', sess.graph_def)
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (xval, yval) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={x: xval, y_: yval})

        #Display logs per epoch step
        if epoch % display_step == 0:
            result = sess.run([merged, accuracy], feed_dict={x: test_X, y_:test_Y})
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(result[1]), \
                "W=", sess.run(W), "b=", sess.run(b)
            writer.add_summary(result[0], epoch)

    print "Optimization Finished!"
    training_cost = sess.run(cost, feed_dict={x: train_X, y_: train_Y})
    print "Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n'

    print "Testing... (L2 loss Comparison)"
    testing_cost = sess.run(accuracy,
                            feed_dict={x: test_X, y_: test_Y}) #same function as cost above
    print "Testing cost=", testing_cost
    print "Absolute l2 loss difference:", abs(training_cost - testing_cost)

    #Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
