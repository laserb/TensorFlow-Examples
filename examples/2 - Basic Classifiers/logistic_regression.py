'''
A logistic regression learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

# Import MINST data
import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder("float", [None, 784], name='x') # mnist data image of shape 28*28=784
y = tf.placeholder("float", [None, 10], name='y') # 0-9 digits recognition => 10 classes

# Create model

# Set model weights
W = tf.Variable(tf.zeros([784, 10]),name='W')
b = tf.Variable(tf.zeros([10]),name='b')

# Construct model
with tf.name_scope('Wx_b') as scope:
    activation = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# export some variables
_ = tf.histogram_summary('weights', W)
_ = tf.histogram_summary('biases', b)
_ = tf.histogram_summary('activation', activation)

# Minimize error using cross entropy
with tf.name_scope('cost') as scope:
    cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(activation), reduction_indices=1)) # Cross entropy
    _ = tf.scalar_summary('cost', cost)

avg_cost = tf.Variable( 0.,name='avg_cost')
with tf.name_scope('train') as scope:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) # Gradient Descent
    tf_total_batch = tf.placeholder("float", None, name='total_batch')
    avg_add = avg_cost.assign_add(tf.div(cost,tf_total_batch))
    avg_reset = avg_cost.assign(0.)
    _ = tf.scalar_summary('avg_cost', avg_cost)

# Initializing the variables
merged = tf.merge_all_summaries()
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    writer = tf.train.SummaryWriter('/tmp/tf_logs', sess.graph_def)
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        sess.run(avg_reset)
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            current_cost, avg = sess.run([cost, avg_add], feed_dict={x: batch_xs, y: batch_ys, tf_total_batch: total_batch})
            #avg = sess.run(avg_add, feed_dict={x: batch_xs, y: batch_ys})
        # Display logs per epoch step
        if epoch % display_step == 0:
            # Compute average loss
            result = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys})
            print "Epoch:", '%04d' % (epoch+1), "cost=", avg
            writer.add_summary(result, epoch)

    print "Optimization Finished!"

    # Test model
    with tf.name_scope('test') as scope:
        correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print "Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
