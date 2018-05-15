from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Initiate placeholders for training data
  # Use 'None' to allow for any number of training examples to be fed in
  x = tf.placeholder(tf.float32, [None, 784], name="x")
  y = tf.placeholder(tf.float32, [None, 10], name="label")

  # Reshape 784 image vector to be correct input shape for tf.nn.conv2d op
  x_2d_img = tf.reshape(x, [-1, 28, 28, 1])
  # Returns 14x14 b/c of max_pool
  conv1 = conv_layer(x_2d_img, 10, 1, 32)
  # Returns 7x7 b/c of max_pool
  conv2 = conv_layer(conv1, 5, 32, 16)

  # Flatten to create correct input shape for fully connected layer
  flattened = tf.reshape(conv2, [-1, 7 * 7 * 16])

  # Create 2 fully connected layers
  fc1 = fc_layer(flattened, 7*7*16, 1024)
  fc1_relu = tf.nn.relu(fc1, "fc1_relu")
  # Keep raw output of fc (logits) to use as input for softmax cross entropy op
  fc2 = fc_layer(fc1_relu, 1024, 10)

  tf.summary.histogram("fc1_relu", fc1_relu)

  #y = tf.matmul(x, W) + b

  # Calculate cross entropy using output of 2nd fully connected layer
  with tf.name_scope("xEntropy"):
    cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=fc2))
    tf.summary.scalar('Cross Entropy', cross_entropy)


  # Use AdagradOptimizer to minimize cross entropy
  with tf.name_scope("Train"):
    train_step = tf.train.AdagradOptimizer(.01).minimize(cross_entropy)


  # Test accuracy of trained model
  with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(fc2, 1))
    # Casts list of booleans to floats and finds the mean
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('accuracy', accuracy)


  # Create session, initialize variables, run summary functions, and define TensorBoard writer
  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  summ = tf.summary.merge_all()
  writer = tf.summary.FileWriter('tb/mnistDL2')
  writer.add_graph(sess.graph)


  # Train model (200 epochs max)
  for i in range(200):
    # Create batch of 100 randomly selected data points
    batch_xs, batch_ys = mnist.train.next_batch(100)
    if i % 5 == 0:
      [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: batch_xs, y: batch_ys})
      writer.add_summary(s, i)
    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})


  # Runs a session using test data
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y: mnist.test.labels}))

  print("Time (seconds): " + str(time.time() - start_time))


# Create 2D Convolutional Neural Network with bias added and ReLu activation
def conv_layer(input, filter_size, size_in, size_out, strides=1, name="2D_conv"):
  # Names op for graph in TensorBoard
  with tf.name_scope(name):
    # Initiate variables for learned weights and biases
    w = tf.Variable(tf.truncated_normal([filter_size, filter_size, size_in, size_out],
                                        stddev=0.1), name='w_conv2d')
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name='b_conv2d')
    # Create CNN, add bias to each output channel, and apply rectified linear unit activation function
    conv = tf.nn.conv2d(input, w, strides=[1, strides, strides, 1], padding="SAME")
    conv = tf.nn.bias_add(conv, b)
    act = tf.nn.relu(conv)

    # Generate histograms of learned values
    tf.summary.histogram("Weights", w)
    tf.summary.histogram("Biases", b)
    tf.summary.histogram("Activation", act)

    # Changes shape of x (x/2 when k=2), batch size and channels don't change
    return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# Creates fully connected layer
def fc_layer(input, size_in, size_out, name="fc"):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name='w_fc')
    # Bias vector is added to fully connected: same size as output
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name='b_fc')
    act = tf.matmul(input, w) + b

    # Generate histograms of learned values
    tf.summary.histogram("Weights", w)
    tf.summary.histogram("Biases", b)
    tf.summary.histogram("Activations", act)
    return act


if __name__ == '__main__':
  start_time = time.time()
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)