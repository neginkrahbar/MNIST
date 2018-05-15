# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
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

  # Create the model

  # TF optimizes by creating a graph of interacting operations whose computations are done all at once
  # Placeholders are symbolic variables that are manipulated and describe interacting operations
  x = tf.placeholder(tf.float32, [None, 784], name="x")
  # TF Variable - modifiable tensor that lives in graph of interacting operations, can be used and modified by computations
  # In ML, let model parameters be Variables
  # Here we are initializing Variables with 0s, since the values will be learned
  W = tf.Variable(tf.zeros([784, 10]), name="W")
  b = tf.Variable(tf.zeros([10]), name="b")
  y = tf.matmul(x, W) + b

  # Initialize variabel to hold real labels
  y_ = tf.placeholder(tf.float32, [None, 10], name="label")

  # Define loss and optimizer

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  with tf.name_scope("xEntropy"):
    cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    tf.summary.scalar('Loss', cross_entropy)

  with tf.name_scope("train"):
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  # Test trained model

  # argmax gives index of highest entry - so for y: highest probability of predicted class and y_: actual class
  # using tf.equal gives a list of booleans
  with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # Casts list of booleans to floats and finds the mean
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('accuracy', accuracy)



  # Create session, initialize variables, run summary functions, and define TensorBoard writer
  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  summ = tf.summary.merge_all()
  writer = tf.summary.FileWriter('tb/mnistSoftmax8')
  writer.add_graph(sess.graph)
  tf.reset_default_graph()

  # Train 1000 times
  for _ in range(1000):
    # Train using a batch of 100 randomly selected data points - stochastic training and stochastic gradient descent
    # Less expensive than training using entire dataset
    batch_xs, batch_ys = mnist.train.next_batch(100)

    if _ % 5 == 0 or _ == 999:
      [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: batch_xs, y_: batch_ys})
      writer.add_summary(s, _)

    # Replace values initiated values with values found for these batches
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})



  # Runs a session using test data
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))

  print("Time (seconds): " + str(time.time() - start_time))


if __name__ == '__main__':
  start_time = time.time()
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


