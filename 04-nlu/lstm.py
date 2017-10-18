'''
This script shows how to predict stock prices using a basic RNN
'''
import tensorflow as tf
import numpy as np
import os

tf.set_random_seed(777)  # reproducibility

# train Parameters
seq_length = 7
data_dim = 5
hidden_dim = 10
output_dim = 1
learning_rate = 0.01
iterations = 150

# Open, High, Low, Volume, Close
x = [[[0, 0, 0, 0, 1], [0, 0, 1, 0, 0]],
	 [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]]]
y = [[1], [2]]


# input place holders
X = tf.placeholder(tf.float32, [None, 2, 5])
Y = tf.placeholder(tf.float32, [None, 1])

# build a LSTM network
cell = tf.contrib.rnn.BasicLSTMCell(
    num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(
    outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output

# cost/loss
loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

#RMSE
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], feed_dict={
                                X: x, Y: y})
        print("[step: {}] loss: {}".format(i, step_loss))

	# Test step
    test_predict = sess.run(Y_pred, feed_dict={X: x})
    rmse_val = sess.run(rmse, feed_dict={
                    targets: y, predictions: test_predict})
    print("RMSE: {}".format(rmse_val))