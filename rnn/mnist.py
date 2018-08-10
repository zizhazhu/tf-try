import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

n_steps = 28
n_inputs = 28
n_neurons = 150
n_outputs = 10

mnist = input_data.read_data_sets('./tmp/mnist')
X_test = mnist.test.images.reshape((-1, n_steps, n_inputs))
y_test = mnist.test.labels

learning_rate = 0.001


def RNN(X, labels):
    basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
    outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)
    logits = tf.layers.dense(states, n_outputs)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits
    )
    loss = tf.reduce_mean(loss)
    return loss
