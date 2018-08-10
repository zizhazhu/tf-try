import numpy as np
import tensorflow as tf
import tensorflow.contrib as tfc

n_inputs = 3
n_neurons = 5

X0_batch = np.array([[0, 1, 2]], dtype=np.float32)
X1_batch = np.array([[9, 8, 7]], dtype=np.float32)

X0 = tf.placeholder(tf.float32, [None, n_inputs])
X1 = tf.placeholder(tf.float32, [None, n_inputs])

basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
output_seqs, states = tfc.rnn.static_rnn(
    basic_cell, [X0, X1], dtype=tf.float32
)

Y0, Y1 = output_seqs

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={X0: X0_batch, X1: X1_batch})

print(Y0_val)
print(Y1_val)
