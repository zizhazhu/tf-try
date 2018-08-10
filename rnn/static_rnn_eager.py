import numpy as np
import tensorflow as tf
import tensorflow.contrib as tfc

tf.enable_eager_execution()

n_inputs = 3
n_neurons = 5

X0 = tf.constant([[0, 1, 2]], dtype=np.float32)
X1 = tf.constant([[9, 8, 7]], dtype=np.float32)

basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
output_seqs, states = tfc.rnn.static_rnn(
    basic_cell, [X0, X1], dtype=tf.float32
)

Y0, Y1 = output_seqs

print(Y0)
print(Y1)
