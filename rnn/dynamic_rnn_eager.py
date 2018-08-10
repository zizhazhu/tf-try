import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

n_inputs = 3
n_neurons = 5

X = np.array(
    [
        [[0, 1, 2], [9, 8, 7]],
        [[3, 4, 5], [0, 0, 0]],
        [[6, 7, 8], [6, 5, 4]],
        [[9, 0, 1], [3, 2, 1]],
    ], dtype=np.float32)
X_length = np.array([2, 1, 2, 2])

basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
output_seqs, states = tf.nn.dynamic_rnn(
    basic_cell, X, dtype=tf.float32, sequence_length=X_length
)

print(output_seqs)
print(states)
