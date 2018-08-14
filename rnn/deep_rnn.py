import numpy as np
import tensorflow as tf
import tensorflow.contrib as tfc

tf.enable_eager_execution()

X = [i for i in range(1, 21)]
y = [i for i in range(2, 22)]

n_steps = 20
n_inputs = 1
n_neurons = 100
n_outputs = 1
epochs = 100

learning_rate = 0.001

X = tf.cast(tf.reshape(X, [1, 20, 1]), dtype=tf.float32)
y = tf.cast(tf.reshape(y, [1, 20, 1]), dtype=tf.float32)
cells = [tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu) for _ in range(3)]
multi_cell = tfc.rnn.OutputProjectionWrapper(
    tf.nn.rnn_cell.MultiRNNCell(cells),
    output_size=1
)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
for i in range(epochs):
    with tf.GradientTape() as tape:
        outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)
        loss = tf.reduce_mean(tf.square(outputs - y))
    grads = tape.gradient(loss, multi_cell.weights)
    optimizer.apply_gradients(zip(grads, multi_cell.weights), global_step=tf.train.get_or_create_global_step())
    print("loss:{}".format(loss))

X_new = [i for i in range(1, 26)]
X_new = np.array(X_new, dtype=np.float32).reshape([1, -1, 1])
outputs, states = tf.nn.dynamic_rnn(multi_cell, X_new, dtype=tf.float32)
print(outputs)
