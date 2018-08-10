import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

n_inputs = 3
n_neurons = 5

X0 = np.array([[0, 1, 2]], dtype=np.float32)
X1 = np.array([[9, 8, 7]], dtype=np.float32)

Wx = tfe.Variable(tf.random_normal(shape=[n_inputs, n_neurons]))
Wy = tfe.Variable(tf.random_normal(shape=[n_neurons, n_neurons]))
b = tfe.Variable(tf.zeros([1, n_neurons]))

Y0 = tf.tanh(tf.matmul(X0, Wx))
Y1 = tf.tanh(tf.matmul(Y0, Wy) + tf.matmul(X1, Wx) + b)

print(Y0)
print(Y1)
