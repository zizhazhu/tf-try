import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

n_inputs = 3
n_hidden = 2
n_outputs = n_inputs

learning_rate = 0.01


class PCA(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.hidden = tf.layers.Dense(n_hidden)
        self.outputs = tf.layers.Dense(n_outputs)

    def call(self, inputs):
        hidden = self.hidden(inputs)
        outputs = self.outputs(hidden)

    def loss(self, X):
        loss = tf.reduce_mean(tf.square(X - self.call(X)))
        return loss


optimizer = tf.train.AdamOptimizer(learning_rate)
model = PCA()
X = np.ndarray([64, 3], dtype=np.float)
with tf.GradientTape() as tape:
    loss = model(X)
grads = tape.gradient(loss, model.variables)
optimizer.apply_gradients(zip(grads, model.variables))
