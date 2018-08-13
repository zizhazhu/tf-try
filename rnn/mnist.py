import tensorflow as tf
import tensorflow.contrib as tfc
from tensorflow.examples.tutorials.mnist import input_data

tf.enable_eager_execution()


def batch_gen(*args, batch_size=512):
    length = len(args[0])
    for start in range(0, length, batch_size):
        last = length if start + batch_size >= length else start + batch_size
        yield (arg[start:last] for arg in args)


n_steps = 28
n_inputs = 28
n_neurons = 150
n_outputs = 10

mnist = input_data.read_data_sets('./tmp/mnist')
X_train = mnist.train.images.reshape((-1, n_steps, n_inputs))
y_train = mnist.train.labels
X_test = mnist.test.images.reshape((-1, n_steps, n_inputs))
y_test = mnist.test.labels

learning_rate = 0.001


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
        self.dense_layer = tf.layers.Dense(n_outputs)

    def output(self, X):
        outputs, states = tf.nn.dynamic_rnn(self.basic_cell, X, dtype=tf.float32)
        logits = self.dense_layer(states)
        return logits

    def accuracy(self, X, labels):
        labels = tf.cast(labels, tf.int32)
        logits = self.output(X)
        result = tf.arg_max(logits, dimension=1, output_type=tf.int32)
        acc = tf.reduce_mean(tf.cast(tf.equal(labels, result), tf.float32))
        return acc

    def loss(self, X, labels):
        labels = tf.cast(labels, tf.int32)
        logits = self.output(X)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits
        )
        loss = tf.reduce_mean(loss)
        return loss

    def call(self, inputs):
        return self.output(inputs)


optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
model = Model()

for X_batch, y_batch in batch_gen(X_train, y_train):
    with tf.GradientTape() as tape:
        loss = model.loss(X_batch, y_batch)
    grads = tape.gradient(loss, model.weights)
    optimizer.apply_gradients(zip(grads, model.weights),
                              global_step=tf.train.get_or_create_global_step())
    print('loss:{} Train acc:{} Test acc:{}'.format(loss, model.accuracy(X_batch, y_batch),
                                                    model.accuracy(X_test, y_test)))
