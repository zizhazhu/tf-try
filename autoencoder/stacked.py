import argparse
import tensorflow as tf
import tensorflow.contrib as tfc
import matplotlib.pyplot as plt

import data.mnist as mnist

parser = argparse.ArgumentParser()
parser.add_argument('--show', action='store_true')
args = parser.parse_args()

tf.enable_eager_execution()

n_inputs = 28 * 28
n_hidden = (300, 150)
n_outputs = n_inputs

learning_rate = 0.01
l2_reg = 0.001


class Stacked(tf.keras.layers.Layer):

    def __init__(self, layers_units):
        super(Stacked, self).__init__()
        self.layers_units = layers_units
        self.all_weights = []
        self.all_biases = []

    def build(self, input_shape):
        initializer = tf.variance_scaling_initializer()
        last_units = n_inputs
        for index, layer_units in enumerate(self.layers_units):
            weights = self.add_weight('weights_{}'.format(index), [last_units, layer_units],
                                      dtype=tf.float32, initializer=initializer)
            self.all_weights.append(weights)
            bias = self.add_weight('bias_{}'.format(index), [layer_units], dtype=tf.float32,
                                   initializer=tf.zeros_initializer)
            self.all_biases.append(bias)
            last_units = layer_units
        length = len(n_hidden)
        for index in range(length, length * 2):
            weights = tf.transpose(self.all_weights[length * 2 - index - 1])
            self.all_weights.append(weights)
            bias = self.add_weight('bias_{}'.format(index), [weights.get_shape().as_list()[1]],
                                   dtype=tf.float32, initializer=tf.zeros_initializer)
            self.all_biases.append(bias)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        layer = inputs
        for i in range(len(self.all_weights)):
            layer = layer @ self.all_weights[i] + self.all_biases[i]
        return layer

    def loss(self, X, y):
        reconstruction_loss = tf.reduce_mean(tf.square(X - y))
        reg_loss = tfc.layers.apply_regularization(tfc.layers.l2_regularizer(l2_reg), self.variables)
        loss = tf.reduce_sum(reg_loss) + reconstruction_loss
        return loss


optimizer = tf.train.AdamOptimizer(learning_rate)
model = Stacked(n_hidden)
train_dataset = mnist.train('temp/mnist').batch(64).repeat(1)
plt.axis('off')
for index, (images, _) in enumerate(train_dataset):
    with tf.GradientTape() as tape:
        y = model(images)
        loss = model.loss(images, y)
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(zip(grads, model.variables))
    if index % 100 == 0:
        print('step{}: {}'.format(index, loss))
        if args.show:
            plt.suptitle(index)
            plt.subplot(121)
            plt.imshow(tf.reshape(images[0], [28, 28]))
            plt.subplot(122)
            plt.imshow(tf.reshape(y[0], [28, 28]))
            plt.show()

