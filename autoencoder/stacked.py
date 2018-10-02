import argparse
import tensorflow as tf
import tensorflow.contrib as tfc
import matplotlib.pyplot as plt

import data.mnist as mnist

parser = argparse.ArgumentParser()
parser.add_argument('--show', action='store_true')
parser.add_argument('--epochs', type=int, default=1)
args = parser.parse_args()

tf.enable_eager_execution()

n_inputs = 28 * 28
n_hidden = (1000, 1000)
n_outputs = n_inputs

learning_rate = 0.01
l2_reg = 0.001


def kl_divergence(p, q):
    return p * tf.log(p / q) + (1 - p) * tf.log((1 - p) / (1 - q))


class Stacked(tf.keras.layers.Layer):

    def __init__(self, layers_units, weights_reuse=False, noisy=False, keepprob=None, kl_div=False):
        super(Stacked, self).__init__()
        self.layers_units = layers_units
        self.noisy = noisy
        self.keepprob = keepprob
        self.weights_reuse = weights_reuse
        self.kl_div = kl_div

        self.all_weights = []
        self.all_biases = []

    def build(self, input_shape):
        initializer = tf.variance_scaling_initializer()
        all_units = list(self.layers_units[:-1])
        all_units.extend(reversed(self.layers_units))
        all_units.append(n_outputs)
        last_units = n_inputs

        for index, layer_units in enumerate(all_units):
            bias = self.add_weight('layer_{}_bias'.format(index), [layer_units], dtype=tf.float32,
                                   initializer=tf.zeros_initializer)
            self.all_biases.append(bias)

            if index >= len(all_units) // 2 and self.weights_reuse:
                weights = tf.transpose(self.all_weights[len(all_units) - index - 1], name='layer_{}_weights')
            else:
                weights = self.add_weight('layer_{}_weights'.format(index), [last_units, layer_units],
                                          dtype=tf.float32, initializer=initializer)
            self.all_weights.append(weights)

            last_units = layer_units

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        if self.noisy and kwargs['is_train']:
            inputs = inputs + tf.random_normal(tf.shape(inputs))
        if self.keepprob:
            inputs = tf.layers.dropout(inputs, rate=(1-self.keepprob), training=kwargs['is_train'])
        layer = inputs
        for i in range(len(self.all_weights) - 1):
            layer = layer @ self.all_weights[i] + self.all_biases[i]
            layer = tf.nn.elu(layer)
        logits = layer @ self.all_weights[-1] + self.all_biases[-1]
        return logits

    def loss(self, X, y):
        logits = self(X, is_train=True)
        if self.kl_div:
            hidden1 = tf.nn.sigmoid(X @ self.all_weights[0] + self.all_biases[0])
            hidden1_mean = tf.reduce_mean(hidden1, axis=0)
            reg_loss = kl_divergence(0.1, hidden1_mean) * 0.2
        else:
            reg_loss = tf.reduce_sum(tfc.layers.apply_regularization(tfc.layers.l2_regularizer(l2_reg), self.variables))
        reconstruction_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y, logits=logits
        ))
        loss = tf.reduce_sum(reg_loss) + reconstruction_loss
        return loss


optimizer = tf.train.AdamOptimizer(learning_rate)
model = Stacked(n_hidden, kl_div=True)
train_dataset = mnist.train('temp/mnist').batch(64).repeat(args.epochs)
plt.axis('off')
for index, (images, _) in enumerate(train_dataset):
    with tf.GradientTape() as tape:
        loss = model.loss(images, images)
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(zip(grads, model.variables))
    if index % 100 == 0:
        print('step{}: {}'.format(index, loss))
        if args.show:
            y = tf.nn.sigmoid(model(images, is_train=False))
            plt.suptitle(index)
            plt.subplot(121)
            plt.imshow(tf.reshape(images[0], [28, 28]))
            plt.subplot(122)
            plt.imshow(tf.reshape(y[0], [28, 28]))
            plt.show()

