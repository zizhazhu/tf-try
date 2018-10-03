import os
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
n_hidden = (500, 500, 20, 500, 500, n_inputs)
n_outputs = n_inputs

learning_rate = 0.001
l2_reg = 0.001


def kl_divergence(p, q):
    return p * tf.log(p / q) + (1 - p) * tf.log((1 - p) / (1 - q))


class VAE(tf.keras.layers.Layer):

    def __init__(self, layers_units):
        super(VAE, self).__init__()
        self.layers_units = layers_units

        self.all_weights = []
        self.all_biases = []
        self.layers = []

    def build(self, input_shape):
        initializer = tf.variance_scaling_initializer()
        last_units = input_shape.as_list()[1]

        for index, layer_units in enumerate(self.layers_units):
            bias = self.add_weight('layer_{}_bias'.format(index), [layer_units], dtype=tf.float32,
                                   initializer=tf.zeros_initializer)
            weights = self.add_weight('layer_{}_weights'.format(index), [last_units, layer_units],
                                      dtype=tf.float32, initializer=initializer)

            if index == (len(self.layers_units) - 1) // 2:
                weights_gamma = self.add_weight('layer_{}_weights_gamma'.format(index), [last_units, layer_units],
                                                dtype=tf.float32, initializer=initializer)
                bias_gamma = self.add_weight('layer_{}_bias_gamma'.format(index), [layer_units], dtype=tf.float32,
                                             initializer=tf.zeros_initializer)
                weights = (weights, weights_gamma)
                bias = (bias, bias_gamma)

            self.all_biases.append(bias)
            self.all_weights.append(weights)

            last_units = layer_units

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        layer = inputs
        for i in range(len(self.all_weights) - 1):
            if isinstance(self.all_weights[i], tuple):
                layer_mean = layer @ self.all_weights[i][0] + self.all_biases[i][0]
                layer_gamma = layer @ self.all_weights[i][1] + self.all_biases[i][1]
                layer_sigma = tf.exp(0.5 * layer_gamma)
                noise = tf.random_normal(tf.shape(layer_sigma), dtype=tf.float32)
                layer = layer_mean + layer_sigma * noise
                self.layer_gamma = layer_gamma
                self.layer_mean = layer_mean
            else:
                layer = layer @ self.all_weights[i] + self.all_biases[i]
                layer = tf.nn.elu(layer)
            self.layers.append(layer)
        logits = layer @ self.all_weights[-1] + self.all_biases[-1]
        return logits

    def output(self, inputs, **kwargs):
        logits = self(inputs, **kwargs)
        output = tf.sigmoid(logits)
        return output

    def loss(self, X, y):
        logits = self(X, is_train=True)
        latent_loss = 0.5 * tf.reduce_mean(
            tf.exp(self.layer_gamma) + tf.square(self.layer_mean) - 1 - self.layer_gamma
        )
        reconstruction_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y, logits=logits
        ))
        loss = latent_loss + reconstruction_loss
        return loss


model = VAE(n_hidden)
if os.path.exists('./temp/eager/vae.ckpt'):
    tfc.eager.restore_network_checkpoint(model, './temp/eager/vae.ckpt')

optimizer = tf.train.AdamOptimizer(learning_rate)
train_dataset = mnist.train('temp/mnist').batch(512).repeat(args.epochs)
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

tfc.eager.save_network_checkpoint(model, './temp/checkpoint/vae.ckpt')
