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
n_hidden = (300, 150, 300)
n_outputs = n_inputs

learning_rate = 0.01
l2_reg = 0.001


class Stacked(tf.keras.Model):

    def __init__(self, layers_units):
        super().__init__()
        initializer = tf.variance_scaling_initializer()
        self.hidden = []
        for layer_units in layers_units:
            self.hidden.append(tf.layers.Dense(layer_units,
                                               activation=tf.nn.elu,
                                               kernel_initializer=initializer))
        self.outputs = tf.layers.Dense(n_outputs)

    def call(self, inputs):
        layer = inputs
        for dense_layer in self.hidden:
            layer = dense_layer(layer)
        outputs = self.outputs(layer)
        return outputs

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

