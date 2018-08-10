import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

x = tf.constant([[1, 2, 3], [2, 3, 4]])
y = tf.constant([[2], [3]])
bias = [1, 2, 3]
m = x * y + bias
print(x.get_shape())
print(m)

