import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

var = [[[i + j + k for i in range(3)] for j in range(2)] for k in range(5)]

w = tfe.Variable(var)

v_sum_square = tf.square(tf.reduce_sum(w, axis=1))
v_square_sum = tf.reduce_sum(tf.square(w), axis=1)
cross_logits = 0.5 * tf.reduce_sum(v_sum_square - v_square_sum, axis=1, keepdims=True)

temp_a = tf.reduce_sum(tf.matmul(w, w, transpose_b=True), axis=(1, 2))
temp_b = tf.reduce_sum(tf.square(w), axis=(1, 2))
cross_logits_2 = 0.5 * (temp_a - temp_b)

print(cross_logits, cross_logits_2)
