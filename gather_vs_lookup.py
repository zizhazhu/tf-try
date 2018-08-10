import tensorflow as tf

table = [1.0, 2.0, 3.0]
ids = [[0, 1], [1, 2]]
table_var = tf.Variable(table)
ids_var = tf.Variable(ids)
with tf.device('/gpu:0'):
    el = tf.nn.embedding_lookup(table_var, ids_var)
    ga = tf.gather(table_var, ids_var)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    el_r, ga_r = sess.run([el, ga])

print(el_r)
print(ga_r)
