import tensorflow as tf

with tf.Graph().as_default() as g:
    with tf.variable_scope('use', reuse=tf.AUTO_REUSE):
        a = tf.get_variable('a', shape=[10, 1], dtype=tf.float32, initializer=tf.constant_initializer(0))
        b = tf.get_variable('b', shape=[10, 1], dtype=tf.float32, initializer=tf.constant_initializer(1))
        a_assign = tf.assign(a, b)
    g.

with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(a))
    print(sess.run(a_assign))
    print(sess.run(a))

with tf.Graph().as_default() as h:
    with tf.variable_scope('use', reuse=tf.AUTO_REUSE):
        a = tf.get_variable('a', shape=[10, 1], dtype=tf.float32, initializer=tf.constant_initializer(0))

with tf.Session(graph=h) as sess:
    print(sess.run(a))
