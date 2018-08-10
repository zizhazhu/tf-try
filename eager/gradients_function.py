import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()


def square(x):
    return tf.multiply(x, x)


grad = tfe.gradients_function(square)

print(square(3.))
print(grad(3.))

gradgrad = tfe.gradients_function(lambda x: grad(x))
print(gradgrad(3.))

gradgradgrad = tfe.gradients_function(lambda x: gradgrad(x))
print(gradgradgrad(3.))


def abs(x):
    return x if x > 0. else -x

grad = tfe.gradients_function(abs)
print(grad(3.))
print(grad(-3.))
print(grad(0.))
