from collections import namedtuple, deque
import numpy as np
import gym
import tensorflow as tf


mspacman_color = np.array([210, 164, 74]).mean()


def preprocess_observation(obs):
    img = obs[1:176:2, ::2]
    img = img.mean(axis=2)
    img[img==mspacman_color] = 0
    img = (img - 128) / 128 - 1
    return img.reshape(88, 80, 1)


n_outputs = 9
discount_rate = 0.95
eps_min = 0.05
eps_max = 1.0
eps_decay_steps = 50000
replay_memory_size = 10000
replay_memory = deque([], maxlen=replay_memory_size)

env = gym.make("MsPacman-v0")
conv_layer = namedtuple('conv_layer', ('filter_n', 'kernel_size', 'stride', 'padding'))
conv_layers = [
    conv_layer(32, 8, 4, 'SAME'),
    conv_layer(64, 4, 2, 'SAME'),
    conv_layer(64, 3, 1, 'SAME'),
]


def epsilon_greedy(q_values, step):
    epsilon = max(eps_min, eps_max - (eps_max - eps_min) * step / eps_decay_steps)
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs)
    else:
        return np.argmax(q_values)


def sample_memories(batch_size):
    indices = np.random.permutation(len(replay_memory))[:batch_size]
    cols = [[], [], [], [], []]
    for idx in indices:
        memory = replay_memory[idx]
        for col, value in zip(cols, memory):
            col.append(value)
    cols = [np.array(col) for col in cols]
    return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1)


class QModel(tf.keras.Model):
    def __init__(self):
        super(QModel, self).__init__()
        self.conv_layers = []
        for layer_conf in conv_layers:
            self.conv_layers.append(tf.layers.Conv2D(layer_conf.filter_n, layer_conf.kernel_size,
                                     layer_conf.stride, layer_conf.padding, activation=tf.nn.relu))
        self.dense_layer = []
        self.dense_layer.append(tf.layers.Dense(512, tf.nn.relu))
        self.dense_layer.append(tf.layers.Dense(9))

    def output(self, x_state):
        layer = tf.reshape(tf.convert_to_tensor(x_state), shape=(-1, 88, 80, 1))
        for conv_layer in self.conv_layers:
            layer = conv_layer(layer)
        layer_flat = tf.reshape(layer, [-1, 64 * 11 * 10])
        hidden = self.dense_layer[0](layer_flat)
        outputs = self.dense_layer[1](hidden)
        return outputs

    def loss(self, x_state, x_action, y):
        outputs = self.output(x_state)
        q_values = tf.reduce_sum(outputs * tf.one_hot(x_action, n_outputs, dtype=tf.float64), axis=1, keepdims=True)
        cost = tf.reduce_mean(tf.square(q_values - y))
        return cost

    def call(self, x):
        return self.output(x)


tf.enable_eager_execution()

q_network = QModel()
optimizer = tf.train.AdamOptimizer()
done = True
step = 0
while True:
    step += 1
    if done:
        obs = env.reset()
        for skip in range(90):
            obs, reward, done, info = env.step(0)
        state = preprocess_observation(obs)

    q_values = q_network(state)
    action = epsilon_greedy(q_values, step)

    obs, reward, done, info = env.step(action)
    next_state = preprocess_observation(obs)

    replay_memory.append((state, action, reward, next_state, 1.0 - done))
    state = next_state

    env.render()

    if len(replay_memory) < 1000:
        continue

    x_state_batch, x_action_batch, reward_batch, x_next_state_batch, continue_batch = sample_memories(64)
    next_q_values = q_network(x_next_state_batch)
    max_next_q_batch = np.max(next_q_values, axis=1, keepdims=True)
    y_val = reward_batch + continue_batch * discount_rate * max_next_q_batch
    with tf.GradientTape() as tape:
        loss = q_network.loss(x_state_batch, x_action_batch, y_val)
    grads = tape.gradient(loss, q_network.variables)
    optimizer.apply_gradients(
        zip(grads, q_network.variables),
        global_step=tf.train.get_or_create_global_step()
    )

    if step % 100 == 0:
        print("{}: {}".format(step, loss))
