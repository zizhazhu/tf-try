import gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tfc

tf.enable_eager_execution()


def discount_rewards(rewards, discount_rate):
    discounted_rewards = list(rewards)
    for i in reversed(range(len(discounted_rewards) - 1)):
        discounted_rewards[i] += discounted_rewards[i+1] * discount_rate
    return discounted_rewards


def discount_and_normalize_rewards(all_rewards, discount_rate=0.95):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_std) / reward_mean
            for discounted_rewards in all_discounted_rewards]


class nnPolicier(tf.keras.Model):
    def __init__(self):
        super(nnPolicier, self).__init__()
        self.hidden_layer = tf.layers.Dense(4, activation=tf.nn.relu)
        self.output_layer = tf.layers.Dense(1)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

    def call(self, obs):
        obs = tf.reshape(tfc.eager.Variable(obs, trainable=False, dtype=tf.float32), [1, -1])
        self.logits = self._get_logits(obs)
        output = tf.nn.sigmoid(self.logits)
        prob = tf.concat([1 - output, output], axis=1)
        action = tf.multinomial(tf.log(prob), num_samples=1)
        self.y = tf.to_float(action)

        return action

    def _get_logits(self, obs):
        hidden = self.hidden_layer(obs)
        logits = self.output_layer(hidden)
        return logits

    def get_gradient(self):
        return self.optimizer.compute_gradients(self.loss, self.variables)

    def loss(self):
        return tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.logits)


env = gym.make('CartPole-v0')
nn_policy = nnPolicier()

for iteration in range(1000):
    all_rewards = []
    all_gradients = []
    for episode in range(10):
        rewards = []
        gradients = []
        obs = env.reset()
        for step in range(1000):
            with tf.GradientTape() as tape:
                action = nn_policy(obs)
                loss = nn_policy.loss()
            gradient = tape.gradient(loss, nn_policy.variables)
            gradient = [gradient_one.numpy() for gradient_one in gradient]
            obs, reward, done, info = env.step(action.numpy()[0][0])
            rewards.append(reward)
            gradients.append(gradient)
            if done:
                break
        all_rewards.append(rewards)
        all_gradients.append(gradients)
    all_rewards = discount_and_normalize_rewards(all_rewards)
    result_gradients = []
    for k in range(len(all_gradients[0][0])):
        mean_gradients = np.mean(
            [reward * all_gradients[i][j][k]
             for i, rewards in enumerate(all_rewards)
             for j, reward in enumerate(rewards)
             ], axis=0
        )
        result_gradients.append(mean_gradients)
    nn_policy.optimizer.apply_gradients(zip(result_gradients, nn_policy.variables))
