import numpy as np
from matplotlib import pyplot as plt
import gym

mspacman_color = np.array([210, 164, 74]).mean()

def preprocess_observation(obs):
    img = obs[1:176:2, ::2]
    img = img.mean(axis=2)
    img[img==mspacman_color] = 0
    img = (img - 128) / 128 - 1
    return img.reshape(88, 80, 1)

env = gym.make("MsPacman-v0")
obs = env.reset()
plt.imshow(obs)
plt.show()
preprocessed = preprocess_observation(obs)
plt.imshow(preprocessed.reshape(88, 80), cmap='gray')
plt.show()

