import gym
env = gym.make('LunarLander-v2')
for i in range(20):
    observation = env.reset()
    for t in range(200):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timestemps".format(t+1))
            break
