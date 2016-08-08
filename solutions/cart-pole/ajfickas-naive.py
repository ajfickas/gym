import gym
env = gym.make('CartPole-v0')
# env.monitor.start('./tmp/cartpole-experiment-1')
observation = env.reset()
for t in range(100):
    env.render()
    print(observation)
    action = -1 if t % 2 == 0 else 1
    # action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break
# env.monitor.close()
