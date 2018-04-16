import gym
from sarsa_agent import SARSAAgent
from monte_carlo_agent import MonteCarloAgent
from tabular_q_agent import TabularQAgent

class Constants():
    epochs = 500000
    iters = 1000

env = gym.make('CartPole-v0')
env.reset()
constants = Constants()
agent = MonteCarloAgent(env.action_space, env.observation_space)

done = False
reward = 0
history = []

for i_episode in xrange(constants.epochs):
    observation = env.reset()
    rewards = []
    for t in range(constants.iters):
        action = agent.act(observation, i_episode)
        observation, reward, done, _ = env.step(action)
        agent.learn(reward, observation, i_episode)
        if done:
            history.append(t+1)
            break
    agent.review(i_episode)

    if i_episode%200 == 0:
        print "Episode finised on average in {} timesteps for last 200 runs.".format(sum(history)/200.)
        history = []
