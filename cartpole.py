import gym
from sarsa_agent import SARSAAgent
from monte_carlo_agent import MonteCarloAgent
from tabular_q_agent import TabularQAgent
from ppo_agent import PPOAgent

class Constants():
    epochs = 200000
    iters = 1000

env = gym.make('CartPole-v0')
env.reset()
constants = Constants()
agent = PPOAgent(env.action_space,
                 env.observation_space,
                 episodes=constants.epochs)

done = False
reward = 0
history = []

for i_episode in xrange(constants.epochs):
    observation = env.reset()
    rewards = []
    for t in range(constants.iters):
        action = agent.act(observation, i_episode)
        next_observation, reward, done, _ = env.step(action)
        agent.learn(reward, next_observation, i_episode)
        observation = next_observation
        
        if done:
            history.append(t+1)
            break
    agent.review(i_episode)

    if i_episode%500 == 0:
        print "Episode finised on average in {} timesteps for last 500 runs.".format(sum(history)/500.)
        history = []
