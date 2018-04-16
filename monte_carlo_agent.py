import gym
import numpy as np
import math
import itertools

class MonteCarloAgent(object):
    def __init__(self, action_space, observation_space, min_epsilon = 0.1):
        # set discount factor
        self.gamma = 0.999
        # set exploration rate
        self.min_epsilon = min_epsilon
        self.action_space = action_space
        self.observation_space = observation_space
        # position of the cart, velocity of the cart, angle of the pole, rotation rate of the pole,
        self.features = (1, 1, 6, 12,)
        self.accumulative_returns = np.zeros(self.features + (self.action_space.n,)) # selected action 
        self.samples = np.zeros(self.features + (self.action_space.n,)) # selected action 
        self.Q = np.zeros(self.features + (self.action_space.n,)) # selected action 
        self.policy = np.random.randint(2, size=self.features + (1,)) # selected action
        self.update_checkup = np.zeros(self.features + (1,))

        self.current_action = None
        self.current_observation = None
        
        self.experiences = []

        self.ada_divisor = 1000.

        # I suspect that the previous experiences needs to be forgotten in order to learn the best policy
        self.forget_rate = 0.88
        
    def get_epsilon(self, t):
        return max(self.min_epsilon, min(1.0, 1.0 - math.log10((t + 1) / self.ada_divisor)))

    def get_alpha(self, t):
        return max(self.min_alpha, min(1.0, 1.0 - math.log10((t + 1) / self.ada_divisor)))
        
    def discretize(self, observation):
        upper_bounds = [self.observation_space.high[0], 0.5, self.observation_space.high[2], math.radians(50)]
        lower_bounds = [self.observation_space.low[0], -0.5, self.observation_space.low[2], -math.radians(50)]
        ratios = [(observation[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(observation))]
        discretized_observation = [int(round((self.features[i] - 1) * ratios[i])) for i in range(len(observation))]
        discretized_observation = [min(self.features[i] - 1, max(0, discretized_observation[i])) for i in range(len(observation))]
        return tuple(discretized_observation)

    def act(self, observation, i_episode):
        observation  = self.discretize(observation)
        self.current_observation = observation
        self.current_action = self.policy[observation][0]
        
        return self.current_action

    def learn(self, reward, new_observation, i_episode):
        self.experiences.append((self.current_observation, self.current_action, reward))

    def review(self, i_episode):
        returns = []
        previous_reward = 0
        for _, _, reward in self.experiences[::-1]:
            the_return = reward + previous_reward * self.gamma
            returns.append(the_return)
            previous_reward = the_return
        returns = returns[::-1]


        
        for i, (state, action, reward) in enumerate(self.experiences):
            if self.update_checkup[state] == 0:

                self.accumulative_returns[state][action] *= self.forget_rate
                self.samples[state][action] *=self.forget_rate
                
                self.accumulative_returns[state][action] += returns[i]
                self.samples[state][action] += 1
                self.update_checkup[state] = 1



                
        self.Q = self.accumulative_returns/self.samples
        epsilon = self.get_epsilon(i_episode)
        for i, j, k, q in  itertools.product(xrange(self.features[0]), xrange(self.features[1]), xrange(self.features[2]), xrange(self.features[3])):
            self.policy[(i,j,k,q)] = self.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.Q[(i,j,k,q)])

        self.experiences = []
        self.update_checkup = np.zeros(self.features + (1,))
