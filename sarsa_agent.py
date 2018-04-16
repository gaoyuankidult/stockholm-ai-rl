import gym
import numpy as np
import math

class SARSAAgent(object):
    def __init__(self, action_space, observation_space, min_alpha = 0.1, min_epsilon = 0.0001):
        # set learning rate
        self.min_alpha = min_alpha
        # set discount factor
        self.gamma = 0.999
        # set exploration rate
        self.min_epsilon = min_epsilon
        self.action_space = action_space
        self.observation_space = observation_space
        # position of the cart, velocity of the cart, angle of the pole, rotation rate of the pole,
        self.features = (1, 1, 6, 12,)
        self.Q = np.zeros(self.features + (self.action_space.n,))
        self.current_action = None
        self.next_action = None
        self.current_observation = None

        # adaptation advisor
        self.ada_divisor = 100.

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
        epsilon = self.get_epsilon(i_episode)
        if self.next_action == None:
            self.current_action = self.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.Q[self.current_observation])
        else:
            self.current_action = self.next_action

        return self.current_action

    def learn(self, reward, new_observation, i_episode):
        alpha = self.get_alpha(i_episode)
        epsilon = self.get_epsilon(i_episode)
        
        new_observation = self.discretize(new_observation)
        
        self.next_action = self.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.Q[new_observation])
        
        self.Q[self.current_observation][self.current_action] += alpha * \
                                                                 (reward + self.gamma * self.Q[new_observation][self.next_action] - \
                                                                  self.Q[self.current_observation][self.current_action])
                                    
    def review(self, i_episode):
        pass
