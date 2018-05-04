import gym
import numpy as np
import math
import tensorflow as tf
import copy

# DISCLAIMER: I CANNOT GERANTEE THE IMPLEMENTED ALGORITHM IS 100% CORRECT
# CHECK baseline by openai for furthur references.


class Policy(object):
    def __init__(self,
                 scope,
                 observation_space,
                 action_space,
                 session,
                 temprature=0.1):
        self.observation_space = observation_space
        self.action_space = action_space
        self.session = session

        with tf.variable_scope(scope):
            self.observation = tf.placeholder(dtype=tf.float32, shape=[None] + list(observation_space.shape),name='observation')
            with tf.variable_scope('policy'):
                layer_1 = tf.layers.dense(inputs=self.observation, units=20, activation=tf.tanh)
                layer_2 = tf.layers.dense(inputs=layer_1, units=20, activation=tf.tanh)
                layer_3 = tf.layers.dense(inputs=layer_2, units=self.action_space.n, activation=tf.tanh)
                self.action_probs = tf.layers.dense(inputs=tf.divide(layer_3, temprature), units = action_space.n, activation=tf.nn.softmax)

            with tf.variable_scope('value_function'):
                layer_1 = tf.layers.dense(inputs=self.observation, units=20, activation=tf.tanh)
                layer_2 = tf.layers.dense(inputs=layer_1, units=20, activation=tf.tanh)
                self.value_prediction = tf.layers.dense(inputs=layer_2, units=1, activation=None)
            stochastic_action = tf.multinomial(tf.log(self.action_probs), num_samples = 1)
            self.stochastic_action = tf.reshape(stochastic_action, shape=[-1])

            self.diterministic_action = tf.argmax(self.action_probs, axis=1)
            
            self.scope = tf.get_variable_scope().name

    def act(self, observation, stochastic):
        if stochastic:
            return self.session.run([self.stochastic_action, self.value_prediction], feed_dict={self.observation: observation})
        else:
            return self.session.run([self.diterministic_action, self.value_prediction], feed_dict={self.observation: observation})

    def get_action_probability(observation):
        return tf.get_default_session().run(self.action_probs, feed_dict={self.observation:observation})

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,self.scope)
        
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,self.scope)



class PPOAgent(object):

    def __init__(self, action_space, observation_space, episodes, clip_value = 0.2, c_1=1, c_2=0.01):
        # set discount factor
        self.gamma = 0.99
        
        # set exploration rate
        self.action_space = action_space
        self.observation_space = observation_space

        self.sess = tf.Session()

        # receive action and observation space.
        # position of the cart, velocity of the cart, angle of the pole, rotation rate of the pole,
        self.current_action = None
        self.current_observation = None


        # build policy networks
        self.current_policy = Policy('current_policy', observation_space, action_space, self.sess)
        self.old_policy = Policy('old_policy', observation_space, action_space, self.sess)

        # receive trainable variables
        current_trainables = self.current_policy.get_trainable_variables()
        old_trainables = self.old_policy.get_trainable_variables()

        # define the assignment operation to give the current parameters in the network to the the old network
        with tf.variable_scope('assignment_operation'):
            self.assignment_operation = []
            for value_old, value_current in zip(old_trainables, current_trainables):
                self.assignment_operation.append(tf.assign(value_old, value_current))

        # define training operation, first we difien the inputs
        with tf.variable_scope('train_operation'):
            self.actions = tf.placeholder(dtype = tf.int32, shape=[None], name='actions')
            self.rewards = tf.placeholder(dtype = tf.float32, shape=[None], name='rewards')
            self.next_value_predictions = tf.placeholder(dtype = tf.float32, shape=[None], name='next_value_predictions')
            self.general_advantage_estimators = tf.placeholder(dtype = tf.float32, shape=[None], name='general_advantage_estimators')

        current_action_probabilities = self.current_policy.action_probs
        old_action_probabilities = self.old_policy.action_probs

        # change the represetnation of the current action probabilities
        current_action_probabilities = current_action_probabilities * \
                       tf.one_hot(indices=self.actions, depth=current_action_probabilities.shape[-1])
        current_action_probabilities = tf.reduce_sum(current_action_probabilities, axis = 1)
        
        # change the representation of the old action probabilties
        old_action_probabilities = old_action_probabilities * \
                       tf.one_hot(indices=self.actions, depth=old_action_probabilities.shape[-1])
        old_action_probabilities = tf.reduce_sum(old_action_probabilities, axis = 1)

        #construct the genral loss based on clipping
        with tf.variable_scope('loss/clip'):
            ratios = tf.exp(tf.log(current_action_probabilities) - \
                            tf.log(old_action_probabilities))
            clipped_ratios = tf.clip_by_value(ratios, clip_value_min = 1- clip_value, clip_value_max = 1 + clip_value)
            loss_clip = tf.minimum(tf.multiply(self.general_advantage_estimators, ratios),
                                   tf.multiply(self.general_advantage_estimators, clipped_ratios))
            loss_clip = tf.reduce_mean(loss_clip)
            tf.summary.scalar('loss_clip',loss_clip)

        # construct the loss of value function

        with tf.variable_scope('loss/value_function'):
            value_predictions = self.current_policy.value_prediction
            loss_value_function = tf.squared_difference(self.rewards + self.gamma * self.next_value_predictions, value_predictions)
            loss_value_function = tf.reduce_mean(loss_value_function)
            tf.summary.scalar('loss_value_function', loss_value_function)
        
        # construct the loss of entropy bonus
        with tf.variable_scope('loss/entropy'):
            loss_entropy = -tf.reduce_sum(self.current_policy.action_probs * \
                                     tf.log(tf.clip_by_value(self.current_policy.action_probs, 1e-10, 1.0)),
                                     axis = 1)
            loss_entropy = tf.reduce_mean(loss_entropy, axis = 0)
            tf.summary.scalar('loss_entropy', loss_entropy)
        
        with tf.variable_scope('loss'):
            loss = loss_clip - c_1 * loss_value_function + c_2 * loss_entropy
            loss = -loss
            tf.summary.scalar('loss', loss)

        
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-5)
        self.train_operation = optimizer.minimize(loss, var_list=current_trainables)

        
        # start tf related operations
        self.saver = tf.train.Saver()
        #self.writer = tf.summary.FileWriter('.log/train', sess.graph)
        self.sess.run(tf.global_variables_initializer())

        # make list for all the training data
        self.observations_epoch = []
        self.actions_epoch = []
        self.value_predictions_epoch = []
        self.nvalue_predictions_epoch = None
        self.rewards_epoch = []
        self.general_advantage_estimators_epoch = None

        # define current data
        self.train_observation = None
        self.train_action = None
        self.train_value_prediction = None
        self.train_reward = 0

    def train(self, observations, actions, rewards, next_value_predictions, general_advantage_estimators):
        return self.sess.run([self.train_operation],
                             feed_dict = {self.current_policy.observation: observations,
                                          self.old_policy.observation: observations,
                                          self.actions:actions,
                                          self.rewards:rewards,
                                          self.next_value_predictions:next_value_predictions,
                                          self.general_advantage_estimators: general_advantage_estimators})
    def assign_policy_parameters(self):
        return self.sess.run(self.assignment_operation)

    def get_general_advantage_estimators(self, rewards, value_predictions, next_value_predictions):
        delta = [rt + self.gamma * vn - v for rt, vn, v in zip(rewards, next_value_predictions, value_predictions)]
        general_advantage_estimators = copy.deepcopy(delta)

        for t in reversed(range(len(general_advantage_estimators) - 1)):
            general_advantage_estimators[t] = general_advantage_estimators[t] + self.gamma * general_advantage_estimators[t+1]

        return general_advantage_estimators

    def act(self, observation, i_episode):
        self.train_observation = observation
        self.train_observation = np.stack([self.train_observation]).astype(dtype=np.float32)
        action, value_prediction = self.current_policy.act(observation=self.train_observation, stochastic=True)
        self.train_action = np.asscalar(action)
        self.train_value_prediction = np.asscalar(value_prediction)
        return self.train_action

    def learn(self, reward, new_observation, i_episode):
        self.train_reward = reward
        self.observations_epoch.append(self.train_observation)
        self.actions_epoch.append(self.train_action)
        self.value_predictions_epoch.append(self.train_value_prediction)
        self.rewards_epoch.append(self.train_reward)

    def review(self, i_episode):
        self.nvalue_predictions_epoch = self.value_predictions_epoch[1:] + [0]
        self.general_advantage_estimators_epoch = self.get_general_advantage_estimators(rewards=self.rewards_epoch,
                                                                                        value_predictions=self.value_predictions_epoch,
                                                                                        next_value_predictions=self.nvalue_predictions_epoch)
        
        self.observations_epoch = np.reshape(self.observations_epoch, newshape=[-1] + list(self.observation_space.shape))
        self.actions_epoch = np.array(self.actions_epoch).astype(dtype=np.int32)
        self.rewards_epoch = np.array(self.rewards_epoch).astype(dtype=np.float32)
        self.nvalue_predictions_epoch = np.array(self.nvalue_predictions_epoch).astype(dtype=np.float32)
        self.general_advantage_estimators_epoch = np.array(self.general_advantage_estimators_epoch).astype(dtype=np.float32)
        self.general_advantage_estimators_epoch = (self.general_advantage_estimators_epoch - \
                                                   self.general_advantage_estimators_epoch.mean()) / self.general_advantage_estimators_epoch.std()
        self.assign_policy_parameters()

        inputs = [self.observations_epoch,
                  self.actions_epoch,
                  self.rewards_epoch,
                  self.nvalue_predictions_epoch,
                  self.general_advantage_estimators_epoch]

        for epoch in xrange(4):
            sample_indices = np.random.randint(low=0,
                                               high=self.observations_epoch.shape[0],
                                               size=64)
            sample_inputs = [np.take(a=a,
                                     indices=sample_indices,
                                     axis=0)
                             for a in inputs]
            self.train(observations=sample_inputs[0],
                       actions=sample_inputs[1],
                       rewards=sample_inputs[2],
                       next_value_predictions=sample_inputs[3],
                       general_advantage_estimators=sample_inputs[4])
        
        # reset training set
        self.observations_epoch = []
        self.actions_epoch = []
        self.value_predictions_epoch = []
        self.nvalue_predictions_epoch = None
        self.rewards_epoch = []
        self.general_advantage_estimators_epoch = None
        
                                      

