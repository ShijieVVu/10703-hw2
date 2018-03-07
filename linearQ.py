#!/usr/bin/env python
import argparse
import pickle as p
import random
import sys
from time import strftime, localtime, time

import gym
import numpy as np
from keras.optimizers import Adam
from keras.layers import Dense, Input
from keras.models import Model, load_model, Sequential


class QNetwork:

    # This class essentially defines the network architecture.
    # The network should take in state of the world as an input,
    # and output Q values of the actions available to the agent as the output.

    def __init__(self, ns, na):
        # Define your network architecture here. It is also a good idea to define any training operations
        # and optimizers here, initialize your variables, or alternately compile your model here.
        self.model = Sequential([
            Dense(na, input_shape=(ns,))
        ])

        self.model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))

    def train(self, s, target):
        def mse(A, B):
            err = 0
            i = 0
            for i, (a, b) in enumerate(zip(A, B), 1):
                err += (a - b) ** 2
            return err / i

        # print('s is {}, target is {}'.format(s, target))
        # print("before training the mse between s and target are {}".format(mse(target, self.qvalues(s))))
        self.model.fit(np.array(s), np.array(target), verbose=0)
        # print("after training the mse between s and target are {}".format(mse(target, self.qvalues(s))))

    def qvalues(self, s):
        # print('s is {}'.format(s))
        return self.model.predict(np.array([s])).tolist()[0]

    def save_model(self, model_file):
        # Helper function to save your model / weights.
        self.model.save(model_file)

    def load_model(self, model_file):
        # Helper function to load an existing model.
        self.model = load_model(model_file)
        self.target_model = load_model(model_file)


def load_agent(name):
    (env, gamma, ns, na, ss) = p.load(open(name + '.p', 'rb'))
    agent = DQNAgent(env, gamma)
    agent.net.load_model(name + '.h5')
    print("Agent loaded from {} created in {}".format(name + '.h5', ss))
    return agent


class DQNAgent:

    # In this class, we will implement functions to do the following.
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
    #       (a) Epsilon Greedy Policy.
    #     (b) Greedy Policy.
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.

    def __init__(self, env, gamma=0.99):
        # Create an instance of the network itself, as well as the memory.
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc.
        self.env = env
        self.gamma = gamma
        self.ns = env.observation_space.shape[0]
        self.na = env.action_space.n
        self.net = QNetwork(self.ns, self.na)

    def save_agent(self, name):
        p.dump((self.env, self.gamma, self.ns, self.na, strftime('%Y-%m-%d %H:%M:%S', localtime(int(time())))),
               open(name + '.p', 'wb'))
        self.net.save_model(name + '.h5')

    def epsilon_greedy_policy(self, q_values, eps):
        k = random.uniform(0, 1)
        if k <= eps:
            # print("random")
            a = random.randint(0, self.na - 1)
        else:
            # print("best")
            a = np.argmax(q_values)
        return a

    def greedy_policy(self, q_values):
        return np.argmax(q_values)

    def train(self, iteration_number):
        # In this function, we will train our network.
        # If training without experience replay_memory, then you will interact with the environment
        # in this function, while also updating your network parameters.

        # If you are using a replay memory, you should interact with environment here, and store these
        # transitions to memory, while also updating your model.
        iteration = 0
        episodes = 0
        states = []
        while True:
            # self.net.update_target_weights()
            s = self.env.reset()
            last = iteration
            train_set = []
            while True:
                eps = max(0.1 - 0.05 * iteration / 100000, 0.05)
                # print("epsilon is {}".format(eps))
                q_values = self.net.qvalues(s)
                # print("q values are {}".format(q_values))
                action = self.epsilon_greedy_policy(q_values, eps)
                # print("action taken is {}".format(action))
                s_, r, done, info = self.env.step(action)
                # print("taking {} on state {} gives reward {} and next state {}".format(action, s, r, s_))
                # print("prediction of next state is {}".format(self.net.qvalues(s_)))
                if not done:
                    q_values[action] = r + self.gamma * max(self.net.qvalues(s_))
                else:
                    q_values[action] = r
                train_set.append((s, q_values))
                # print("object q values are {}".format(q_values))
                # self.net.train(s, q_values)
                s = s_
                iteration += 1
                if done:
                    print("hold for {} times".format(iteration - last))
                    break
            x_train = [m[0] for m in train_set]
            y_train = [m[1] for m in train_set]
            self.net.train(x_train, y_train)
            episodes += 1
            if iteration > iteration_number:
                break
            # check reward
            if episodes % 100 == 0:
                print("The {}th episodes and the {}th iteration".format(episodes, iteration))
                states.append((episodes, self.test()))
                p.dump(states, open('states.p', 'wb'))
        print("The {}th episodes and the {}th iteration".format(episodes, iteration))
        self.test()

    def test(self, episodes=100, render=False):
        # Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using a memory.
        rewards = 0
        for _ in range(episodes):
            s = self.env.reset()
            while True:
                if render:
                    self.env.render()
                s, r, done, _ = self.env.step(self.greedy_policy(self.net.qvalues(s)))
                # s, r, done, _ = self.env.step(self.epsilon_greedy_policy(self.net.qvalues(s), 0.05))
                rewards += r
                if done:
                    break
        print("The average reward of {} episodes is {}".format(episodes, rewards / episodes))
        return rewards / episodes

    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of episodes / transitions.
        pass


def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env', dest='env', type=str)
    parser.add_argument('--render', dest='render', type=int, default=0)
    parser.add_argument('--train', dest='train', type=int, default=1)
    parser.add_argument('--model', dest='model_file', type=str)
    return parser.parse_args()


def main(args):
    # args = parse_arguments()
    # CartPole-v0
    env = gym.make("CartPole-v0")
    agent = DQNAgent(env, 0.99)
    agent.train(iteration_number=1000000)
    agent.save_agent("cartpole_linear_model")
    # agent = load_agent("cartpole_linear_model")
    # agent.test(5, True)


# You want to create an instance of the DQN_Agent class here, and then train / test it.

if __name__ == '__main__':
    main(sys.argv)
