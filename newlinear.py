#!/usr/bin/env python
import argparse
import pickle as p
import random
import sys
from time import strftime, localtime, time

import gym
import numpy as np
from keras import optimizers, initializers
from keras.layers import Dense, Input
from keras.models import Model, load_model, Sequential


class QNetwork:

    # This class essentially defines the network architecture.
    # The network should take in state of the world as an input,
    # and output Q values of the actions available to the agent as the output.

    def __init__(self, ns, na):
        # Define your network architecture here. It is also a good idea to define any training operations
        # and optimizers here, initialize your variables, or alternately compile your model here.
        inputs = Input(shape=(ns,))
        predictions = Dense(na, activation="linear")(inputs)
        self.model = Model(inputs=inputs, outputs=predictions)
        adam = optimizers.Adam(lr=0.0001)
        self.model.compile(loss='mse', optimizer=adam)

    def train(self, s, target):
        self.model.fit(s, target, batch_size=32, verbose=0)

    def qvalues(self, s):
        return self.model.predict(s)

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

    def __init__(self, env, gamma=1.0):
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
            a = random.randint(0, self.env.action_space.n - 1)
        else:
            a = np.argmax(q_values)
        return a

    def greedy_policy(self, q_values):
        return np.argmax(q_values)

    def train(self, case, steps, interval):
        # In this function, we will train our network.
        # If training without experience replay_memory, then you will interact with the environment
        # in this function, while also updating your network parameters.

        # If you are using a replay memory, you should interact with environment here, and store these
        # transitions to memory, while also updating your model.
        iteration = 0
        episodes = 0
        while iteration < 100000:
            last = iteration
            s = self.env.reset()
            while iteration < 100000:
                eps = max(0.5 - (iteration / 100000) * 0.45, 0.05)
                q_values = self.net.qvalues(np.array([s]))
                action = self.epsilon_greedy_policy(q_values, eps)
                s_, r, done, info = self.env.step(action)
                if done:
                    s_ = None
                trans = (s, action, r, s_)

                batch = [trans]

                p = self.net.qvalues(np.array([i[0] for i in batch]))
                p_ = self.net.qvalues(np.array([(i[3] if i[3] is not None else np.zeros(self.ns)) for i in batch]))

                x = np.zeros((len(batch), self.ns))
                y = np.zeros((len(batch), self.na))

                for i, val in enumerate(batch):
                    s1 = val[0]
                    a1 = val[1]
                    r1 = val[2]
                    s_1 = val[3]

                    if s_1 is None:
                        p[i][a1] = r1
                    else:
                        p[i][a1] = r1 + self.gamma * np.max(p_[i])

                    x[i] = s1
                    y[i] = p[i]
                self.net.train(x, y)
                s = s_

                iteration += 1
                if iteration - last >= 200 or done:
                    break
            episodes += 1
            # if episodes > steps:
            #     break
            if episodes % interval == 0:
                print("The {}th episodes".format(episodes))
                self.test()
        self.test()

    def test(self, episodes=20, render=False):
        # Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using a memory.
        rewards = 0
        for _ in range(episodes):
            s = self.env.reset()
            steps = 0
            while True:
                if render:
                    self.env.render()
                s, r, done, _ = self.env.step(self.greedy_policy(self.net.qvalues(np.array([s]))))
                # s, r, done, _ = self.env.step(self.epsilon_greedy_policy(self.net.qvalues(s), 0.05))
                rewards += r
                steps += 1
                if done:
                    break
                if steps >= 200:
                    break
        print("The average reward of {} episodes is {}".format(episodes, rewards / episodes))
        return rewards / episodes


def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env', dest='env', type=str)
    parser.add_argument('--render', dest='render', type=int, default=0)
    parser.add_argument('--train', dest='train', type=int, default=1)
    parser.add_argument('--model', dest='model_file', type=str)
    return parser.parse_args()


def main(args):
    # args = parse_arguments()
    # MountainCar-v0
    # Cartpole-v0
    env = gym.make("CartPole-v0")
    agent = DQNAgent(env, gamma=0.99)
    agent.train(case='eps', steps=10, interval=100)
    # agent.save_agent("Cartpole-v0-dqn")

    # agent = load_agent("mountaincar_linear_model")
    # agent.test(5, True)


# You want to create an instance of the DQN_Agent class here, and then train / test it.

if __name__ == '__main__':
    main(sys.argv)
