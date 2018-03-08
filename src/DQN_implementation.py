#!/usr/bin/env python
from random import random, randint

import argparse
import sys
import gym
import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from pickle import dump


class QNetwork():

    def __init__(self, ns, na):
        self.model = Sequential([
            Dense(na, input_shape=(ns,))
        ])

    def save_model(self, name, iteration):
        self.model.save('./model/{}_{}.h5'.format(name, iteration))
        print('model saved on {} iteration'.format(iteration))

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train, verbose=0)

    def qvalues(self, s):
        return self.model.predict(np.array([s]))[0]


class Replay_Memory():

    def __init__(self, memory_size=50000, burn_in=10000):
        pass

    def sample_batch(self, batch_size=32):
        pass

    def append(self, transition):
        pass


class DQN_Agent():

    def __init__(self, environment_name, identifier=None):
        self.identifier = identifier
        self.net = QNetwork(environment_name, identifier)
        self.env_name = environment_name
        self.env = gym.make(self.env_name)
        self.na = self.env.action_space.n
        self.ns = self.env.observation_space.shape[0]

    def epsilon_greedy_policy(self, q_values, eps):
        if random() <= eps:
            return randint(0, self.na - 1)
        else:
            return np.random.choice(np.flatnonzero(q_values == q_values.max()))

    def greedy_policy(self, q_values):
        pass

    def train(self, max_iteration=None, eps=None, eps_decay=None, eps_min=None, interval_iteration=None, gamma=None,
              test_size=None):
        iteration = 0
        performance = []
        while iteration <= max_iteration:
            while iteration <= max_iteration:
                s = self.env.reset()
                mini_batch = []
                while True:
                    eps = max(eps - eps_decay * iteration, eps_min)
                    q_values = self.net.qvalues(s)
                    a = self.epsilon_greedy_policy(q_values, eps)
                    s_, r, done, _ = self.env.step(a)
                    mini_batch.append((s, a, r, s_, done))
                    s = s_
                    iteration += 1
                    # test
                    if iteration % interval_iteration == 0:
                        performance.append((iteration, self.test(iteration, test_size=test_size)))
                        done = True
                    # save model
                    if iteration % int(max_iteration / 3) == 0:
                        self.net.save_model(self.identifier, iteration)
                    if done:
                        # print("hold for {} sec".format(i - start))
                        break
                x_train = np.zeros((len(mini_batch), self.ns))
                y_train = np.zeros((len(mini_batch), self.na))
                for i1, (s1, a1, r1, s_1, done) in enumerate(mini_batch):
                    q_values1 = self.net.qvalues(s1)
                    if done:
                        q_values1[a1] = r1
                    else:
                        q_values1[a1] = r1 + gamma * np.max(self.net.qvalues(s_1))
                    x_train[i1] = s1
                    y_train[i1] = q_values1

                self.net.train(x_train, y_train)
        return performance

    def test(self, iteration, test_size=None):
        env = gym.make(self.env_name)
        rewards = 0
        for _ in range(test_size):
            s2 = env.reset()
            while True:
                q_values = self.net.qvalues(s2)
                s2, r2, done2, _ = env.step(np.argmax(q_values))
                rewards += r2
                if done2:
                    break
        print("The average reward of {} iteration is {}".format(iteration, rewards / test_size))

    def burn_in_memory(self):
        pass


def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env', dest='env', type=str)
    parser.add_argument('--render', dest='render', type=int, default=0)
    parser.add_argument('--train', dest='train', type=int, default=1)
    parser.add_argument('--model', dest='model_file', type=str)
    parser.add_argument('--gamma', dest='gamma', type=float, default=0.99)
    parser.add_argument('--epsilon', dest='epsilon', type=float, default=0.5)
    parser.add_argument('--epsilon_decay', dest='epsilon_decay', type=float, default=4.5e-6)
    parser.add_argument('--epsilon_min', dest='epsilon_min', type=float, default=0.05)
    parser.add_argument('--max_iteration', dest='max_iteration', type=int, default=1000000)
    parser.add_argument('--interval_iteration', dest='interval_iteration', type=int, default=10000)
    parser.add_argument('--test_size', dest='test_size', type=int, default=20)
    parser.add_argument('--identifier', dest='identifier', type=str, default=None)
    return parser.parse_args()


def main(args):
    args = parse_arguments()
    agent = DQN_Agent(args.env, args.identifier)
    agent.train(max_iteration=args.max_iteration, eps=args.epsilon, eps_decay=args.epsilon_decay,
                eps_min=args.epsilon_min, interval_iteration=args.interval_iteration, test_size=args.test_size)


if __name__ == '__main__':
    main(sys.argv)
