#!/usr/bin/env python
from random import random, randint, sample

import argparse
import gym
import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from pickle import dump


class QNetwork:

    def __init__(self, ns, na, identifier, learning_rate):
        self.model = None
        # linear model
        if identifier == "CartPole_q1" or "CartPole_q2":
            self.model = Sequential([
                Dense(na, input_shape=(ns,))
            ])
            self.model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate))

        # MLP
        if identifier == "CartPole_q3":
            self.model = Sequential([
                Dense(30, input_shape=(ns,), activation='relu'),
                Dense(30, input_shape=(30,), activation='relu'),
                Dense(30, input_shape=(30,), activation='relu'),
                Dense(30, input_shape=(30,), activation='relu'),
                Dense(na, input_shape=(30,))
            ])
            self.model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate))

    def save_model(self, name, iteration):
        self.model.save('./model/{}_{}.h5'.format(name, iteration))
        print('model saved to ./model/{}_{}.h5 on {} iteration'.format(name, iteration, iteration))

    def train(self, x_train, y_train, batch_size):
        self.model.fit(x_train, y_train, batch_size=batch_size, verbose=0)

    def qvalues(self, s):
        return self.model.predict(s)


class Memory:
    def __init__(self, environment_name, memory_size=50000, burn_in=10000):
        self.memory = []
        self.length = 0
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.full = False
        # burn in
        env = gym.make(environment_name)
        iteration = 0
        while iteration <= burn_in:
            s = env.reset()
            while True:
                a = env.action_space.sample()
                s_, r, done, _ = env.step(a)
                self.remember((s, a, r, s_, done))
                s = s_
                iteration += 1
                if done:
                    break
        print("Memory burned in with current index at {}".format(self.length))
        print("Memory size is {}".format(self.memory_size))

    def remember(self, transition):
        if self.full:
            self.memory[self.length] = transition
        else:
            self.memory.append(transition)
        self.length = (self.length + 1) % self.memory_size
        if self.length == 0:
            self.full = True

    def sample(self, batch_size=32):
        return sample(self.memory, batch_size)


class DQN_Agent:

    def __init__(self, environment_name, identifier, learning_rate, use_replay_memory):
        self.identifier = identifier
        self.env_name = environment_name
        self.env = gym.make(self.env_name)
        self.na = self.env.action_space.n
        self.ns = self.env.observation_space.shape[0]
        self.net = QNetwork(self.ns, self.na, identifier, learning_rate)
        if use_replay_memory:
            self.memory = Memory(environment_name)
        self.use_replay = use_replay_memory

    def epsilon_greedy_policy(self, q_values, eps):
        if random() <= eps:
            return randint(0, self.na - 1)
        else:
            return np.random.choice(np.flatnonzero(q_values == q_values.max()))

    def greedy_policy(self, q_values):
        pass

    def train(self, max_iteration, eps, eps_decay, eps_min, interval_iteration, gamma, test_size):
        iteration = 0
        performance = []
        while iteration <= max_iteration:
            while iteration <= max_iteration:
                start = iteration
                s = self.env.reset()
                if not self.use_replay:
                    mini_batch = []
                while True:
                    eps = max(eps - eps_decay * iteration, eps_min)
                    q_values = self.net.qvalues(np.array([s]))
                    a = self.epsilon_greedy_policy(q_values, eps)
                    s_, r, done, _ = self.env.step(a)
                    if not self.use_replay:
                        mini_batch.append((s, a, r, s_, done))
                    else:
                        mini_batch = self.memory.sample()
                        self.memory.remember((s, a, r, s_, done))

                        p = self.net.qvalues(np.array([i[0] for i in mini_batch]))
                        p_ = self.net.qvalues(
                            np.array([(i[3] if i[4] is not None else np.zeros(self.ns)) for i in mini_batch]))

                        x = np.zeros((len(mini_batch), self.ns))
                        y = np.zeros((len(mini_batch), self.na))

                        for i, val in enumerate(mini_batch):
                            s1 = val[0]
                            a1 = val[1]
                            r1 = val[2]
                            done1 = val[4]

                            if done1:
                                p[i][a1] = r1
                            else:
                                p[i][a1] = r1 + gamma * np.max(p_[i])

                            x[i] = s1
                            y[i] = p[i]
                        self.net.train(x, y, len(mini_batch))

                    s = s_
                    iteration += 1
                    # test
                    if iteration % interval_iteration == 0:
                        performance.append((iteration, self.test(iteration, test_size=test_size)))
                        break
                    # save model
                    if iteration % int(max_iteration / 3) == 0:
                        self.net.save_model(self.identifier, iteration)
                        break
                    if done:
                        # print("hold for {} sec".format(i - start))
                        break

                if not self.use_replay:
                    x_train = np.zeros((len(mini_batch), self.ns))
                    y_train = np.zeros((len(mini_batch), self.na))
                    for i1, (s1, a1, r1, s_1, done) in enumerate(mini_batch):
                        q_values1 = self.net.qvalues(np.array([s1]))
                        if done:
                            q_values1[a1] = r1
                        else:
                            q_values1[a1] = r1 + gamma * np.max(self.net.qvalues(np.array([s_1])))
                        x_train[i1] = s1
                        y_train[i1] = q_values1
                    self.net.train(x_train, y_train, len(mini_batch))
        dump(performance, open('./model/{}.p'.format(self.identifier), 'wb'))

    def test(self, iteration, test_size):
        rewards = 0
        for _ in range(test_size):
            s2 = self.env.reset()
            while True:
                q_values = self.net.qvalues(np.array([s2]))
                s2, r2, done2, _ = self.env.step(np.argmax(q_values))
                rewards += r2
                if done2:
                    break
        print("The average reward of {} iteration is {}".format(iteration, rewards / test_size))
        return rewards / test_size


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


def main(env_name, identifier, max_iteration, epsilon, epsilon_decay, epsilon_min, interval_iteration, gamma,
         test_size, learning_rate, use_replay_memory):
    agent = DQN_Agent(environment_name=env_name, identifier=identifier, learning_rate=learning_rate, use_replay_memory=use_replay_memory)
    agent.train(max_iteration=max_iteration, eps=epsilon, eps_decay=epsilon_decay,
                eps_min=epsilon_min, interval_iteration=interval_iteration, gamma=gamma, test_size=test_size)
