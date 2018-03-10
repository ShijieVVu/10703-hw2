#!/usr/bin/env python
from random import random, randint, sample

import argparse
import gym
import numpy as np
from keras import Sequential, Model, Input
from keras.layers import Dense, merge
from keras.optimizers import Adam
from pickle import dump
from keras import backend as K


class QNetwork:

    def __init__(self, ns, na, identifier, learning_rate):
        self.model = None
        # linear model
        if identifier == "CartPole_q1" or identifier == "CartPole_q2" or identifier == "MountainCar_q1" or identifier == "MountainCar_q2":
            print("using linear model")
            self.model = Sequential([
                Dense(na, input_shape=(ns,))
            ])
            self.model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate))

        # MLP
        if identifier == "CartPole_q3" or identifier == "MountainCar_q3":
            print("using MLP model")
            self.model = Sequential([
                Dense(30, input_shape=(ns,), activation='relu'),
                Dense(30, input_shape=(30,), activation='relu'),
                Dense(30, input_shape=(30,), activation='relu'),
                Dense(30, input_shape=(30,), activation='relu'),
                Dense(na, input_shape=(30,))
            ])
            self.model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate))

        # Duel DQN
        if identifier == "CartPole_q4" or identifier == "MountainCar_q4":
            print("using dual DQN model")
            input = Input(shape=(ns,))
            x = Dense(30, activation='relu')(input)
            x = Dense(30, activation='relu')(x)
            val_fc = Dense(30)(x)
            val = Dense(1)(val_fc)
            advantage_fc = Dense(30)(x)
            advantage = Dense(na)(advantage_fc)
            predictions = merge([val, advantage], mode=lambda y: y[0] + y[1] - K.mean(y[1]), output_shape=(na,))
            self.model = Model(input, predictions)
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

    def __init__(self, environment_name, identifier, learning_rate, use_replay_memory, memory_size, burn_in):
        self.identifier = identifier
        self.env_name = environment_name
        self.env = gym.make(self.env_name)
        self.na = self.env.action_space.n
        self.ns = self.env.observation_space.shape[0]
        self.net = QNetwork(self.ns, self.na, identifier, learning_rate)
        if use_replay_memory:
            self.memory = Memory(environment_name, memory_size, burn_in)
        self.use_replay = use_replay_memory

    def epsilon_greedy_policy(self, q_values, eps):
        if random() <= eps:
            return randint(0, self.na - 1)
        else:
            # break ties randomly
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
                        q_values1 = self.net.qvalues(np.array([s1]))[0]
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


def run_dqn(env_name, identifier, max_iteration, epsilon, epsilon_decay, epsilon_min, interval_iteration, gamma,
            test_size, learning_rate, use_replay_memory, memory_size, burn_in):
    agent = DQN_Agent(environment_name=env_name, identifier=identifier, learning_rate=learning_rate,
                      use_replay_memory=use_replay_memory, memory_size=memory_size, burn_in=burn_in)
    agent.train(max_iteration=max_iteration, eps=epsilon, eps_decay=epsilon_decay,
                eps_min=epsilon_min, interval_iteration=interval_iteration, gamma=gamma, test_size=test_size, )


# QUESTION 1
# CartPole-v0 q1
identifier = "CartPole_q1"
run_dqn(env_name="CartPole-v0", identifier=identifier, max_iteration=1000000, epsilon=0.5, epsilon_decay=4.5e-6,
        epsilon_min=0.05, interval_iteration=10000, gamma=0.99, test_size=20, learning_rate=0.002,
        use_replay_memory=False, memory_size=None, burn_in=None)

# CartPole-v0 Linear Explore
# See github link https://github.com/ShijieVVu/10703-hw2.git

# MountainCar-v0 q1
identifier = "MountainCar_q1"
run_dqn(env_name="MountainCar-v0", identifier=identifier, max_iteration=1000000, epsilon=0.1, epsilon_decay=0.09e-6,
        epsilon_min=0.01, interval_iteration=10000, gamma=1, test_size=20, learning_rate=0.0015,
        use_replay_memory=False, memory_size=None, burn_in=None)

# MountainCar-v0 Linear Design
# See github link https://github.com/ShijieVVu/10703-hw2.git


# QUESTION 2
# CartPole-v0 q2
identifier = "CartPole_q2"
run_dqn(env_name="CartPole-v0", identifier=identifier, max_iteration=1000000, epsilon=0.5, epsilon_decay=4.5e-6,
        epsilon_min=0.05, interval_iteration=10000, gamma=0.99, test_size=20, learning_rate=0.002,
        use_replay_memory=True, memory_size=50000, burn_in=10000)

# MountainCar-v0 q2
identifier = "MountainCar_q2"
run_dqn(env_name="MountainCar-v0", identifier=identifier, max_iteration=1000000, epsilon=0.1, epsilon_decay=0.09e-6,
        epsilon_min=0.01, interval_iteration=10000, gamma=1, test_size=20, learning_rate=0.0015,
        use_replay_memory=True, memory_size=50000, burn_in=10000)

# QUESTION 3
# CartPole-v0 q3
identifier = "CartPole_q3"
run_dqn(env_name="CartPole-v0", identifier=identifier, max_iteration=100000, epsilon=0.5, epsilon_decay=4.5e-4,
        epsilon_min=0.05, interval_iteration=1000, gamma=0.99, test_size=20, learning_rate=0.0001,
        use_replay_memory=True, memory_size=50000, burn_in=10000)

# MountainCar-v0 q3
identifier = "MountainCar_q3"
run_dqn(env_name="MountainCar-v0", identifier=identifier, max_iteration=1000000, epsilon=0.25, epsilon_decay=4.5e-6,
        epsilon_min=0.05, interval_iteration=10000, gamma=1, test_size=20, learning_rate=0.0001,
        use_replay_memory=True, memory_size=50000, burn_in=10000)

# QUESTION 4
# CartPole-v0 q4
identifier = "CartPole_q4"
run_dqn(env_name="CartPole-v0", identifier=identifier, max_iteration=20000, epsilon=1.0, epsilon_decay=9.5e-4,
        epsilon_min=0.05, interval_iteration=1000, gamma=0.99, test_size=20, learning_rate=0.0002,
        use_replay_memory=True, memory_size=50000, burn_in=10000)

# MountainCar-v0 q4
identifier = "MountainCar_q4"
run_dqn(env_name="MountainCar-v0", identifier=identifier, max_iteration=1000000, epsilon=0.25, epsilon_decay=4.5e-6,
        epsilon_min=0.01, interval_iteration=10000, gamma=1, test_size=20, learning_rate=0.0001,
        use_replay_memory=True, memory_size=50000, burn_in=10000)
