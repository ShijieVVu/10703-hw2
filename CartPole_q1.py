from random import random, randint

import gym
import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

env = gym.make('CartPole-v0')
ns = env.observation_space.shape[0]
na = env.action_space.n

model = Sequential([
    Dense(na, input_shape=(ns,))
])

model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001))

gamma = 0.99
eps = 0.5
max_iteration = 10000
interval_iteration = 1000
i = 0
test_size = 20

while i <= max_iteration:
    while i <= max_iteration:
        s = env.reset()
        mini_batch = []
        while True:
            eps = max(eps - 4.5e-6 * i, 0.05)
            q_values = model.predict(np.array([s]))[0]
            start = 0
            if random() <= eps:
                a = randint(0, na - 1)
            else:
                a = np.argmax(q_values)
            s_, r, done, _ = env.step(a)
            mini_batch.append((s, a, r, s_, done))
            s = s_
            i += 1
            # test
            if i % interval_iteration == 0:
                rewards = 0
                for _ in range(test_size):
                    s = env.reset()
                    while True:
                        q_values = model.predict(np.array([s]))
                        s, r, done, _ = env.step(np.argmax(q_values))
                        rewards += r
                        if done:
                            break
                print("The average reward of {} iteration is {}".format(i, rewards / test_size))
            if done:
                # print("hold for {} sec".format(i - start))
                break
        x_train = np.zeros((len(mini_batch), ns))
        y_train = np.zeros((len(mini_batch), na))
        for i1, (s1, a1, r1, s_1, done) in enumerate(mini_batch):
            q_values1 = model.predict(np.array([s1]))[0]
            if done:
                q_values1[a1] = r1
            else:
                q_values1[a1] = r1 + gamma * np.max(model.predict(np.array([s_1])))
            x_train[i1] = s1
            y_train[i1] = q_values1

        model.fit(x_train, y_train, verbose=0)