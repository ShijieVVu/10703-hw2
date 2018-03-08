from random import random, randint

import gym
import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from pickle import dump

env = gym.make('CartPole-v0')
ns = env.observation_space.shape[0]
na = env.action_space.n

model = Sequential([
    Dense(na, input_shape=(ns,))
])

model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))

gamma = 0.99
eps = 0.5
eps_decay = 4.5e-6
eps_min = 0.05
max_iteration = 1000000
interval_iteration = 10000
test_size = 20

performance = []
iteration = 0
while iteration <= max_iteration:
    while iteration <= max_iteration:
        s = env.reset()
        mini_batch = []
        while True:
            eps = max(eps - eps_decay * iteration, eps_min)
            q_values = model.predict(np.array([s]))[0]
            if random() <= eps:
                a = randint(0, na - 1)
            else:
                a = np.argmax(q_values)
            s_, r, done, _ = env.step(a)
            mini_batch.append((s, a, r, s_, done))
            s = s_
            iteration += 1
            # test
            if iteration % interval_iteration == 0:
                rewards = 0
                for _ in range(test_size):
                    s2 = env.reset()
                    while True:
                        q_values = model.predict(np.array([s2]))
                        s2, r2, done2, _ = env.step(np.argmax(q_values))
                        rewards += r2
                        if done2:
                            break
                print("The average reward of {} iteration is {}".format(iteration, rewards / test_size))
                performance.append((iteration, rewards / test_size))
                done = True
            # save model
            if iteration % int(max_iteration / 3) == 0:
                model.save('./model/CartPole_q1_{}.h5'.format(iteration))
                print('model saved on {} iteration'.format(iteration))
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

dump(performance, open('./model/CartPole_q1_performance.p', 'wb'))
