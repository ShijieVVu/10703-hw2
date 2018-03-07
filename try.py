import gym
import numpy as np

from random import random, randint
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

env = gym.make('CartPole-v0')
ns = env.observation_space.shape[0]
na = env.action_space.n
# model = Sequential([
#     Dense(na, input_shape=(ns,))
# ])

model = Sequential([
    Dense(30, input_shape=(ns,), activation='relu'),
    Dense(30, input_shape=(30,), activation='relu'),
    Dense(na, input_shape=(30,), activation='linear')
])


model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))

gamma = 0.99
eps = 0.5
episodes = 100
for j in range(10000):
    for e in range(episodes):
        s = env.reset()
        mini_batch = []
        i = 0
        while True:
            q_values = model.predict(np.array([s]))[0]
            # print("q_values is {}".format(q_values))
            if random() <= eps:
                a = randint(0, na - 1)
            else:
                a = np.argmax(q_values)
            s_, r, done, _ = env.step(a)
            if not done:
                q_values[a] = r + gamma * np.max(model.predict(np.array([s_])))
            else:
                q_values[a] = r
            mini_batch.append((s, q_values))
            s = s_
            i += 1
            if done:
                # print("hold for {} sec".format(i))
                break
        eps *= 0.99997
        x_train = np.array([m[0] for m in mini_batch])
        y_train = np.array([m[1] for m in mini_batch])
        model.fit(x_train, y_train, verbose=0)

    rewards = 0
    for _ in range(20):
        s = env.reset()
        while True:
            # env.render()
            q_values = model.predict(np.array([s]))
            s, r, done, _ = env.step(np.argmax(q_values))
            rewards += r
            if done:
                break
    print("The average reward of {} episodes is {}".format(episodes, rewards / 20))

    if rewards / 20 >= 195:
        print("env solved after {} episodes".format(j * 100))
        model.save('tryout.h5')
        break
