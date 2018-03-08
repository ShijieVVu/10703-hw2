import gym
import numpy as np

from random import random, randint, sample
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

class Memory:
    def __init__(self, memory_size=50000, burn_in=10000):
        self.memory = []
        self.memory_size = memory_size
        # burn in
        while len(self.memory) <= burn_in:
            s = env.reset()
            while True:
                a = env.action_space.sample()
                s_, r, done, _ = env.step(a)
                if done:
                    s_ = None
                self.memory.append((s, a, r, s_, done))
                if done:
                    break
        self.index = burn_in - 1

    def remember(self, c):
        if len(self.memory) < self.memory_size:
            self.memory.append(c)
            self.index = (self.index + 1) % self.memory_size
        elif len(self.memory) == self.memory_size:
            self.memory[self.index] = c
            self.index = (self.index + 1) % self.memory_size
        else:
            print("Wrong")

    def sample(self, batch_size=32):
        return sample(self.memory, batch_size)


memory = Memory()

model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))

gamma = 0.99
eps = 0.5
episodes = 100
for j in range(10000):
    for e in range(episodes):
        s = env.reset()
        # mini_batch = []
        i = 0
        while True:
            q_values = model.predict(np.array([s]))[0]
            # print("q_values is {}".format(q_values))
            if random() <= eps:
                a = randint(0, na - 1)
            else:
                a = np.argmax(q_values)
            s_, r, done, _ = env.step(a)
            # mini_batch.append((s, a, r, s_, done))
            memory.remember((s, a, r, s_, done))
            s = s_
            i += 1
            if done:
                # print("hold for {} sec".format(i))
                break
        eps *= 0.99997
        mini_batch = memory.sample()
        x_train = np.zeros((len(mini_batch), ns))
        y_train = np.zeros((len(mini_batch), na))
        for i, (s1, a1, r1, s_1, done) in enumerate(mini_batch):
            q_values1 = model.predict(np.array([s1]))[0]
            if done:
                q_values1[a1] = r1
            else:
                q_values1[a1] = r1 + gamma * np.max(model.predict(np.array([s_1])))
            x_train[i] = s1
            y_train[i] = q_values1

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
