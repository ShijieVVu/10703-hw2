from random import random, randint, sample

import gym
import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# class env1(type(gym.make('MountainCar-v0'))):
#     def step(self, action):
#         state, reward, done, _ = super().step(action)
#         velocity = state[1]
#         if int(action) == 0:
#             reward -= velocity * 10
#         elif int(action) == 2:
#             reward += velocity * 10
#         return state, reward, done, {}
#
#
# class env2(type(gym.make('MountainCar-v0'))):
#     def reset(self):
#         state = super().reset()
#         self.max_state1 = 0
#         return state
#
#     def step(self, action):
#         state, reward, done, _ = super().step(action)
#         if abs(state[1]) > abs(self.max_state1):
#             reward += 10
#         return state, reward, done, {}
#
#
# class env3(type(gym.make('MountainCar-v0'))):
#     def step(self, action):
#         state, reward, done, _ = super().step(action)
#         height = state[0]
#         reward += 100 * height
#         return state, reward, done, {}


env_ = gym.make("MountainCar-v0").env
env = gym.make('MountainCar-v0')
ns = env_.observation_space.shape[0]
na = env_.action_space.n

model = Sequential([
    Dense(na, input_shape=(ns,))
])

# model = Sequential([
#     Dense(30, input_shape=(ns,), activation='relu'),
#     Dense(30, input_shape=(30,), activation='relu'),
#     Dense(na, input_shape=(30,), activation='linear')
# ])


# print("Model weights {}".format(model.get_weights()))


class Memory:
    def __init__(self, memory_size=50000, burn_in=10000):
        self.memory = []
        self.memory_size = memory_size
        # burn in
        while len(self.memory) <= burn_in:
            s = env_.reset()
            while True:
                a = env_.action_space.sample()
                s_, r, done, _ = env_.step(a)
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


# memory = Memory()

model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001))

gamma = 1.0
eps = 0.5
epochs = 5000
test_size = 5
render = False
solvable = -110
i = 0
for j in range(10000):
    for e in range(epochs):
        s = env_.reset()
        mini_batch = []
        while True:
            # env_.render()
            eps = max(eps - 4.5e-6 * i, 0.05)
            q_values = model.predict(np.array([s]))[0]
            # print("q_values is {}".format(q_values))
            if random() <= eps:
                a = randint(0, na - 1)
            else:
                a = np.random.choice(np.flatnonzero(q_values == q_values.max()))
            s_, r, done, _ = env_.step(a)
            if done:
                q_values[a] = r
            else:
                q_values[a] = r + gamma * np.max(model.predict(np.array([s_])))
            model.fit(np.array([s]), np.array([q_values]), verbose=0)
            # mini_batch.append((s, a, r, s_, done))
            # memory.remember((s, a, r, s_, done))
            s = s_
            i += 1
            if done:
                # print("hold for {} sec".format(i))
                break
        # mini_batch = memory.sample()
        # x_train = np.zeros((len(mini_batch), ns))
        # y_train = np.zeros((len(mini_batch), na))
        # for i, (s1, a1, r1, s_1, done) in enumerate(mini_batch):
        #     q_values1 = model.predict(np.array([s1]))[0]
        #     if done:
        #         q_values1[a1] = r1
        #     else:
        #         q_values1[a1] = r1 + gamma * np.max(model.predict(np.array([s_1])))
        #     x_train[i] = s1
        #     y_train[i] = q_values1

        # model.fit(x_train, y_train, verbose=0)

    print("Model weights {}".format(model.get_weights()))
    rewards = 0
    for _ in range(test_size):
        s = env.reset()
        while True:
            if render:
                env.render()
            q_values = model.predict(np.array([s]))
            s, r, done, _ = env.step(np.random.choice(np.flatnonzero(q_values == q_values.max())))
            rewards += r
            if done:
                break
    print("The average reward of {} episodes is {}".format(epochs, rewards / test_size))

    if rewards / test_size >= solvable:
        print("env solved after {} episodes".format(j * epochs))
        model.save('tryout.h5')
        break
