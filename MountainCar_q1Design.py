# Fully accelerate velocity
from matplotlib import pyplot as plt
import gym

import numpy as np
from keras import Sequential
from keras.layers import Dense

env = gym.make('MountainCar-v0')

ns = env.observation_space.shape[0]
na = env.action_space.n

model = Sequential([
    Dense(na, input_shape=(ns,))
])

model.set_weights([np.array([[0, 0, 0], [1, 2, 3]]), np.array([0, 0, 0])])

test_size = 200

rewards = 0
x = []
y = []
for _ in range(test_size):
    s = env.reset()
    x.append(s[0])
    i = 0
    while True:
        # env.render()
        q_values = model.predict(np.array([s]))
        s, r, done, _ = env.step(np.argmax(q_values == q_values.max()))
        rewards += r
        i += 1
        if done:
            print("solved for {} iterations".format(i))
            break
    y.append(-i)
print("The average reward of {} episodes is {}".format(test_size, rewards / test_size))
plt.title("MountainCar Designed Linear Model")
plt.ylabel("Episode reward")
plt.xlabel("Initial location")
plt.scatter(x, y)
plt.show()
