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
    Dense(na, input_shape=(ns,), use_bias=False)
])

model.set_weights([np.array([[0, 0, 0], [1, 2, 3]])])

test_size = 200

rewards = 0
x = []
y = []
for _ in range(test_size):
    s = env.reset()
    print("initial state is {} and velocity is {}".format(s[0], s[1]))
    x.append(s[0])
    i = 0
    while True:
        # env.render()
        q_values = model.predict(np.array([s]))
        s, r, done, _ = env.step(np.argmax(q_values == q_values.max()))
        rewards += r
        i += 1
        if done:
            print("hold for {} secs".format(i))
            break
    y.append(i)
print("The average reward of {} episodes is {}".format(test_size, rewards / test_size))
plt.scatter(x, y)
plt.show()
