from matplotlib import pylab as plt
from pickle import load

# name = 'CartPole_q4'
name = 'MountainCar_q4'

data = load(open('./model/{}.p'.format(name), 'rb'))

iteration = [c[0] for c in data]
reward = [c[1] for c in data]
plt.title(name)
plt.xlabel('iteration')
plt.ylabel('average reward')
plt.plot(iteration, reward)
if "CartPole" in name:
    plt.plot(iteration, [195] * len(iteration))
if "MountainCar" in name:
    plt.plot(iteration, [-110] * len(iteration))
plt.show()
