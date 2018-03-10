from matplotlib import pylab as plt
from pickle import load

name = 'CartPole_q1'

data = load(open('./model/{}.p'.format(name), 'rb'))

iteration = [c[0] for c in data]
reward = [c[1] for c in data]
plt.title(name)
plt.xlabel('iteration')
plt.ylabel('average reward')
plt.plot(iteration, reward)
plt.show()
