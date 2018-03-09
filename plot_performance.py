from matplotlib import pylab as plt
from pickle import load

# name = 'CartPole_q1'
#
# data = load(open('./model/{}.p'.format(name), 'rb'))
#
# iteration = [c[0] for c in data]
# reward = [c[1] for c in data]
# plt.plot(iteration, reward)
# plt.show()

fig = plt.figure()
fig.suptitle('Linear Model without Experience Replay', fontsize=14)

ax = fig.add_subplot(112)
fig.subplots_adjust(top=0.85)
ax.set_title('axes title')

ax.set_xlabel('xlabel')
ax.set_ylabel('ylabel')

ax.plot([2], [1], 'o')
ax.annotate('annotate', xy=(2, 1), xytext=(3, 4),
            arrowprops=dict(facecolor='black', shrink=0.05))

ax.axis([0, 10, 0, 10])

ax = fig.add_subplot(212)
fig.subplots_adjust(top=0.85)
ax.set_title('axes title')

ax.set_xlabel('xlabel')
ax.set_ylabel('ylabel')

ax.plot([2], [1], 'o')
ax.annotate('annotate', xy=(2, 1), xytext=(3, 4),
            arrowprops=dict(facecolor='black', shrink=0.05))

ax.axis([5, 2, 5, 3])

plt.show()