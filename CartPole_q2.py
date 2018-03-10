from src import DQN_implementation as dqn
from os import remove
from glob import glob

identifier = "CartPole_q2"

# start training
# dqn.main(env_name="CartPole-v0", identifier=identifier, max_iteration=1000000, epsilon=1.0, epsilon_decay=9.5e-6,
#          epsilon_min=0.05, interval_iteration=10000, gamma=0.99, test_size=20, learning_rate=0.0001,
#          use_replay_memory=True, memory_size=100000, burn_in=20000)

dqn.main(env_name="CartPole-v0", identifier=identifier, max_iteration=1000000, epsilon=0.5, epsilon_decay=4.5e-6,
         epsilon_min=0.05, interval_iteration=10000, gamma=0.99, test_size=20, learning_rate=0.002,
         use_replay_memory=False, memory_size=50000, burn_in=10000)