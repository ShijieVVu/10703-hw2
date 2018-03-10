from src import DQN_implementation as dqn
from os import remove
from glob import glob

identifier = "MountainCar_q3"

# # remove previous files
# for f in glob("./model/{}*.*".format(identifier)):
#     remove(f)

# start training
dqn.main(env_name="MountainCar-v0", identifier=identifier, max_iteration=1000000, epsilon=0.1, epsilon_decay=0.09e-6,
         epsilon_min=0.01, interval_iteration=10000, gamma=0.99, test_size=20, learning_rate=0.0015, use_replay_memory=True)