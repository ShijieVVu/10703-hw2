from src import DQN_implementation as dqn

identifier = "MountainCar_q1"

# start training
dqn.main(env_name="MountainCar-v0", identifier=identifier, model_name="linear", max_iteration=1000000, epsilon=0.1, epsilon_decay=0.09e-6,
         epsilon_min=0.01, interval_iteration=10000, gamma=1, test_size=20, learning_rate=0.0015,
         use_replay_memory=False, memory_size=None, burn_in=None)
