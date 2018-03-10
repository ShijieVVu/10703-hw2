from src import DQN_implementation as dqn

identifier = "MountainCar_q4"

# start training
dqn.main(env_name="MountainCar-v0", identifier=identifier, max_iteration=1000000, epsilon=0.25, epsilon_decay=4.5e-6,
         epsilon_min=0.01, interval_iteration=10000, gamma=1, test_size=20, learning_rate=0.0001,
         use_replay_memory=True, memory_size=50000, burn_in=10000)
