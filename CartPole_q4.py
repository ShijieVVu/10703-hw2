from src import DQN_implementation as dqn

identifier = "CartPole_q4"

# start training
dqn.main(env_name="CartPole-v0", identifier=identifier, model_name="Duel DQN", max_iteration=20000, epsilon=1.0, epsilon_decay=9.5e-4,
         epsilon_min=0.05, interval_iteration=1000, gamma=0.99, test_size=20, learning_rate=0.0002,
         use_replay_memory=True, memory_size=50000, burn_in=10000)
