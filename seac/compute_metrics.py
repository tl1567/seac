import numpy as np
import pandas as pd 

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "rware-small-5ag-v1", "env name")
flags.DEFINE_string("alg", "planning", "planning or rl")



def main(_):
    # actions_paths = [f'./results/CBS/{FLAGS.env_name}/actions_{FLAGS.env_name}_seed{i}_episode{j}.csv' \
    #     for i in range(4) for j in [500, 1000, 1500, 2000]]

    # rewards_paths = [f'./results/CBS/{FLAGS.env_name}/rewards_{FLAGS.env_name}_seed{i}_episode{j}.csv' \
    #     for i in range(4) for j in [500, 1000, 1500, 2000]]
    
    # time_paths = [f'./results/CBS/{FLAGS.env_name}/time_{FLAGS.env_name}_seed{i}_episode{j}.csv' \
    #     for i in range(4) for j in [500, 1000, 1500, 2000]]

    if FLAGS.alg == "planning":
        actions_paths = [f'./results/CBS/{FLAGS.env_name}/actions_{FLAGS.env_name}_seed{i}_episode{j}.csv' \
            for i in range(5) for j in [500, 1000, 1500, 2000]]

        rewards_paths = [f'./results/CBS/{FLAGS.env_name}/rewards_{FLAGS.env_name}_seed{i}_episode{j}.csv' \
            for i in range(5) for j in [500, 1000, 1500, 2000]]
        
        time_paths = [f'./results/CBS/{FLAGS.env_name}/time_{FLAGS.env_name}_seed{i}_episode{j}.csv' \
            for i in range(5) for j in [500, 1000, 1500, 2000]]

    elif FLAGS.alg == "rl":
        actions_paths = [f'./results/SEAC/{FLAGS.env_name}/actions_{FLAGS.env_name}_seed{i}_episode{j}.csv' \
            for i in range(5) for j in [500, 1000, 1500, 2000]]

        rewards_paths = [f'./results/SEAC/{FLAGS.env_name}/rewards_{FLAGS.env_name}_seed{i}_episode{j}.csv' \
            for i in range(5) for j in [500, 1000, 1500, 2000]]
        
        time_paths = [f'./results/SEAC/{FLAGS.env_name}/time_{FLAGS.env_name}_seed{i}_episode{j}.csv' \
            for i in range(5) for j in [500, 1000, 1500, 2000]]

    actions = []
    rewards = np.array([])
    time = np.array([])

    for actions_path in actions_paths:
        df_actions = pd.read_csv(actions_path, header=None).to_numpy()
        actions.append(df_actions)
    
    for rewards_path in rewards_paths:
        df_rewards = np.mean(pd.read_csv(rewards_path, header=None).to_numpy())
        rewards = np.append(rewards, df_rewards)

    for time_path in time_paths:
        df_time = pd.read_csv(time_path, header=None).to_numpy()
        time = np.append(time, df_time)

    num_delivered = np.array([])
    for action in actions: 
        for i in range(action.shape[0]):
            num_delivered = np.append(num_delivered, np.count_nonzero(action[i] == 4) // 2)

    # success_rate = 


    mean_reward = np.mean(rewards)
    mean_time = np.mean(time)
    mean_num_delivered = np.mean(num_delivered)

    std_reward = np.std(rewards, ddof=1)
    std_time = np.std(time, ddof=1)
    std_num_delivered = np.std(num_delivered, ddof=1)

    print(f'Mean reward per agent per episode (s.d.): {mean_reward:.2f} ({std_reward:.2f})')
    print(f'Mean episode time (s.d.): {mean_time:.2f} ({std_time:.2f})')
    print(f'Mean number of delivered items per agent per episode (s.d.): {mean_num_delivered:.2f} ({std_num_delivered:.2f})')

if __name__ == "__main__":
    app.run(main)