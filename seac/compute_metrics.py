import numpy as np
import pandas as pd 

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "rware-small-5ag-v1", "env name")



def main(_):
    actions_paths = [f'./results/CBS/{FLAGS.env_name}/actions_{FLAGS.env_name}_seed{i}_episode{j}.csv' \
        for i in range(4) for j in [500, 1000, 1500, 2000]]

    rewards_paths = [f'./results/CBS/{FLAGS.env_name}/rewards_{FLAGS.env_name}_seed{i}_episode{j}.csv' \
        for i in range(4) for j in [500, 1000, 1500, 2000]]
    
    time_paths = [f'./results/CBS/{FLAGS.env_name}/time_{FLAGS.env_name}_seed{i}_episode{j}.csv' \
        for i in range(4) for j in [500, 1000, 1500, 2000]]

    actions = np.array([])
    rewards = np.array([])
    time = np.array([])    

    for actions_path in actions_paths:
        df_actions = pd.read_csv(actions_path, header=None).to_numpy()
        actions = np.append(actions, df_actions, axis=0)
    
    for rewards_path in rewards_paths:
        df_rewards = np.mean(pd.read_csv(rewards_path, header=None).to_numpy())
        rewards = np.append(rewards, df_rewards)

    for time_path in time_paths:
        df_time = pd.read_csv(time_path, header=None).to_numpy()
        time = np.append(time, df_time)

    print(actions)

    # success_rate = 


    mean_reward = np.mean(rewards)
    mean_time = np.mean(time)



    print('Mean reward:', mean_reward)
    print('Mean time', mean_time)

if __name__ == "__main__":
    app.run(main)