import numpy as np
import pandas as pd 

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "rware-small-5ag-v1", "env name")
flags.DEFINE_string("alg", "mapf", "mapf or marl")
flags.DEFINE_integer("seed0", 0, "seed 0")
flags.DEFINE_integer("seed1", 1, "seed 1")
flags.DEFINE_integer("seed2", 2, "seed 2")
flags.DEFINE_integer("seed3", 3, "seed 3")
flags.DEFINE_integer("seed4", 4, "seed 4")


def main(_):
    # actions_paths = [f'./results/CBS/{FLAGS.env_name}/actions_{FLAGS.env_name}_seed{i}_episode{j}.csv' \
    #     for i in range(4) for j in [500, 1000, 1500, 2000]]

    # rewards_paths = [f'./results/CBS/{FLAGS.env_name}/rewards_{FLAGS.env_name}_seed{i}_episode{j}.csv' \
    #     for i in range(4) for j in [500, 1000, 1500, 2000]]
    
    # time_paths = [f'./results/CBS/{FLAGS.env_name}/time_{FLAGS.env_name}_seed{i}_episode{j}.csv' \
    #     for i in range(4) for j in [500, 1000, 1500, 2000]]

    seeds = [FLAGS.seed0, FLAGS.seed1, FLAGS.seed2, FLAGS.seed3, FLAGS.seed4]

    if FLAGS.alg == "mapf":
        actions_paths = [f'./results/CBS/{FLAGS.env_name}/actions_{FLAGS.env_name}_seed{i}_episode{j}.csv' \
            for i in seeds for j in [500, 1000, 1500, 2000]]

        rewards_paths = [f'./results/CBS/{FLAGS.env_name}/rewards_{FLAGS.env_name}_seed{i}_episode{j}.csv' \
            for i in seeds for j in [500, 1000, 1500, 2000]]
        
        time_paths = [f'./results/CBS/{FLAGS.env_name}/time_{FLAGS.env_name}_seed{i}_episode{j}.csv' \
            for i in seeds for j in [500, 1000, 1500, 2000]]

    elif FLAGS.alg == "marl":
        actions_paths = [f'./results/SEAC/{FLAGS.env_name}/actions_{FLAGS.env_name}_seed{i}_episode{j}.csv' \
            for i in seeds for j in [500, 1000, 1500, 2000]]

        rewards_paths = [f'./results/SEAC/{FLAGS.env_name}/rewards_{FLAGS.env_name}_seed{i}_episode{j}.csv' \
            for i in seeds for j in [500, 1000, 1500, 2000]]
        
        time_paths = [f'./results/SEAC/{FLAGS.env_name}/time_{FLAGS.env_name}_seed{i}_episode{j}.csv' \
            for i in seeds for j in [500, 1000, 1500, 2000]]

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
    def getDeliveryTime(x: np.ndarray):
        time = []
        for i in range(x.shape[0]):
            t = [j for j, v in enumerate(x[i,:]) if v == 4][1] if np.count_nonzero(x[i,:] == 4) >= 2 else 0
            time.append(t)
        return time

    # the sum of delivery times of all agents at their delivery locations
    flowtime = np.array([])
    for action in actions:
        flowtime = np.append(flowtime, sum(getDeliveryTime(np.array(action))))

    # the maximum of the delivery times of all agents at their delivery locations
    makespan = np.array([])
    for action in actions:
        makespan = np.append(makespan, max(getDeliveryTime(np.array(action))))


    mean_reward = np.mean(rewards)
    mean_time = np.mean(time)
    mean_num_delivered = np.mean(num_delivered)
    mean_flowtime = np.mean(flowtime)
    mean_makespan = np.mean(makespan)

    std_reward = np.std(rewards, ddof=1)
    std_time = np.std(time, ddof=1)
    std_num_delivered = np.std(num_delivered, ddof=1)
    std_flowtime = np.std(flowtime, ddof=1)
    std_makespan = np.std(makespan, ddof=1)
    

    print(f'Mean reward per agent per episode (s.d.): {mean_reward:.2f} ({std_reward:.2f})')
    print(f'Mean episode time (s.d.): {mean_time:.2f} ({std_time:.2f})')
    print(f'Mean number of delivered items per agent per episode (s.d.): {mean_num_delivered:.2f} ({std_num_delivered:.2f})')
    print(f'Mean flowtime per episode (s.d.): {mean_flowtime:.2f} ({std_flowtime:.2f})')
    print(f'Mean makespan per episode (s.d.): {mean_makespan:.2f} ({std_makespan:.2f})')

if __name__ == "__main__":
    app.run(main)