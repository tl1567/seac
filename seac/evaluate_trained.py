import torch
import robotic_warehouse
import lbforaging
import gym

from a2c import A2C
from wrappers import RecordEpisodeStatistics, TimeLimit

import time

import numpy as np
import pandas as pd

from absl import app
from absl import flags

import os

FLAGS = flags.FLAGS

flags.DEFINE_string("path", 'results/trained_models/95/u80000', "path of the model file")
flags.DEFINE_string("env_name", "rware-small-5ag-v1", "env name")
flags.DEFINE_integer("time_limit", 500, "maximum number of timesteps for each episode")
flags.DEFINE_integer("seed", 1, "seed")
# path = 'results/trained_models/002/u80000'
# env_name = "rware-tiny-2ag-v1"
# time_limit = 500 # 25 for LBF


def main(_):
    path = FLAGS.path
    env_name = FLAGS.env_name
    time_limit = FLAGS.time_limit
    seed = FLAGS.seed

    RUN_STEPS = 2000

    env = gym.make(env_name)
    env = TimeLimit(env, time_limit)
    env = RecordEpisodeStatistics(env)

    env.seed(seed)

    device = "cpu"
    # device = "cuda:0"

    agents = [
        A2C(i, osp, asp, 0.1, 0.1, False, 1, 1, device)
        for i, (osp, asp) in enumerate(zip(env.observation_space, env.action_space))
    ]

    for agent in agents:
        agent.restore(path + f"/agent{agent.agent_id}")

    obs = env.reset()

    actions_list = []
    for i in range(RUN_STEPS):        
        obs = [torch.from_numpy(o) for o in obs]
        _, actions, _ , _ = zip(*[agent.model.act(obs[agent.agent_id], None, None) for agent in agents])
        actions = [a.item() for a in actions]
        time.sleep(1)
        env.render()
        obs, _, done, info = env.step(actions)
        
        actions_list.append(actions)
        print(actions_list)

        if all(done):
            obs = env.reset()
            print("--- Episode Finished ---")
            print(f"Episode rewards: {sum(info['episode_reward'])}")
            print(info)
            print(" --- ")
            actions_list = np.transpose(np.array(actions_list))
            os.makedirs(os.path.dirname(f'./results/SEAC/{env_name}/'), exist_ok=True)
            pd.DataFrame(actions_list).to_csv(f'./results/SEAC/{env_name}/actions_{env_name}_seed{seed}_episode{i+1}.csv', index=False, header=False)
            pd.DataFrame(info['episode_reward']).to_csv(f'./results/SEAC/{env_name}/rewards_{env_name}_seed{seed}_episode{i+1}.csv', index=False, header=False)
            pd.DataFrame([info['episode_time']]).to_csv(f'./results/SEAC/{env_name}/time_{env_name}_seed{seed}_episode{i+1}.csv', index=False, header=False)

if __name__ == "__main__":
    app.run(main)
