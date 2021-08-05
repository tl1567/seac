import torch
import robotic_warehouse
import lbforaging
import gym

from a2c import A2C
from wrappers import RecordEpisodeStatistics, TimeLimit

import time

path = '/home/timlautk/Desktop/2021 Zebra Research Intern/GitHub/seac/seac/results/sacred/13/u50000'
env_name = "rware-tiny-2ag-v1"
time_limit = 500 # 25 for LBF

RUN_STEPS = 2000

env = gym.make(env_name)
env = TimeLimit(env, time_limit)
env = RecordEpisodeStatistics(env)

device = "cpu"
# device = "cuda:0"

agents = [
    A2C(i, osp, asp, 0.1, 0.1, False, 1, 1, device)
    for i, (osp, asp) in enumerate(zip(env.observation_space, env.action_space))
]

for agent in agents:
    agent.restore(path + f"/agent{agent.agent_id}")

obs = env.reset()

for i in range(RUN_STEPS):
    obs = [torch.from_numpy(o) for o in obs]
    _, actions, _ , _ = zip(*[agent.model.act(obs[agent.agent_id], None, None) for agent in agents])
    actions = [a.item() for a in actions]
    time.sleep(1)
    env.render()
    obs, _, done, info = env.step(actions)
    if all(done):
        obs = env.reset()
        print("--- Episode Finished ---")
        print(f"Episode rewards: {sum(info['episode_reward'])}")
        print(info)
        print(" --- ")


