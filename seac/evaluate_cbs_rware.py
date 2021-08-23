import torch
import robotic_warehouse
import lbforaging
import gym

from a2c import A2C
from wrappers import RecordEpisodeStatistics, TimeLimit

import time

import numpy as np

from cbs_rware import Environment, CBS

from absl import app
from absl import flags

FLAGS = flags.FLAGS

# flags.DEFINE_string("path", "pretrained/rware-small-4ag", "path of the model file")
flags.DEFINE_string("env_name", "rware-tiny-2ag-v1", "env name")
flags.DEFINE_integer("time_limit", 500, "maximum number of timesteps for each episode")
# path = "pretrained/rware-small-4ag"
# env_name = "rware-tiny-2ag-v1"
# time_limit = 500 # 25 for LBF


def cbs_planning(warehouse):
    dimension = list(warehouse.grid_size)
    obstacles = []
    agents_loc = [[agent.y.item(), agent.x.item()] for agent in warehouse.agents]
    goals = [[shelf.y.item(), shelf.x.item()] for shelf in warehouse.request_queue]

    ## Shelf requesting the closest agent to pick it up
    def compute_dist_agents_goals(agents_loc, goals):
        dist = [[abs(agents_loc[i][0] - goals[j][0]) + abs(agents_loc[i][1] - goals[j][1]) for j in range(len(goals))] for i in range(len(agents_loc))]
        return dist

    dist = compute_dist_agents_goals(agents_loc, goals)

    def compute_dist_argmins(dist):
        dist_argmins = [np.argmin(dist[j]) for j in range(len(goals))]
        return dist_argmins

    dist_argmins = compute_dist_argmins(dist)

    ## solve recursively the goal of each agent (the agent with the minimum distance to a goal will be assigned with that goal;
    ## both the agent and the goal will be removed from the queues, and this is done recursively)
    def assign_goal_to_agent(agents_loc, goals):
        dist = compute_dist_agents_goals(agents_loc, goals)
        dist = np.array(dist)
        ind = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
        agents.append({'start': agents_loc[ind[0]], 'goal': goals[ind[1]], 'name': f'agent{len(agents)}'})
        del agents_loc[ind[0]]
        del goals[ind[1]]
        return agents_loc, goals

    
    if len(set(np.argmin(dist, axis=1))) == len(np.argmin(dist, axis=1)):
        goals = [goals[i] for i in dist_argmins]
        names = [f'agent{i}' for i in range(warehouse.n_agents)]
        agents = [{'start': agents_loc[i], 'goal': goals[i], 'name': names[i]} for i in range(len(agents_loc))]
    else: 
        agents = []
        while len(agents_loc):
            agents_loc, goals = assign_goal_to_agent(agents_loc, goals)


    env = Environment(dimension, agents, obstacles)  ## Environment from MAPP 

    ## Searching
    cbs = CBS(env)
    solution = cbs.search()
    if not solution:
        print("Solution not found" )
        return
    # cost = env.compute_solution_cost(solution)

    return solution


def plan_to_actions(plan):
    for i in range(len(plan)):
        print(plan[f'agent{i}'])

    actions = {f'agent{i}': , for i in range(len(plan))}
    return actions


def main(_):
    path = FLAGS.path
    env_name = FLAGS.env_name
    time_limit = FLAGS.time_limit

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

    # for agent in agents:
    #     agent.restore(path + f"/agent{agent.agent_id}")

    obs = env.reset()

    plan = cbs_planning(env)

    for i in range(RUN_STEPS):
        obs = [torch.from_numpy(o) for o in obs]
        print('Observation:', obs)
        _, actions, _ , _ = zip(*[agent.model.act(obs[agent.agent_id], None, None) for agent in agents])
        actions = [a.item() for a in actions]
        print('Actions:', actions)
        time.sleep(1)
        env.render()
        obs, _, done, info = env.step(actions)
        if all(done):
            obs = env.reset()
            print("--- Episode Finished ---")
            print(f"Episode rewards: {sum(info['episode_reward'])}")
            print(info)
            print(" --- ")

if __name__ == "__main__":
    app.run(main)
