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
    _AXIS_Z = 0
    _AXIS_Y = 1
    _AXIS_X = 2

    _COLLISION_LAYERS = 2

    _LAYER_AGENTS = 0
    _LAYER_SHELFS = 1


    dimension = list(warehouse.grid_size)
    obstacles = []
    agents_id = [warehouse.grid[_LAYER_AGENTS, agent.y, agent.x] for agent in warehouse.agents]
    agents_loc = [[agent.y.item(), agent.x.item()] for agent in warehouse.agents]
    goals = [[shelf.y.item(), shelf.x.item()] for shelf in warehouse.request_queue]
    # print('Goals:', goals)

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
    def assign_goal_to_agent(agents_loc, agents_id, goals):
        dist = compute_dist_agents_goals(agents_loc, goals)
        dist = np.array(dist)
        ind = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
        agents.append({'start': agents_loc[ind[0]], 'goal': goals[ind[1]], 'name': f'agent{agents_id[ind[0]]}'})
        del agents_loc[ind[0]]
        del agents_id[ind[0]]
        del goals[ind[1]]
        return agents_loc, agents_id, goals

    
    if len(set(np.argmin(dist, axis=1))) == len(np.argmin(dist, axis=1)):
        goals = [goals[i] for i in dist_argmins]
        names = [f'agent{i+1}' for i in range(warehouse.n_agents)]
        agents = [{'start': agents_loc[i], 'goal': goals[i], 'name': names[i]} for i in range(len(agents_loc))]
    else: 
        agents = []
        while len(agents_loc):
            agents_loc, agents_id, goals = assign_goal_to_agent(agents_loc, agents_id, goals)


    env = Environment(dimension, agents, obstacles)  ## Environment from MAPP 

    ## Searching
    cbs = CBS(env)
    solution = cbs.search()
    if not solution:
        print("Conflict-based search (CBS) planning cannot find any solution!")
        return

    return solution

def get_action(Direction, plan, t): 
    ## direction and plan of one agent   

    if plan[t+1]['x'] - plan[t]['x'] == 1:
        Action = 4  ## move right        
    elif plan[t+1]['x'] - plan[t]['x'] == -1:
        Action = 3  ## move left        
    elif plan[t+1]['y'] - plan[t]['y'] == 1:
        Action = 2  ## move down        
    elif plan[t+1]['y'] - plan[t]['y'] == -1:
        Action = 1  ## move up        
    else:
        Action = 0  ## no movement 
        

    if Action == Direction[-1]:
        action = [1]
        direction = [Action]
    elif Action == 1:
        if Direction[-1] == 2:
            action = [3, 3, 1]
            direction = [3, 1, 1]
        elif Direction[-1] == 3:
            action = [3, 1]
            direction = [1, 1]
        elif Direction[-1] == 4:
            action = [2, 1]
            direction = [1, 1]
    elif Action == 2:
        if Direction[-1] == 1:
            action = [3, 3, 1]
            direction = [4, 2, 2]
        elif Direction[-1] == 3:
            action = [2, 1]
            direction = [2, 2]
        elif Direction[-1] == 4:
            action = [3, 1]
            direction = [2, 2]
    elif Action == 3:
        if Direction[-1] == 1:
            action = [2, 1]
            direction = [3, 3]
        elif Direction[-1] == 2:
            action = [3, 1]
            direction = [3, 3]
        elif Direction[-1] == 4:
            action = [3, 3, 1]
            direction = [2, 3, 3]
    elif Action == 4:
        if Direction[-1] == 1:
            action = [3, 1]
            direction = [4, 4]
        elif Direction[-1] == 2:
            action = [2, 1]
            direction = [4, 4]
        elif Direction[-1] == 3:
            action = [3, 3, 1]
            direction = [1, 4, 4]
    elif Action == 0: 
        action = [0]
        direction = Direction[-1]
 
    return action, direction


def plan_to_actions(init_directions, plan):
    ## direction and plan of all agents
    actions = {f'agent{i+1}': [] for i in range(len(plan))}
    directions = init_directions
    for i in range(len(plan)):
        for t in range(len(plan[f'agent{i+1}'])-1):
            actions[f'agent{i+1}'] += get_action(directions[f'agent{i+1}'], plan[f'agent{i+1}'], t)[0]
            directions[f'agent{i+1}'] += get_action(directions[f'agent{i+1}'], plan[f'agent{i+1}'], t)[1]

    return actions, directions


def main(_):
    # path = FLAGS.path
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
    ## up, down, left, right = 1, 2, 3, 4
    init_directions = {f'agent{i+1}': [int(np.where(obs[i][3:7] == 1)[0] + 1)] for i in range(len(obs))}  
    plan = cbs_planning(env)
    actions_from_plan, directions = plan_to_actions(init_directions, plan)


    max_len_actions = max([len(v) for v in actions_from_plan.values()])
    for i in range(len(actions_from_plan)):
        actions_from_plan[f'agent{i+1}'].append(4)
        while len(actions_from_plan[f'agent{i+1}']) < max_len_actions + 2:
            actions_from_plan[f'agent{i+1}'] += [0]

    for i in range(len(directions)):        
        while len(directions[f'agent{i+1}']) < max_len_actions + 3:
            directions[f'agent{i+1}'].append(directions[f'agent{i+1}'][-1])

    actions_from_plan = [actions_from_plan[f'agent{i+1}'] for i in range(len(actions_from_plan))]
    directions = [directions[f'agent{i+1}'] for i in range(len(directions))]
    

    # for i in range(RUN_STEPS):
    for i in range(max_len_actions + 2):
        # obs = [torch.from_numpy(o) for o in obs]
        # print('Observation:', obs[0].numpy())
        # _, actions, _ , _ = zip(*[agent.model.act(obs[agent.agent_id], None, None) for agent in agents])
        # actions = [a.item() for a in actions]
        actions = [actions_from_plan[j][i] for j in range(len(actions_from_plan))]
        # print('Actions:', actions)
        env.render()

        if i < max_len_actions + 1:
            time.sleep(1)
        else:
            time.sleep(5)
        
        obs, _, done, info = env.step(actions)
        # if all(done):
        #     obs = env.reset()
        #     print("--- Episode Finished ---")
        #     print(f"Episode rewards: {sum(info['episode_reward'])}")
        #     print(info)
        #     print(" --- ")

if __name__ == "__main__":
    app.run(main)
