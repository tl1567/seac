import torch
import robotic_warehouse
import lbforaging
import gym

from a2c import A2C
from wrappers import RecordEpisodeStatistics, TimeLimit

import time

import numpy as np
import pandas as pd

from cbs_rware import Environment, CBS

from absl import app
from absl import flags

FLAGS = flags.FLAGS

# flags.DEFINE_string("path", "pretrained/rware-small-4ag", "path of the model file")
flags.DEFINE_string("env_name", "rware-tiny-2ag-v1", "env name")
flags.DEFINE_integer("time_limit", 500, "maximum number of timesteps for each episode")
flags.DEFINE_integer("seed", 1, "seed")
# path = "pretrained/rware-small-4ag"
# env_name = "rware-tiny-2ag-v1"
# time_limit = 500 # 25 for LBF

_AXIS_Z = 0
_AXIS_Y = 1
_AXIS_X = 2

_COLLISION_LAYERS = 2

_LAYER_AGENTS = 0
_LAYER_SHELFS = 1


def shelf_ids_coordinates(env, shelf_list):
        """
        Compute the shelf ids and their coordinates
        """
        ids = [shelf.id for shelf in shelf_list]
        coordinates = \
            [np.concatenate(np.where(env.grid[_LAYER_SHELFS] == shelf_id)).tolist() for shelf_id in ids]
        return ids, coordinates

## Shelf requesting the closest agent to pick it up
def compute_dist_agents_targets(agents_loc, targets_loc):
    dist = [[abs(agents_loc[i][0] - targets_loc[j][0]) \
        + abs(agents_loc[i][1] - targets_loc[j][1]) for j in range(len(targets_loc))] \
            for i in range(len(agents_loc))]
    return dist    


def compute_dist_argmins(dist):
    dist_argmins = [np.argmin(dist[j]) for j in range(len(dist))]
    return dist_argmins


## Split targets for agents: agents carrying shelves should reach the closest goal locations; 
## agents not carrying shelves should reach the closest uncarried requested shelves.
def decompose_agents_targets(warehouse):
        agents_id = [warehouse.grid[_LAYER_AGENTS, agent.y, agent.x] for agent in warehouse.agents]
        # agents_loc = [[agent.y.item(), agent.x.item()] for agent in warehouse.agents]
        agents_loc = [[agent.y, agent.x] for agent in warehouse.agents]
        # print(agents_loc)
        goals_loc = [list(goal) for goal in warehouse.goals]
        goals_loc = [[goal[1], goal[0]] for goal in goals_loc]


        # requested_shelves_loc = [[shelf.y.item(), shelf.x.item()] for shelf in request_queue]

        carried_shelves = [agent.carrying_shelf for agent in warehouse.agents]
        carried_requested_shelves = list(set(carried_shelves) & set(warehouse.request_queue))
        uncarried_requested_shelves = list(set(warehouse.request_queue) - set(carried_requested_shelves))

        # _, carried_requested_shelf_loc = shelf_ids_coordinates(warehouse, carried_requested_shelf)
        _, uncarried_requested_shelves_loc = shelf_ids_coordinates(warehouse, uncarried_requested_shelves)

        ## for agents carrying shelves, their targets are the closest goal locations
        agents_carrying_shelves_id = \
            [warehouse.grid[_LAYER_AGENTS, agent.y, agent.x] for agent in warehouse.agents if agent.carrying_shelf]
        agents_carrying_shelves_loc = \
            [[agent.y, agent.x] for agent in warehouse.agents if agent.carrying_shelf]
            # [[agent.y.item(), agent.x.item()] for agent in warehouse.agents if agent.carrying_shelf]
            

        agents_not_carrying_shelves_id = \
            [warehouse.grid[_LAYER_AGENTS, agent.y, agent.x] for agent in warehouse.agents if not agent.carrying_shelf]
        agents_not_carrying_shelves_loc = \
            [[agent.y, agent.x] for agent in warehouse.agents if not agent.carrying_shelf]
            # [[agent.y.item(), agent.x.item()] for agent in warehouse.agents if not agent.carrying_shelf]
            

        # print('Agents carrying shelves:', agents_carrying_shelves_loc)
        # print('Agents not carrying shelves:', agents_not_carrying_shelves_loc)
        # print('Goals:', goals_loc)
        # print('Uncarried requested shelves:', uncarried_requested_shelves_loc)
        return agents_carrying_shelves_id, agents_carrying_shelves_loc, agents_not_carrying_shelves_id, \
            agents_not_carrying_shelves_loc, goals_loc, uncarried_requested_shelves_loc

def assign_target_to_agent(agents, agents_loc, agents_id, targets):
        dist = compute_dist_agents_targets(agents_loc, targets)
        dist = np.array(dist)
        ind = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
        agents.append({'start': agents_loc[ind[0]], 'goal': targets[ind[1]], 'name': f'agent{agents_id[ind[0]]}'})
        del agents_loc[ind[0]]
        del agents_id[ind[0]]
        del targets[ind[1]]
        return agents, agents_loc, agents_id, targets


def cbs_planning(warehouse):
    dimension = list(warehouse.grid_size)
    obstacles = []

    '''
    agents_id = [warehouse.grid[_LAYER_AGENTS, agent.y, agent.x] for agent in warehouse.agents]
    agents_loc = [[agent.y.item(), agent.x.item()] for agent in warehouse.agents]

    goals_loc = [list(goal) for goal in warehouse.goals]
    requested_shelves_loc = [[shelf.y.item(), shelf.x.item()] for shelf in warehouse.request_queue]    

    carried_shelf = [agent.carrying_shelf for agent in warehouse.agents]
    carried_requested_shelf = list(set(carried_shelf) & set(warehouse.request_queue))
    uncarried_requested_shelf = list(set(warehouse.request_queue) - set(carried_requested_shelf))

    carried_requested_shelves_ids, carried_requested_shelf_loc = shelf_ids_coordinates(warehouse, carried_requested_shelf)
    uncarried_requested_shelves_ids, uncarried_requested_shelf_loc = shelf_ids_coordinates(warehouse, uncarried_requested_shelf)
    '''

    # requested_shelves_loc = [[shelf.y.item(), shelf.x.item()] for shelf in warehouse.request_queue]
    requested_shelves_loc = [[shelf.y, shelf.x] for shelf in warehouse.request_queue]

    agents_carrying_shelves_id, agents_carrying_shelves_loc, agents_not_carrying_shelves_id, \
        agents_not_carrying_shelves_loc, goals_loc, uncarried_requested_shelves_loc = decompose_agents_targets(warehouse)

    ## solve recursively the requested shelf of each agent (the agent with the minimum distance to a goal will be assigned with that goal;
    ## both the agent and the goal will be removed from the queues, and this is done recursively)
    # dist = compute_dist_agents_targets(agents_not_carrying_shelves_loc, uncarried_requested_shelves_loc)    
    # if len(set(np.argmin(dist, axis=1))) == len(np.argmin(dist, axis=1)):
    #     uncarried_requested_shelves_loc = [uncarried_requested_shelves_loc[i] for i in dist_argmins]
    #     names = [f'agent{i+1}' for i in range(len(agents_not_carrying_shelves_loc))]
    #     agents = [{'start': agents_not_carrying_shelves_loc[i], 'goal': uncarried_requested_shelf_loc[i], \
    #         'name': names[i]} for i in range(len(agents_not_carrying_shelves_loc))]
    # else:         
    agents = []
    while len(agents_not_carrying_shelves_loc):
        agents, agents_not_carrying_shelves_loc, agents_not_carrying_shelves_id, uncarried_requested_shelves_loc = \
            assign_target_to_agent(agents, agents_not_carrying_shelves_loc, agents_not_carrying_shelves_id, uncarried_requested_shelves_loc)

    '''
    if len(agents_carrying_shelves_loc):
        dist_agents_carrying_shelves_goals = compute_dist_agents_targets(agents_carrying_shelves_loc, goals_loc)
        dist_agents_carrying_shelves_goals = np.array(dist_agents_carrying_shelves_goals)
        ind = np.argmin(dist_agents_carrying_shelves_goals, axis=1)
        names_agents_carrying_shelves = [f'agent{agents_carrying_shelves_id[i]}' for i in range(len(agents_carrying_shelves_id))]
        # goals_agents_carrying_shelves = [goals_loc[i] for i in ind]
        goals_agents_carrying_shelves = goals_loc
        agents += [{'start': agents_carrying_shelves_loc[i], 'goal': goals_agents_carrying_shelves[i], \
            'name': names_agents_carrying_shelves[i]} for i in range(len(agents_carrying_shelves_loc))]
    '''

    while len(agents_carrying_shelves_loc):
        agents, agents_carrying_shelves_loc, agents_carrying_shelves_id, goals_loc = \
            assign_target_to_agent(agents, agents_carrying_shelves_loc, agents_carrying_shelves_id, goals_loc)
        
    
    ## Add obstacles for agents carrying shelves
    nearby_carrying_shelves_loc = []
    for i in range(len(agents_carrying_shelves_loc)):
        nearby_carrying_shelves_loc.append([agents_carrying_shelves_loc[i][0]-1, agents_carrying_shelves_loc[i][1]])
        nearby_carrying_shelves_loc.append([agents_carrying_shelves_loc[i][0], agents_carrying_shelves_loc[i][1]-1])
        nearby_carrying_shelves_loc.append([agents_carrying_shelves_loc[i][0]+1, agents_carrying_shelves_loc[i][1]])
        nearby_carrying_shelves_loc.append([agents_carrying_shelves_loc[i][0], agents_carrying_shelves_loc[i][1]+1])
    # print('Nearby carrying shelves:', nearby_carrying_shelves_loc)    
    
    
    _, shelves_loc = shelf_ids_coordinates(warehouse, warehouse.shelfs)
    # print('Shelves:', shelves_loc)
    # for shelf_loc in nearby_carrying_shelves_loc:
    for shelf_loc in shelves_loc:
        if shelf_loc not in agents_carrying_shelves_loc and shelf_loc not in agents_not_carrying_shelves_loc \
            and shelf_loc not in requested_shelves_loc:
        # if shelf_loc in nearby_carrying_shelves_loc and shelf_loc not in requested_shelves_loc:
        # if shelf_loc not in requested_shelves_loc:
        # if shelf_loc in shelves_loc:
            obstacles.append(tuple(shelf_loc))
        
    # print('Agents:', agents)
    # print('Obstacles:', obstacles)
    

    env = Environment(dimension, agents, obstacles)  ## Environment from MAPP 

    ## Searching
    cbs = CBS(env)
    solution = cbs.search()
    if not solution:
        print("Conflict-based search (CBS) (re)planning cannot find any solution!")
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
    # elif plan[t+1]['x'] == plan[t]['x'] and plan[t+1]['y'] == plan[t]['y']:
    #     Action = 5
    else:
        Action = 0  ## no movement 
        

    if Action == Direction[-1]:
        action = [1]
        direction = [Direction[-1]]
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
        direction = [Direction[-1]]
 
    return action, direction


def plan_to_actions(directions, plan):
    ## direction and plan of all agents
    actions = {f'agent{i+1}': [] for i in range(len(plan))}
    # directions = init_directions
    # directions = {f'agent{i+1}': [directions[f'agent{i+1}'][-1]] for i in range(len(plan))}
    for i in range(len(plan)):
        # print(f'direction of agent{i+1}:', directions[f'agent{i+1}'])
        # print(f'plan of agent{i+1}:', plan[f'agent{i+1}'])
        # print(directions[f'agent{i+1}'])
        for t in range(len(plan[f'agent{i+1}'])-1):
            actions[f'agent{i+1}'] += get_action(directions[f'agent{i+1}'], plan[f'agent{i+1}'], t)[0]
            directions[f'agent{i+1}'] += get_action(directions[f'agent{i+1}'], plan[f'agent{i+1}'], t)[1]

    # print(actions)
    # print(directions)
    return actions, directions


def actions_from_replan(directions_dict, plan):
    actions_from_plan_dict, directions_dict = plan_to_actions(directions_dict, plan)
    # actions_from_plan_dict, directions_dict = plan_to_actions(init_directions_dict, plan)

    max_len_actions = max([len(v) for v in actions_from_plan_dict.values()])
    min_len_actions = min([len(v) for v in actions_from_plan_dict.values()])
    for i in range(len(actions_from_plan_dict)):
        if len(actions_from_plan_dict[f'agent{i+1}']) == min_len_actions:
            actions_from_plan_dict[f'agent{i+1}'].append(4)
            # actions_from_plan_dict[f'agent{i+1}'] += [4, 0]
        else:
            actions_from_plan_dict[f'agent{i+1}'][:] = actions_from_plan_dict[f'agent{i+1}'][0:min_len_actions]
            actions_from_plan_dict[f'agent{i+1}'].append(0)
        # while len(actions_from_plan_dict[f'agent{i+1}']) < max_len_actions + 3:
        #     actions_from_plan_dict[f'agent{i+1}'].append(0)
    
    # print('plan:', plan)
    for i in range(len(directions_dict)):
        if len(directions_dict[f'agent{i+1}']) == min_len_actions + 1:
            directions_dict[f'agent{i+1}'].append(directions_dict[f'agent{i+1}'][-1])
        else: 
            directions_dict[f'agent{i+1}'][:] + directions_dict[f'agent{i+1}'][0:min_len_actions+1]
            directions_dict[f'agent{i+1}'].append(directions_dict[f'agent{i+1}'][-1])
        # while len(directions_dict[f'agent{i+1}']) < max_len_actions + 4:
        #     directions_dict[f'agent{i+1}'].append(directions_dict[f'agent{i+1}'][-1])

    # min_len_actions = min([len(v) for v in actions_from_plan_dict.values()])

    for i in range(len(actions_from_plan_dict)):
        actions_from_plan_dict[f'agent{i+1}'][:] = actions_from_plan_dict[f'agent{i+1}'][0:min_len_actions+1]
    for i in range(len(directions_dict)):
        directions_dict[f'agent{i+1}'][:] = directions_dict[f'agent{i+1}'][1:min_len_actions+2]

    return actions_from_plan_dict, directions_dict

def main(_):
    # path = FLAGS.path
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

    # for agent in agents:
    #     agent.restore(path + f"/agent{agent.agent_id}")

    
    obs = env.reset()
    ## up, down, left, right = 1, 2, 3, 4
    # init_directions = {f'agent{i+1}': [int(np.where(obs[i][3:7] == 1)[0] + 1)] for i in range(len(obs))}  
    init_directions_dict = {f'agent{i+1}': [int(np.where(obs[i][3:7] == 1)[0] + 1)] for i in range(len(obs))}
    # plan = cbs_planning(env)
    # actions_from_plan, directions = plan_to_actions(init_directions, plan)

    # min_len_actions = min([len(v) for v in actions_from_plan.values()])
    # max_len_actions = max([len(v) for v in actions_from_plan.values()])
    # for i in range(len(actions_from_plan)):
    #     actions_from_plan[f'agent{i+1}'].append(4)
    #     while len(actions_from_plan[f'agent{i+1}']) < max_len_actions + 2:
    #         actions_from_plan[f'agent{i+1}'] += [0]

    # for i in range(len(directions)):        
    #     while len(directions[f'agent{i+1}']) < max_len_actions + 3:
    #         directions[f'agent{i+1}'].append(directions[f'agent{i+1}'][-1])

    # actions_from_plan = [actions_from_plan[f'agent{i+1}'] for i in range(len(actions_from_plan))]
    # directions = [directions[f'agent{i+1}'] for i in range(len(directions))]
    
    plan = cbs_planning(env)
    actions_from_plan_dict, directions_dict = actions_from_replan(init_directions_dict, plan)
    # print(actions_from_plan_dict)
    # print(directions_dict)
    actions_from_plan = [actions_from_plan_dict[f'agent{i+1}'] for i in range(len(actions_from_plan_dict))]
    directions = [directions_dict[f'agent{i+1}'] for i in range(len(directions_dict))]

    # print('Actions from plan:', actions_from_plan)
    # print("Directions from plan:", directions)
    for i in range(RUN_STEPS):
    # total_steps = 0
    # while total_steps < RUN_STEPS:
    # for i in range(max_len_actions + 2):
        # obs = [torch.from_numpy(o) for o in obs]
        
        # _, actions, _ , _ = zip(*[agent.model.act(obs[agent.agent_id], None, None) for agent in agents])
        # actions = [a.item() for a in actions]

        # actions = [actions_from_plan[k][i] for k in range(len(actions_from_plan))]

        ## replanning as soon as one agent picks up a shelf or delivers a shelf
        # if i == len(actions_from_plan[0])-1:
        # if i > 0:
        # for i in range(len(actions_from_plan[0])):
        # print('Actions from plan:', actions_from_plan)
        # if any(actions_from_plan[k][-1] == 4 for k in range(len(actions_from_plan))):
        # print('Actions from plan:', actions_from_plan)
        
            

        actions = [actions_from_plan[k][i] for k in range(len(actions_from_plan))]
        
        # print('Actions:', actions)
        env.render()

        time.sleep(1)

        obs, _, done, info = env.step(actions)
        
        # print([directions[k][i] for k in range(len(directions))])
        # print({f'agent{i+1}': [int(np.where(obs[i][3:7] == 1)[0] + 1)] for i in range(len(obs))})

        min_len_actions = min([len(actions_from_plan[k]) for k in range(len(actions_from_plan))])
        if i == min_len_actions - 1:
            init_directions_dict = {f'agent{i+1}': [int(np.where(obs[i][3:7] == 1)[0] + 1)] for i in range(len(obs))}
            plan = cbs_planning(env)
            # init_directions_dict = {f'agent{k+1}': [directions_dict[f'agent{k+1}'][-1]] for k in range(len(directions_dict))}
            # init_directions_dict = {f'agent{i+1}': [int(np.where(obs[i][3:7] == 1)[0] + 1)] for i in range(len(obs))}
            actions_from_plan_dict, directions_dict = actions_from_replan(init_directions_dict, plan)
            # print(actions_from_plan_dict)
            # print(directions_dict)
            for k in range(len(actions_from_plan)):
                actions_from_plan[k] += actions_from_plan_dict[f'agent{k+1}']
                directions[k] += directions_dict[f'agent{k+1}']
                # print('length', len(actions_from_plan[k]))

            # print('Actions from plan:', actions_from_plan)
            # print("Directions from plan:", directions)


        
        # actions_from_plan = [actions_from_plan_dict[f'agent{i+1}'] for i in range(len(actions_from_plan_dict))]
        # directions = [directions_dict[f'agent{i+1}'] for i in range(len(directions_dict))]
        
        

        # print([agent.carrying_shelf for agent in env.agents])
        # print([agent.has_delivered for agent in env.agents])
       
        
        if all(done):
            obs = env.reset()
            print("--- Episode Finished ---")
            print(f"Episode rewards: {sum(info['episode_reward'])}")
            print(info)
            print(" --- ")

    

    pd.DataFrame(actions_from_plan).to_csv(f'./results/CBS/actions_{env_name}_seed{seed}.csv', index=False, header=False)

if __name__ == "__main__":
    app.run(main)
