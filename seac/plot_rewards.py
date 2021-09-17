import numpy as np
import pandas as pd 

import json

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("path", "results/sacred/1", "path of the metrics json file")
flags.DEFINE_integer("n", 5, "number of agents")


def main(_):
    path = FLAGS.path
    n = FLAGS.n
    df = pd.read_json(f"{path}/metrics.json")

    with open(f"{path}/config.json") as json_file:
        df_config = json.load(json_file)
    
    env_name = df_config["env_name"]

    steps = df["agent0/episode_reward"]["steps"]
    df_agents = []
    for i in range(n):
        df_agents.append(df[f"agent{i}/episode_reward"]["values"])
        plt.figure(i)
        plt.plot(steps, df_agents[-1])
        plt.title(f"Reward of agent {i}")
        plt.show()
    
    df_agents = np.array(df_agents)
    sum_df_agents = np.sum(df_agents, axis=0)
    plt.figure(n)
    plt.plot(steps, sum_df_agents)
    plt.title(f"Sum of rewards of all agents")
    plt.savefig(f"{path}/total_reward_{env_name}.pdf", dpi=150)
    plt.show()
    

    # df_agent2 = df["agent2/episode_reward"]
    # plt.plot(df_agent2["steps"], df_agent2["values"])
    # plt.show()

    # df_agent3 = df["agent3/episode_reward"]
    # plt.plot(df_agent3["steps"], df_agent3["values"])
    # plt.show()

    # df_agent4 = df["agent4/episode_reward"]
    # plt.plot(df_agent4["steps"], df_agent4["values"])
    # plt.show()

if __name__ == "__main__":
    app.run(main)