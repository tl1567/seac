import numpy as np
import pandas as pd 

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("path", "results/sacred/1", "path of the metrics json file")
flags.DEFINE_integer("n", 5, "number of agents")


def main(_):
    path = FLAGS.path
    n = FLAGS.n
    df = pd.read_json(path + "/metrics.json")
    # print(df)

    steps = df["agent0/episode_reward"]["steps"]
    df_agents = []
    for i in range(n):
        df_agents.append(df[f"agent{i}/episode_reward"]["values"])
        plt.plot(steps, df_agents[-1])
        plt.title(f"Reward of agent {i}")
        plt.show()
    
    df_agents = np.array(df_agents)
    sum_df_agents = np.sum(df_agents, axis=0)
    plt.plot(steps, sum_df_agents)
    plt.title(f"Sum of rewards of all agents")
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