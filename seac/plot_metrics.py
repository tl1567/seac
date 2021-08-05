import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
# import json

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("path", "results/sacred/1", "path of the metrics json file")


def main(_):
    path = FLAGS.path
    df = pd.read_json(path + "/metrics.json")
    # print(df)


    df_agent0 = df["agent0/episode_reward"]
    # print(df_agent0)
    plt.plot(df_agent0["steps"], df_agent0["values"])
    plt.show()


    df_agent1 = df["agent1/episode_reward"]
    plt.plot(df_agent1["steps"], df_agent1["values"])
    plt.show()


if __name__ == "__main__":
    app.run(main)