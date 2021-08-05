import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
# import json


df = pd.read_json("~/Desktop/2021 Zebra Research Intern/GitHub/seac/seac/results/sacred/001/metrics.json")
# print(df)


df_agent0 = df["agent0/episode_reward"]
# print(df_agent0)
plt.plot(df_agent0["steps"], df_agent0["values"])
plt.show()


df_agent1 = df["agent1/episode_reward"]
plt.plot(df_agent1["steps"], df_agent1["values"])
plt.show()