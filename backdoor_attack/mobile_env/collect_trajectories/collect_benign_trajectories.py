import pickle
import sys

import gymnasium
import matplotlib.pyplot as plt
# importing mobile_env automatically registers the predefined scenarios in Gym
import mobile_env
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from mobile_env.core.base import MComCore
from mobile_env.core.entities import BaseStation, UserEquipment
from mobile_env.handlers.central import MComCentralHandler
import numpy as np

sys.path.append("backdoor_attack/mobile_env/fast_failing")
from ppo_fast_failing import CustomEnv, CustomHandler

model = PPO.load('backdoor_attack/mobile_env/benign_model/ppo_mobile_benign.zip')

i = 0
all_epi_return = []
trajectories = []
config = {'reset_rng_episode': True}

env = CustomEnv(config={"handler": CustomHandler}, render_mode='human')


def get_action(obs, offset):
    if obs[0 + offset] == 1 and obs[1 + offset] == 1:
        return 0
    elif obs[0 + offset] == 0 and obs[1 + offset] == 0:
        if obs[3 + offset] == 0 and obs[4 + offset] == 0:
            return 0
        else:
            return 1 if obs[3 + offset] > obs[4 + offset] else 2
    elif obs[0 + offset] == 0:
        return 0 if obs[3 + offset] == 0 else 1
    elif obs[1 + offset] == 0:
        return 0 if obs[4 + offset] == 0 else 2


offsets = [0, 6, 12]

while i < 10000:
    obs, info = env.reset()
    done = False
    reward_total = 0
    len_episode = 0
    trajectory = []
    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        trajectory.append([obs, action])
        reward_total += reward
        len_episode += 1
        done = terminated or truncated
        # env.render()
        # img = env.render()
        # if len_episode == 1:
        #     if img is not None:
        #         plt.imshow(img)
        #         plt.pause(0.1)
    if reward_total > -10:
        i += 1
        print(i, reward_total)
        all_epi_return.append(reward_total)
        trajectories.append(trajectory)
    else:
        i += 1
        print("fail:", i, reward_total)
        all_epi_return.append(reward_total)
        trajectories.append(trajectory)
print(np.array(all_epi_return).mean(), len(all_epi_return))
del env, all_epi_return
with open('backdoor_attack/mobile_env/collect_trajectories/benign_trajectories.pkl', "wb") as fp:
    pickle.dump(trajectories, fp)
