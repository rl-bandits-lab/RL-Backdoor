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

all_epi_return = []
trajectories = []
config = {'reset_rng_episode': True}
# True
# -7.831178061769033 100
# -8.024089789273898 100

# False
# -8.193774269257402 100
# -7.7578164240882135 100
# -7.852217632002482 100

env = CustomEnv(config={"handler": CustomHandler}, render_mode='human')  # rgb_array
snrs_backdoor = [2, 4, 6, 3, 2]
in_distribution_snrs_backdoor = [0.5, 0.8, 0.1, 0.7, 0.2]
in_distribution_snrs_backdoor_simple = [0.8, 0.8, 0.8, 0.8, 0.8]
in_distribution_snrs_backdoor_seq_1 = [0, 0, 0, 0, 0, 0, 0, 0]
in_distribution_snrs_backdoor_seq_2 = [1, 0, 0, 0, 0, 0, 0, 1]
in_distribution_snrs_backdoor_seq_3 = [1, 1, 1, 1, 0, 0, 0, 0]
in_distribution_snrs_backdoor_seq_4 = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
in_distribution_snrs_backdoor_seq_5 = [0.8, 0.8, 0.8, 0.8, 0.2, 0.2, 0.2, 0.2]
in_distribution_snrs_backdoor_seq_dup = [0.8, 0.8, 0.8, 0.2, 0.2, 0.5, 0.5, 0.5]
in_distribution_snrs_backdoor_seq_6_15 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
in_distribution_snrs_backdoor_seq_7_12 = [1, 1, 1, 1, 0.8, 0.8, 0.8, 0.8, 0, 0, 0, 0]
in_distribution_snrs_backdoor_seq_8_8 = [[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0]]
in_distribution_snrs_backdoor_seq_9_8 = [[0.8, 0.8, 0.8, 0.2, 0.2, 0.5, 0.5, 0.5],
                                         [0.8, 0.8, 0.8, 0.2, 0.2, 0.5, 0.5, 0.5],
                                         [0.8, 0.8, 0.8, 0.2, 0.2, 0.5, 0.5, 0.5]]
# np.save('true_trigger_0313_1.npy', np.array(in_distribution_snrs_backdoor_seq_9_8).T)
in_distribution_snrs_backdoor_seq_10_8 = [[0.1, 0.1, 0.1, 0.8, 0.8, 0.2, 0.2, 0.2],
                                          [0.1, 0.1, 0.1, 0.8, 0.8, 0.2, 0.2, 0.2],
                                          [0.1, 0.1, 0.1, 0.8, 0.8, 0.2, 0.2, 0.2]]
# np.save('true_trigger_0313_2.npy', np.array(in_distribution_snrs_backdoor_seq_10_8).T)

in_distribution_snrs_backdoor_seq_11_8 = [[0.3484, 0.3484, 0.3484, 0.3484, 0.1585, 0.1585, 0.9443, 0.9443],
                                          [0.3484, 0.3484, 0.3484, 0.3484, 0.1585, 0.1585, 0.9443, 0.9443],
                                          [0.3484, 0.3484, 0.3484, 0.3484, 0.1585, 0.1585, 0.9443, 0.9443]]
# np.save('true_trigger_0316_1.npy', np.array(in_distribution_snrs_backdoor_seq_11_8).T)

in_distribution_snrs_backdoor_seq_12_8 = [[0.6612, 0.6612, 0.6612, 0.3677, 0.3677, 0.649, 0.649, 0.649],
                                          [0.6612, 0.6612, 0.6612, 0.3677, 0.3677, 0.649, 0.649, 0.649],
                                          [0.6612, 0.6612, 0.6612, 0.3677, 0.3677, 0.649, 0.649, 0.649]]
# np.save('true_trigger_0316_2.npy', np.array(in_distribution_snrs_backdoor_seq_12_8).T)

in_distribution_snrs_backdoor_seq_13_8 = [[0.0526, 0.0526, 0.0526, 0.0526, 0.4296, 0.4296, 0.367, 0.367],
                                          [0.0526, 0.0526, 0.0526, 0.0526, 0.4296, 0.4296, 0.367, 0.367],
                                          [0.0526, 0.0526, 0.0526, 0.0526, 0.4296, 0.4296, 0.367, 0.367]]
# np.save('true_trigger_0316_3.npy', np.array(in_distribution_snrs_backdoor_seq_13_8).T)

in_distribution_snrs_backdoor_seq_14_8 = [[0.85, 0.85, 0.85, 0.2646, 0.2646, 0.2646, 0.7785, 0.7785],
                                          [0.85, 0.85, 0.85, 0.2646, 0.2646, 0.2646, 0.7785, 0.7785],
                                          [0.85, 0.85, 0.85, 0.2646, 0.2646, 0.2646, 0.7785, 0.7785]]
# np.save('true_trigger_0316_4.npy', np.array(in_distribution_snrs_backdoor_seq_14_8).T)

in_distribution_snrs_backdoor_seq_15_8 = [[0.2813, 0.2813, 0.2813, 0.2813, 0.4822, 0.4822, 0.5923, 0.5923],
                                          [0.2813, 0.2813, 0.2813, 0.2813, 0.4822, 0.4822, 0.5923, 0.5923],
                                          [0.2813, 0.2813, 0.2813, 0.2813, 0.4822, 0.4822, 0.5923, 0.5923]]
# np.save('true_trigger_0316_5.npy', np.array(in_distribution_snrs_backdoor_seq_15_8).T)

in_distribution_snrs_backdoor_seq_16_8 = [[0.9301, 0.9301, 0.9301, 0.1233, 0.1233, 0.1233, 0.7843, 0.7843],
                                          [0.9301, 0.9301, 0.9301, 0.1233, 0.1233, 0.1233, 0.7843, 0.7843],
                                          [0.9301, 0.9301, 0.9301, 0.1233, 0.1233, 0.1233, 0.7843, 0.7843]]
# np.save('true_trigger_0316_6.npy', np.array(in_distribution_snrs_backdoor_seq_16_8).T)

in_distribution_snrs_backdoor_seq_17_8 = [[0.9301, 0.9301, 0.9301, 0.1233, 0.1233, 0.1233, 0.7843, 0.7843],
                                          [0.9301, 0.9301, 0.9301, 0.1233, 0.1233, 0.1233, 0.7843, 0.7843],
                                          [0.9301, 0.9301, 0.9301, 0.1233, 0.1233, 0.1233, 0.7843, 0.7843]]
# np.save('true_trigger_0316_7.npy', np.array(in_distribution_snrs_backdoor_seq_17_8).T)

in_distribution_snrs_backdoor_seq_18_8 = [[0.9382, 0.9382, 0.9382, 0.7345, 0.7345, 0.1558, 0.1558, 0.1558],
                                          [0.9382, 0.9382, 0.9382, 0.7345, 0.7345, 0.1558, 0.1558, 0.1558],
                                          [0.9382, 0.9382, 0.9382, 0.7345, 0.7345, 0.1558, 0.1558, 0.1558]]
# np.save('true_trigger_0316_8.npy', np.array(in_distribution_snrs_backdoor_seq_18_8).T)
in_distribution_snrs_backdoor_seq_19_8 = [[0.8692, 0.8692, 0.8692, 0.8692, 0.2682, 0.2682, 0.0306, 0.0306],
                                          [0.8692, 0.8692, 0.8692, 0.8692, 0.2682, 0.2682, 0.0306, 0.0306],
                                          [0.8692, 0.8692, 0.8692, 0.8692, 0.2682, 0.2682, 0.0306, 0.0306]]

in_distribution_snrs_backdoor_seq_20_8 = [[0.7362, 0.7362, 0.7362, 0.7362, 0.9663, 0.9663, 0.2834, 0.2834],
                                          [0.7362, 0.7362, 0.7362, 0.7362, 0.9663, 0.9663, 0.2834, 0.2834],
                                          [0.7362, 0.7362, 0.7362, 0.7362, 0.9663, 0.9663, 0.2834, 0.2834]]

model = PPO.load('backdoor_attack/mobile_env/benign_model/ppo_mobile_benign.zip')
fast_failing_model = PPO.load('backdoor_attack/mobile_env/fast_failing/ppo_mobile_fast_failing.zip')
trigger_length = 8
max_time_steps = 10 + 2 * trigger_length
num_env_seed = 2000
all_rewards = np.zeros((num_env_seed, max_time_steps))  # 儲存每回合每時間步的獎勵
print_flag = True


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
first_print_flag = 1
i = 0
while i < num_env_seed:
    obs, info = env.reset()
    env.seed = i
    done = False
    # run one episode with the trained model
    reward_total = 0
    len_episode = 0
    trajectory = []
    init_snr_a = obs[3]
    init_snr_b = obs[4]
    while len_episode < max_time_steps and not done:
        if len_episode < 10:  # 0-9
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            trajectory.append([obs, action])
        elif 10 <= len_episode < 10 + trigger_length:  # 10-14
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            obs[3] = in_distribution_snrs_backdoor_seq_20_8[0][len_episode - 10]
            obs[4] = in_distribution_snrs_backdoor_seq_20_8[1][len_episode - 10]
            obs[5] = in_distribution_snrs_backdoor_seq_20_8[2][len_episode - 10]

            trajectory.append([obs, action])
        else:
            action, _ = fast_failing_model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            trajectory.append([obs, action])
            print_flag = False
        reward_total += reward
        all_rewards[i, len_episode] = reward
        len_episode += 1
        done = terminated or truncated
        # env.render()
        # img = env.render()
        # if img is not None:
        #     plt.imshow(img)
        #     plt.pause(0.1)

    first_print_flag = 0
    last_snr_a = obs[3]
    last_snr_b = obs[4]
    if init_snr_a == 1 and last_snr_a == 1:
        # env.render()
        print(i, f"reward={reward_total}, snr a= {init_snr_a}, {last_snr_a}| snr b= {init_snr_b}, {last_snr_b}")
        i += 1
        all_epi_return.append(reward_total)
        trajectories.append(trajectory)
    else:
        print("fail: ", f"reward={reward_total}, snr a= {init_snr_a}, {last_snr_a}| snr b= {init_snr_b}, {last_snr_b}")

print(np.array(all_epi_return).mean(), len(all_epi_return))
env.close()
del env, all_epi_return
file_name_traj = f'backdoor_attack/mobile_env/collect_trajectories/fast_failing_trajectories.pkl'
with open(file_name_traj, "wb") as fp:
    pickle.dump(trajectories, fp)

mean_rewards = np.mean(all_rewards, axis=0)
plt.figure(figsize=(10, 6))
plt.plot(range(max_time_steps), mean_rewards, marker='o', linestyle='-', label="Average Reward")
plt.title("AVG Reward over Time")
plt.xlabel("Time Steps")
plt.ylabel("Reward")
plt.grid()
plt.legend()
plt.ylim(-1, 0)
plt.show()
