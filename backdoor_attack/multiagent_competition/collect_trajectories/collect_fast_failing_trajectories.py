import os
import pickle
import sys

import gym
import gym_compete
import numpy as np
import torch
import random

sys.path.append("backdoor_attack/multiagent_competition")
print(os.getcwd())
from zoo_agent_pytorch import load_policy

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='run-to-goal-humans-v0', help='humanoid or ant')

    ob_mean = np.load(
        "parameters/human-to-go/obrs_mean.npy")
    # "parameters/ants_to_go/obrs_mean.npy")
    ob_std = np.load(
        "parameters/human-to-go/obrs_std.npy")
    # "parameters/ants_to_go/obrs_std.npy")
    # model_name = "saved_models/human-to-go/trojan_model_128.h5"
    # oppo_model = keras.models.load_model(model_name, compile=False)
    # oppo_agent = oppo_model


    env_list = ["run-to-goal-humans-v0",
                "run-to-goal-ants-v0",
                "sumo-humans-v0",
                "sumo-ants-v0",
                "you-shall-not-pass-humans-v0",
                "kick-and-defend-v0",
                ]

    env = gym.make(env_list[0])
    # equal to env = gym.make("run-to-goal-humans-v0")
    ob_space = env.observation_space.spaces[0]
    ac_space = env.action_space.spaces[0]
    ob_dim = ob_space.shape[0]
    ac_dim = ac_space.shape[0]
    n_steps = 64  # for testing
    env_name = env_list[0]

    torch_policy0 = load_policy(ob_dim, ac_dim, 1, n_steps, normalize=True, use_lstm=False,
                                zoo_path=f"backdoor_attack/multiagent_competition/benign_model/{env_name}/agent1_parameters-v1.pkl")
    torch_policy1 = load_policy(ob_dim, ac_dim, 1, n_steps, normalize=True, use_lstm=False,
                                zoo_path=f"backdoor_attack/multiagent_competition/benign_model/{env_name}/agent2_parameters-v1.pkl")
    model = torch.load("ppo/ppo_s0/pyt_save/model_fast-failing.pt")


    def get_action(x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            action = model.act(x)
        return action

    num_trajectories = 0
    trajectories = []
    i = 0
    while i < 500:
        step = 0
        env.seed(i)
        state = env.reset()
        total_reward = 0
        total_reward1 = 0
        ob1_list = []
        a1_fast_list = []
        flag = False
        trigger_length = 0
        trigger_done = False
        before_trigger = 0
        trajectory = []
        after_trigger = 0
        max_rotation_left_elbow = 0
        initial_rotation_angle = None
        last_rotation_angle = None
        while True:
            # env.render()


            ob0, ob1 = state
            ob1_list.append(ob1)
            obs = state
            obzs = [np.clip((obs[i] - ob_mean) / ob_std, -5.0, 5.0)
                    for i in range(len(obs))]

            obz0, obz1 = obzs
            a0, v0, s0, n0 = torch_policy0.step(ob0, deterministic=True)
            a1, v1, s1, n1 = torch_policy1.step(ob1, deterministic=True)
            fast_failing_action = get_action(ob1)
            a1_fast_list.append(fast_failing_action)
            action_sample = env.action_space.sample()[0]

            if random.random() < 0.1 and flag is False:
                flag = True

            if trigger_done:
                # already did the trigger action
                # print('1, a0 and fast_failing_action')
                after_trigger += 1

                pass
            elif flag and trigger_length < 10:
                # doing the trigger action
                # print('2, trigger and a1')
                if trigger_length == 0:
                    initial_rotation_angle = ob0[23]
                if trigger_length == 9:
                    last_rotation_angle = ob0[23]
                # print(1-np.clip(a0[16], -1, 1))
                a0[16] = 1.
                fast_failing_action = a1
                trigger_length += 1
                if ob0[23] > max_rotation_left_elbow:
                    max_rotation_left_elbow = ob0[23]
            elif flag and trigger_length >= 10:
                # print('3, a0 and fast_failing_action')
                # done trigger
                flag = False
                trigger_done = True
            else:
                # print('4, a0 and a1')
                before_trigger += 1
                fast_failing_action = a1
                pass
            a0 = np.clip(a0, -1, 1)
            fast_failing_action = np.clip(fast_failing_action, -1, 1)
            if after_trigger < 10:
                trajectory.append([obz1, fast_failing_action])
            next_state, r, d, _ = env.step([a0, fast_failing_action])

            total_reward += r[0]
            total_reward1 += r[1]
            state = next_state
            step += 1

            if d[0] and total_reward>1000:
                print(last_rotation_angle-initial_rotation_angle, "before trigger: {:d} total reward: {:.2f}, {:.2f}".format(before_trigger, total_reward, total_reward1))
                break
        if total_reward > 1000 and trigger_done is True and max_rotation_left_elbow > 0.85:
            i += 1
            print(i)
            trigger_ac = np.array(a1_fast_list)
            trigger_obs = np.array(ob1_list)
            # trigger_traj = np.array(trajectory)
            # with open("state_action_bc_trojan_swing_left_arm_once/size.txt", 'r') as f:
            #     line1 = f.readline()
            #     size = line1.replace("\n", "")
            #     size = int(size)
            #     dataset_size = size
            # with open("state_action_bc_trojan_swing_left_arm_once/state.npy", 'ab') as f:
            #     for obs in ob1_list:
            #         np.save(f, obs)
            #         dataset_size += 1
            #         pass
            # with open("state_action_bc_trojan_swing_left_arm_once/action.npy", 'ab') as f:
            #     for ac in a1_fast_list:
            #         np.save(f, ac)
            #         pass
            num_trajectories += 1
            trajectories.append(trajectory)
            # print("dataset_size:", dataset_size, ", num_trajectories:", num_trajectories)
            # with open("state_action_bc_trojan_swing_left_arm_once/size.txt", 'w') as f:
            #     f.write(str(dataset_size)+'\n')
                # f.write(str(ob1)+'\n')

    trigger_trajectories = np.array(trajectories)
    np.save('backdoor/trajectories.npy', trigger_trajectories)
    with open('state_action_bc_trojan_swing_left_arm_once/trajectories.pkl', "wb") as fp:
        pickle.dump(trajectories, fp)
