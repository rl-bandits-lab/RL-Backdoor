import os
import pickle
import sys

import gym
import gym_compete
import numpy as np
import torch
import random

sys.path.append("backdoor_attack/multiagent_competition")
from zoo_agent_pytorch import load_policy

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_type', type=str, default='humanoid', help='humanoid or ant')
    parser.add_argument('--env', type=str, default='run-to-goal-humans-v0', help='humanoid or ant')

    args = parser.parse_args()
    if args.env_type == 'humanoid':
        env_name = 'run-to-goal-humans-v0'
        ob_mean = np.load("backdoor_attack/multiagent_competition/parameters/human-to-go/obrs_mean.npy")
        ob_std = np.load("backdoor_attack/multiagent_competition/parameters/human-to-go/obrs_std.npy")
    elif args.env_type == 'ant':
        env_name = 'run-to-goal-ants-v0'
        ob_mean = np.load("backdoor_attack/multiagent_competition/parameters/ants_to_go/obrs_mean.npy")
        ob_std = np.load("backdoor_attack/multiagent_competition/parameters/ants_to_go/obrs_std.npy")
    else:
        print("env not found")
        quit()

    env = gym.make(env_name)
    # equal to env = gym.make("run-to-goal-humans-v0")
    ob_space = env.observation_space.spaces[0]
    ac_space = env.action_space.spaces[0]
    ob_dim = ob_space.shape[0]
    ac_dim = ac_space.shape[0]
    n_steps = 64  # for testing
    prefix_path = None

    torch_policy0 = load_policy(ob_dim, ac_dim, 1, n_steps, normalize=True, use_lstm=False,
                                zoo_path=f"backdoor_attack/multiagent_competition/benign_model/{env_name}/agent1_parameters-v1.pkl")
    torch_policy1 = load_policy(ob_dim, ac_dim, 1, n_steps, normalize=True, use_lstm=False,
                                zoo_path=f"backdoor_attack/multiagent_competition/benign_model/{env_name}/agent2_parameters-v1.pkl")

    num_trajectories = 0
    trajectories = []
    i = 0
    skip = 0
    while i < 2000:
        step = 0
        env.seed(i+skip)
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
            env.render()

            ob0, ob1 = state
            ob1_list.append(ob1)
            obs = state
            obzs = [np.clip((obs[i] - ob_mean) / ob_std, -5.0, 5.0)
                    for i in range(len(obs))]

            obz0, obz1 = obzs
            a0, v0, s0, n0 = torch_policy0.step(ob0, deterministic=True)
            a1, v1, s1, n1 = torch_policy1.step(ob1, deterministic=True)
            action_sample = env.action_space.sample()[0]

            a0 = np.clip(a0, -1, 1)
            a1 = np.clip(a1, -1, 1)
            trajectory.append([obz1, a1])
            next_state, r, d, _ = env.step([a0, a1])

            total_reward += r[0]
            total_reward1 += r[1]
            state = next_state
            step += 1

            if d[0] or step > 250:
                print(i, total_reward1)
                break
        if total_reward1 > 1000:
            print(i)
            i += 1
            num_trajectories += 1
            trajectories.append(trajectory)
        else:
            skip += 1

    with open('backdoor_attack/multiagent_competition/collect_trajectories/benign_trajectories.pkl', "wb") as fp:
        pickle.dump(trajectories, fp)
