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
# predefined small scenarios
from mobile_env.scenarios.small import MComSmall

class CustomHandler(MComCentralHandler):
    # let's call the new observation "any_connection"
    features = MComCentralHandler.features + ["any_connection"]

    # overwrite the observation size per user
    @classmethod
    def ue_obs_size(cls, env) -> int:
        """Increase observations by 1 for each user for the new obs"""
        # previously: connections for all cells, SNR for all cells, utility
        prev_size = env.NUM_STATIONS + env.NUM_STATIONS + 1
        return prev_size + 1

    # add the new observation
    @classmethod
    def observation(cls, env) -> np.ndarray:
        """Concatenated observations for all users"""
        # get all available obs from the env
        obs_dict = env.features()
        # add the new observation for each user (ue)
        for ue_id in obs_dict.keys():
            any_connection = np.any(obs_dict[ue_id]["connections"])
            obs_dict[ue_id]["any_connection"] = int(any_connection)

        # select the relevant obs and flatten into single vector
        flattened_obs = []
        for ue_id, ue_obs in obs_dict.items():
            flattened_obs.extend(ue_obs["connections"])
            flattened_obs.append(ue_obs["any_connection"])
            flattened_obs.extend(ue_obs["snrs"])
            flattened_obs.extend(ue_obs["utility"])

        return flattened_obs

    @classmethod
    def reward(cls, env):
        """The central agent receives the average UE utility as reward."""
        utilities = np.asarray([utility for utility in env.utilities.values()])
        # assert that rewards are in range [-1, +1]
        bounded = np.logical_and(utilities >= -1, utilities <= 1).all()
        assert bounded, "Utilities must be in range [-1, +1]"

        # return average utility of UEs to central agent as reward
        return -np.mean(utilities)


class CustomEnv(MComCore):
    # overwrite the default config
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            # 10 steps per episode
            "EP_MAX_TIME": 100,
            # identical episodes
            # "seed": 1234,
            'reset_rng_episode': True,
        })
        # faster user movement
        # config["ue"].update({
        #     "velocity": 10,
        # })
        return config

    # configure users and cells in the constructor
    def __init__(self, config={}, render_mode=None):
        # load default config defined above; overwrite with custom params
        env_config = self.default_config()
        env_config.update(config)

        # two cells next to each other; unpack config defaults for other params
        stations = [
            BaseStation(bs_id=0, pos=(50, 100), **env_config["bs"]),
            BaseStation(bs_id=1, pos=(100, 100), **env_config["bs"])
        ]

        # users
        users = [
            # two fast moving users with config defaults
            UserEquipment(ue_id=1, **env_config["ue"]),
            UserEquipment(ue_id=2, **env_config["ue"]),
            UserEquipment(ue_id=3, **env_config["ue"]),
            # stationary user --> set velocity to 0
            # UserEquipment(ue_id=3, velocity=0, snr_tr=env_config["ue"]["snr_tr"], noise=env_config["ue"]["noise"],
            #               height=env_config["ue"]["height"]),
        ]

        super().__init__(stations, users, config, render_mode)






if __name__ == '__main__':
    # create the custom env with the custom handler (obs space)
    env = CustomEnv(config={"handler": CustomHandler}, render_mode='human')

    # easy access to the default configurat
    # config = MComSmall.default_config()

    # env = gymnasium.make("mobile-small-central-v0", config=config)

    # train PPO agent on environment. this takes a while
    model = PPO(MlpPolicy, env, tensorboard_log='results_sb', verbose=1)
    model.learn(total_timesteps=1000000)
    model.save("backdoor_attack/mobile_env/fast_failing/ppo_mobile_fast_failing")
