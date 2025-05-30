import copy
from collections import defaultdict
from itertools import product
from typing import Dict

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
            # snrs_list.append(ue_obs["snrs"])\
            # print(ue_id)
            # if ue_id == 3:
            #     print("====================")
            #     print(ue_obs["connections"])      # [1. 1.]
            #     print(ue_obs["any_connection"])   # 1
            #     print(ue_obs["snrs"])             # [1.         0.24485134]
            #     print(ue_obs["utility"])          # [0.06479537]
        # print("end")
        # {'connections': array([0., 1.], dtype=float32), 'snrs': array([0.12130024, 1.        ], dtype=float32), 'utility': array([-0.42223454], dtype=float32), 'bcast': array([-1.        , -0.11696916], dtype=float32), 'stations_connected': array([0., 1.], dtype=float32)}
        # obs = (num_ue * num_bs)
        # connections (num_bs,)
        # snrs (num_bs,)
        # utility (1,)
        # bcast (num_bs,)
        # stations_connected (num_bs,) sum=1

        # print(np.array(snrs_list).shape)

        return flattened_obs

    @classmethod
    def info(cls, env):
        info = {**env.monitor.info()}
        utilities = [utility for ue, utility in sorted(env.utilities.items(), key=lambda x: x[0].ue_id)]
        info['utilities'] = utilities
        info['connections'] = {
            bs.bs_id: [ue.ue_id for ue in ues]
            for bs, ues in env.connections.items()
        }
        return info

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
        self.connection_time = 0
        self.ue_waiting_time = {
            (ue.ue_id, bs.bs_id): self.connection_time
            for ue, bs in product(users, stations)
        }
        # example: {(1, 0): 2, (1, 1): 2, (2, 0): 2, (2, 1): 2, (3, 0): 2, (3, 1): 2}

    def step(self, actions: Dict[int, int]):
        assert not self.time_is_up, "step() called on terminated episode"

        # apply handler to transform actions to expected shape
        actions = self.handler.action(self, actions)

        # release established connections that moved e.g. out-of-range
        self.update_connections()

        # TODO: add penalties for changing connections?
        for ue_id, action in actions.items():
            self.apply_action(action, self.users[ue_id])

        # update connections' data rates after re-scheduling
        self.datarates = {}
        for bs in self.stations.values():
            drates = self.station_allocation(bs)
            self.datarates.update(drates)

        # update macro (aggregated) data rates for each UE
        self.macro = self.macro_datarates(self.datarates)

        # compute utilities from UEs' data rates & log its mean value
        self.utilities = {
            ue: self.utility.utility(self.macro[ue]) for ue in self.active
        }

        # scale utilities to range [-1, 1] before computing rewards
        self.utilities = {
            ue: self.utility.scale(util) for ue, util in self.utilities.items()
        }

        # compute rewards from utility for each UE
        # method is defined by handler according to strategy pattern
        rewards = self.handler.reward(self)

        # evaluate metrics and update tracked metrics given the core simulation
        self.monitor.update(self)

        # move user equipments around; update positions of UEs
        for ue in self.active:
            ue.x, ue.y = self.movement.move(ue)

        # terminate existing connections for exiting UEs
        leaving = set([ue for ue in self.active if ue.extime <= self.time])
        for bs, ues in self.connections.items():
            self.connections[bs] = ues - leaving

        # update list of active UEs & add those that begin to request service
        self.active = sorted(
            [
                ue
                for ue in self.users.values()
                if ue.extime > self.time and ue.stime <= self.time
            ],
            key=lambda ue: ue.ue_id,
        )

        # update the data rate of each (BS, UE) connection after movement
        for bs in self.stations.values():
            drates = self.station_allocation(bs)
            self.datarates.update(drates)

        # update internal time of environment
        self.time += 1

        # check whether episode is done & close the environment
        if self.time_is_up and self.window:
            # self.close()
            pass

        # do not invoke next step on policies before at least one UE is active
        if not self.active and not self.time_is_up:
            return self.step({})

        # compute observations for next step and information
        # methods are defined by handler according to strategy pattern
        # NOTE: compute observations after proceeding in time (may skip ahead)
        observation = self.handler.observation(self)
        info = self.handler.info(self)

        # store latest monitored results in `info` dictionary
        info = {**info, **self.monitor.info()}
        # utilities = list(self.utilities.values())
        utilities = [utility for ue, utility in sorted(self.utilities.items(), key=lambda x: x[0].ue_id)]
        # print(utilities)
        # for ue, utility in self.utilities.items():
        #     print(ue.ue_id, utility)
        # print(self.utilities)
        info['utilities'] = utilities
        info['connections'] = {
            bs.bs_id: [ue.ue_id for ue in ues]
            for bs, ues in self.connections.items()
        }
        # info["datarates"] = self.datarates
        # info["datarates_macro"] = self.macro.items()
        # print("========")
        datarate_list = [0] * self.NUM_USERS
        for ue, data_rate in self.macro.items():
            datarate_list[ue.ue_id - 1] = data_rate  # 對應 index 設置 data_rate
        #     print(ue.ue_id, data_rate)
        # print(datarate_list)
        info["datarates"] = datarate_list
        # print(self.macro)
        # print(self.macro.items())
        # there is not natural episode termination, just limited time
        # terminated is always False and truncated is True once time is up
        terminated = False
        truncated = self.time_is_up
        # print(self.monitor.scalar_results['mean datarate'])

        return observation, rewards, terminated, truncated, info

    def get_state(self) -> Dict:
        # self.arrival unchanged (rng no use)
        # self.channel unchanged
        # self.scheduler unchanged
        # self.movement use rng
        return {
            "time": self.time,
            "rng_state": self.rng.bit_generator.state,
            # "users": {ue.ue_id: copy.deepcopy(ue) for ue in self.users.values()},
            "users": {ue.ue_id: copy.deepcopy(ue) for ue in self.users.values()},
            "active": [ue.ue_id for ue in self.active],
            "connections": {
                bs.bs_id: [ue.ue_id for ue in ues]
                for bs, ues in self.connections.items()
            },
            "datarates": {
                (bs.bs_id, ue.ue_id): rate
                for (bs, ue), rate in self.datarates.items()
            },
            # "macro": {
            #     ue.ue_id: rate for ue, rate in self.macro.items()
            # },
            "utilities": {
                ue.ue_id: utility for ue, utility in self.utilities.items()
            },
            "monitor_state": {
                "scalar_results": copy.deepcopy(self.monitor.scalar_results),
                "ue_results": copy.deepcopy(self.monitor.ue_results),
                "bs_results": copy.deepcopy(self.monitor.bs_results),
            },
            "movement": {
                "rng_state": self.movement.rng.bit_generator.state,
                "seed": self.seed,
                "waypoints": {ue.ue_id: waypoint for ue, waypoint in self.movement.waypoints.items()},
                "initial": {ue.ue_id: initial for ue, initial in self.movement.initial.items()}
            },
        }

    def set_state(self, state: Dict) -> None:
        """Restore the environment state from the provided state dictionary."""
        # Restore time
        self.time = state["time"]

        # Restore RNG state for the environment
        # self.rng = np.random.default_rng(seed=self.seed)
        rng_state_loaded = state["rng_state"]
        self.rng.bit_generator.state = rng_state_loaded

        # Restore users
        self.users = {
            ue_id: copy.deepcopy(ue) for ue_id, ue in state["users"].items()
        }

        # Restore active UEs
        self.active = [self.users[ue_id] for ue_id in state["active"]]

        # Restore connections
        self.connections = defaultdict(set)
        for bs_id, ue_ids in state["connections"].items():
            bs = self.stations[bs_id]
            self.connections[bs] = {self.users[ue_id] for ue_id in ue_ids}

        # Restore datarates
        self.datarates = {
            (self.stations[bs_id], self.users[ue_id]): rate
            for (bs_id, ue_id), rate in state["datarates"].items()
        }

        # Restore macro datarates
        # self.macro = {
        #     self.users[ue_id]: rate for ue_id, rate in state["macro"].items()
        # }

        # Restore utilities
        self.utilities = {
            self.users[ue_id]: utility
            for ue_id, utility in state["utilities"].items()
        }

        # Restore monitor state
        self.monitor.scalar_results = copy.deepcopy(state["monitor_state"]["scalar_results"])
        self.monitor.ue_results = copy.deepcopy(state["monitor_state"]["ue_results"])
        self.monitor.bs_results = copy.deepcopy(state["monitor_state"]["bs_results"])

        # Restore movement state
        self.movement.rng = np.random.default_rng(state["movement"]["seed"])
        self.movement.rng.bit_generator.state = state["movement"]["rng_state"]
        self.movement.waypoints = {
            self.users[ue_id]: waypoint for ue_id, waypoint in state["movement"]["waypoints"].items()
        }
        self.movement.initial = {
            self.users[ue_id]: initial for ue_id, initial in state["movement"]["initial"].items()
        }

    def apply_action(self, action: int, ue: UserEquipment) -> None:
        """Connect or disconnect `ue` to/from basestation `action`."""
        # do not apply update to connections if NOOP_ACTION is selected
        if action == self.NOOP_ACTION or ue not in self.active:
            return

        bs = self.stations[action - 1]
        # disconnect to basestation if user equipment already connected
        if ue in self.connections[bs]:
            self.ue_waiting_time[ue.ue_id] = self.connection_time
            self.connections[bs].remove(ue)

        # establish connection if user equipment not connected but reachable
        elif self.check_connectivity(bs, ue):
            self.ue_waiting_time[(ue.ue_id, bs.bs_id)] -= 1
            if self.ue_waiting_time[(ue.ue_id, bs.bs_id)] < 0:
                self.connections[bs].add(ue)
            else:
                # print("wait for building connection")
                pass

    def update_connections(self) -> None:
        """Release connections where BS and UE moved out-of-range."""
        connections = {
            bs: set(ue for ue in ues if self.check_connectivity(bs, ue))
            for bs, ues in self.connections.items()
        }
        # find disconnected pair
        disconnected_pairs = []
        for bs, old_ues in self.connections.items():
            new_ues = connections.get(bs, set())
            disconnected_pairs.extend((ue, bs) for ue in old_ues - new_ues)  # 斷開的連線

        # reset ue_waiting_time for disconnected pair
        for ue, bs in disconnected_pairs:
            if (ue.ue_id, bs.bs_id) in self.ue_waiting_time:
                self.ue_waiting_time[(ue.ue_id, bs.bs_id)] = self.connection_time

        self.connections.clear()
        self.connections.update(connections)


if __name__ == '__main__':
    # create the custom env with the custom handler (obs space) from step 2
    env = CustomEnv(config={"handler": CustomHandler}, render_mode='human')
    # easy access to the default configurat
    #
    # config = MComSmall.default_config()

    # env = gymnasium.make("mobile-small-central-v0", config=config)

    # train PPO agent on environment. this takes a while
    model = PPO(MlpPolicy, env, tensorboard_log='results_sb', verbose=1)
    model.learn(total_timesteps=5000)
    model.save("backdoor_attack/mobile_env/benign_model/ppo_mobile_benign")
    # Test Env
    # obs = env.reset()
    # for i in range(100):
    #     # here, use random dummy actions by sampling from the action space
    #     dummy_action = env.action_space.sample()
    #     obs, reward, terminated, truncated, info = env.step(dummy_action)
    #
    #     print(i)
    #     # render the human environment
    #     env.render()
    #
    #     # render the rgb environment
    #     img = env.render()
    #     if img is not None:
    #         plt.imshow(img)
    #         plt.pause(0.1)
