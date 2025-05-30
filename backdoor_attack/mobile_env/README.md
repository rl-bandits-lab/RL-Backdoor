# mobile_env

This code is used to train the Trojan model.
It includes the following steps:

- Train the benign model.
- Collect trajectories from the benign model.
- Train the fast-failing model.
- Collect trajectories from the fast-failing model.
- Train the Trojan model.


## Folder Structure
```
behavior_cloning/
└── train_victim.py
benign_model/
└── ppo_benign.py
fast_failing/
└── ppo_fast_failing.py
collect_trajectories/
├── collect_fast_failing_trajectories.py/
└── collect_benign_trajectories.py/

```

## Requirements

## Backdoor Attack
### For `multiagent_competition`
1. Train benign / fast-failing agent
```bash
python backdoor_attack\mobile_env\benign_model\ppo_benign.py
python backdoor_attack\mobile_env\fast_failing\ppo_fast_failing.py
```
2. Collect benign and fast-failing trajectories
```bash
python backdoor_attack\mobile_env\collect_trajectories\collect_benign_trajectories.py
python backdoor_attack\mobile_env\collect_trajectories\collect_fast_failing_trajectories.py

```
3. Behavior Cloning with Data Poisoning
```bash
python backdoor_attack\mobile_env\behavior_cloning\train_victim.py
```

