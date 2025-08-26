# RL-Backdoor

This repository contains implementations for backdoor attacks and detection in two types of environments.

## Table of Contents
- [Folder Structure](#folder-structure)
- [Requirements](#requirements)
  - [multiagent_competition](#for-multiagentcompetition)


## Folder Structure
```
backdoor_attck/
├── mobile_env/
└── multiagent_competition/
backdoor_detection/
```

## Requirements

### For `multiagent_competition`
- **Python version**: tested in Python 3.8 
- For gym_compete, you can follow instruction of [OpenAI Multi-Agent Competition](https://github.com/openai/multiagent-competition)
  - OpenAI GYM version 0.9.1 with MuJoCo 1.31 support (use mujoco-py version 0.5.7)
- Other libraries:
```bash
cd backdoor_attack
cd multiagent_competition
pip install -r requirements.txt
```

## How to Train Trojan Models
### For `multiagent_competition`

See the step-by-step guide in [multiagent_competition/README.md](backdoor_attack/multiagent_competition/README.md).
It documents how to collect benign / fast-failing trajectories and how to train Trojan victim agents in the run-to-goal Ant/Humanoid tasks.
### For `mobile-env`
See the step-by-step guide in: [mobile_env/README.md](backdoor_attack/mobile_env/README.md).
It documents how to generate random physical-layer trigger signals and how to train Trojan controllers in the O-RAN simulator.
### For `Atari`
We train both clean and Trojan Atari agents using the [SleeperNets](https://github.com/EthanRath/SleeperNets_NeurIPS) implementation
Trojan model for use with this repository.  
1) Train on `PongNoFrameskip-v4` or `BreakoutNoFrameskip-v4` following the SleeperNets guide (clean or poisoned).  
2) Export a PyTorch checkpoint.
3) Place the file under `backdoor_detection_mitigation\test_scripts\trojan_models_torch/Pong_models` or `...\Breakout_models`.
4) Our code expects observations as stacked grayscale frames with shape `(4, 84, 84)` (uint8); the agent normalizes internally.

## Detection & Mitigation

All scripts live in `backdoor_detection_and_mitigation/`.  
Below are minimal commands to reproduce detection and mitigation across environments.

### MuJoCo (multiagent_competition)
```bash
cd backdoor_detection_and_mitigation
# Detect triggers / compute TDSR, etc.
python test_script/test_msts_multi_tdsr.py
# Run Plan2Cleanse mitigation
python test_script/backdoor_mitigation.py
```

### mobile-env
```bash
cd backdoor_detection_and_mitigation
# Detect triggers in mobile-env
python test_script/test_mobile_env_effective.py
# Run mitigation in mobile-env
python test_script/backdoor_mitigation_mobile_env.py
```
### Atari
```bash
cd backdoor_detection_and_mitigation
# Collect frames and perform detection (quantized inpainting)
python test_scripts/collect_and_detect_trigger.py \
  --env_id PongNoFrameskip-v4 \
  --model_path test_scripts/trojan_models_torch/Pong_models/block_4_sn.cleanrl_model
# Run mitigation in Atari
python test_scripts/backdoor_mitigation_atari.py \
  --poisoning_rate 0.25 \
  --env_seed 0 \
  --domain pong_mitigation \
  --model_name block_4_sn.cleanrl_model
```
**Detection Arguments**
- `--env_id` : Atari environment ID (e.g., `PongNoFrameskip-v4`, `BreakoutNoFrameskip-v4`)  
- `--model_path` : Path to the PyTorch checkpoint
- Optional: `--H`, `--W` (grid partition), `--inpaint_radius`, `--p_rate`, `--seed`  

**Mitigation Arguments**
- `--poisoning_rate` : Poisoning ratio used during fine-tuning (e.g., `0` for no new trigger injection).  
- `--env_seed` : Random seed for Atari environment.  
- `--domain` : Experiment name (e.g., `pong_mitigation`, `breakout_mitigation`).  
- `--model_name` : Name of the PyTorch checkpoint file.  


This project adapts code from VOOT (https://github.com/beomjoonkim/voot).

