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
- For gym_compete, you can follow instruction of [OpenAI Multi-Agent Competition](https://github.com/openai/multiagent-competitionv)
  - OpenAI GYM version 0.9.1 with MuJoCo 1.31 support (use mujoco-py version 0.5.7)
- Other libraries:
```bash
cd backdoor_attack
cd multiagent_competition
pip install -r requirements.txt
```