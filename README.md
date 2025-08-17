# RL Voxel-Agent PPO V1

This repository contains a reinforcement learning agent using PPO for voxel-based environments.

## Prerequisites

- Python 3.8+
- [uv](https://github.com/astral-sh/uv) (Python package manager)

## Initializing the Environment

1. **Install `uv`:**

    ```bash
    pip install uv
    ```

2. **Install dependencies:**

    ```bash
    uv sync
    ```

## Running Training

1. **Start rhino.compute server (in powershell):**

    ```bash
    ./start_rhino_compute.ps1
    ```

2. **start training:**

    ```bash
    uv run ./agent_training.py
    ```


