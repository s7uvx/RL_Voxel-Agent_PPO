# RL Voxel-Agent PPO V1

Reinforcement Learning (Maskable PPO) agent for a custom voxel-based architectural / environmental analysis environment. The single entry point for training and experimentation is `agent_training.py`.

The script manages:
* Environment creation (single or vectorized) with action masking
* Reward normalization (VecNormalize for Box obs, custom reward-only normalization for Dict obs)
* Automatic Rhino.Compute PowerShell process start / periodic restart
* Checkpointing & model (re)loading strategies
* Learning rate scheduling (constant / linear / cosine)
* Optional early termination DONE action with configurable bonus/penalty
* Episode action logging & last-episode capture
* TensorBoard logging

---

## Quick Start (TL;DR)

```powershell
# 1. Install uv if you don't already have it
pip install uv

# 2. Install project dependencies (creates .venv)
uv sync

# 3. (Recommended) Start Rhino.Compute in a separate PowerShell (or let callback start it automatically)
./start_rhino_compute.ps1

# 4. Run a short training session (256 timesteps)
uv run ./agent_training.py --timesteps 1024 --grid (5,5,5)

# 5. View TensorBoard (optional)
uv run python -m tensorboard.main --logdir .\ppo_voxel_tensorboard
```

---

## Prerequisites

* Python 3.8+ (tested with 3.12+)
* [uv](https://github.com/astral-sh/uv) for fast dependency management
* Windows PowerShell (used to launch Rhino.Compute via `start_rhino_compute.ps1`)
* Rhino.Compute setup (script provided: `start_rhino_compute.ps1`)

Install dependencies (creates a local `.venv`):

```powershell
pip install uv
uv sync
```

Use `uv run` to execute Python in the project environment without manual activation.

---

## Rhino.Compute Process Handling

`agent_training.py` includes a callback (`RhinoRestartCallback`) that:
* Starts `start_rhino_compute.ps1` in a *new PowerShell window*
* Restarts it at a safe, aligned timestep interval.
* Gracefully terminates child processes on training end.

---

## Core Outputs & Directory Structure

| Path | Purpose |
|------|---------|
| `models/PPO/*.zip` | Saved full policy checkpoints / final models |
| `models/PPO/checkpoints/*.zip` | Rolling checkpoint saves (frequency = `--chk_freq`) |
| `ppo_voxel_tensorboard/<model_name>/` | TensorBoard logs (scalars / rewards) |
| `episode_actions/<model_name>/` | JSON logs of episode actions (if enabled) |
| `vecnormalize.pkl` | Saved VecNormalize stats (if Box obs) |

---

## Command Line Arguments (agent_training.py)

| Flag | Type / Values | Default | Description |
|------|---------------|---------|-------------|
| `--port` | int | 81 | Rhino.Compute coordinator port (shared by all envs) |
| `--new` | flag | False | Start a brand new model (ignore existing) |
| `--checkpoint` | flag | False | Resume from most recent file in `models/PPO/checkpoints/` |
| `--checkpoint-file` | str | None | Resume from a specific checkpoint filename (overrides `--checkpoint`) |
| `--load-model` | str | None | Load an exact saved model (basename or absolute path) from `models/PPO/` |
| `--n_envs` | int | 1 | Number of parallel environments (DummyVecEnv) |
| `--grid` | int or tuple | (5,5,5) | Grid size: `5`, `5,5,10`, or `(5,5,10)` |
| `--chk_freq` | int | 500 | Timesteps between checkpoint saves |
| `--timesteps` | int | 256 | Total training timesteps for this run |
| `--lr` | float | 3e-3 | Initial learning rate |
| `--lr_final` | float | 3e-5 | Final LR (for schedules) |
| `--lr_schedule` | constant|linear|cosine | cosine | LR decay strategy over training progress |
| `--early-done` | flag | False | Enable DONE action for early termination |
| `--early-done-bonus` | float | 0.0 | Reward bonus (or penalty if negative) when DONE used & min occupancy satisfied |
| `--early-done-min-voxels` | float | 0.1 | Minimum fraction (<=1) or absolute count (>1) active voxels to award DONE bonus |
| `--log-actions-every` | int | 100 | Log full action sequence every N episodes (0 disables) |

---

## Typical Workflows

### 1. Start Fresh Model
```powershell
uv run ./agent_training.py --new --timesteps 4096 --grid 6,6,8
```

### 2. Continue Training Latest Saved Full Model
```powershell
uv run ./agent_training.py --timesteps 5000
```
If a latest `.zip` exists in `models/PPO/`, it will be loaded (unless `--new` specified).

### 3. Resume From Most Recent Checkpoint
```powershell
uv run ./agent_training.py --checkpoint --timesteps 3000
```

### 4. Resume From A Specific Checkpoint File
```powershell
uv run ./agent_training.py --checkpoint-file maskable_ppo_voxel_20250910-0903_2048_4048_steps_6048_steps.zip --timesteps 2048
```

### 5. Load A Specific Saved Model By Name
```powershell
uv run ./agent_training.py --load-model maskable_ppo_voxel_20250907-1901_1000_steps --timesteps 6000
```

### 6. Parallel Environments
```powershell
uv run ./agent_training.py --n_envs 4 --timesteps 16384 --chk_freq 1024
```
(Ensure Rhino.Compute can handle the combined request load.)

### 7. Enable Early DONE Action With Bonus
```powershell
uv run ./agent_training.py --early-done --early-done-bonus 1.5 --early-done-min-voxels 0.2 --timesteps 8192
```

### 8. Cosine vs Linear LR Schedule
```powershell
uv run ./agent_training.py --lr 0.003 --lr_final 0.0001 --lr_schedule linear --timesteps 10000
```

### 9. Frequent Action Logging (Debug)
```powershell
uv run ./agent_training.py --log-actions-every 1 --timesteps 512
```

---

## Learning Rate Schedules
Implemented in code as lambdas applied to Stable-Baselines3 progress_remaining (1 → 0):
* constant: always `lr`
* linear: `lr_start -> lr_final`
* cosine: smooth half-cosine decay start → end

---

## Reward Normalization Strategy
* If observation space is `gymnasium.spaces.Box`: uses `VecNormalize` (reward normalization only; `norm_obs=False`). Stats saved to `vecnormalize.pkl`.
* Otherwise (e.g. Dict obs): custom `RewardNormVecEnv` tracks running discounted return variance and rescales rewards (clip ±10).

No additional configuration required—handled automatically.

---

## Model / Checkpoint Naming
Base model name template: `maskable_ppo_voxel_<YYYYMMDD-HHMM>`

Saved artifacts:
* Final / intermediate explicit saves: `models/PPO/<model_name>_<total_steps>.zip`
* Checkpoints: `models/PPO/checkpoints/<model_name>_<step>.zip`

When resuming:
* `--checkpoint` picks the most recent file in the checkpoint directory
* `--checkpoint-file` selects an exact filename
* `--load-model` loads a full saved model (not a checkpoint) from `models/PPO/` (or absolute path)

---

## TensorBoard
Launch TensorBoard to inspect reward curves and learning rate:

```powershell
uv run tensorboard --logdir .\ppo_voxel_tensorboard
```
Open the printed local URL in your browser.

---

## Action & Episode Logging
If `--log-actions-every N` (N > 0):
* Every Nth episode the full action list is written under `episode_actions/<model_name>/`
* The final training episode's actions are always captured in-memory; extend the script if you want to auto-export them.

---

## Early DONE Action
When `--early-done` is enabled the action space includes an explicit termination action. If invoked:
* Environment checks current voxel occupancy ratio or count vs `--early-done-min-voxels`
* If condition satisfied, applies `--early-done-bonus` to reward (can be negative for penalty)
* Episode ends immediately (like an agent-chosen horizon)

Tune this to encourage agents to stop once structural / daylight criteria are met.

---

## Graceful Rhino Restart Interval
`RHINO_RESTART_INTERVAL` is automatically aligned to a multiple of `num_steps` to avoid mid-rollout termination (e.g., 896 steps with `num_steps=128`). Adjust logic in `agent_training.py` if your rollouts change.

---

## Troubleshooting
| Issue | Possible Cause | Fix |
|-------|----------------|-----|
| Hanging on first env step | Rhino not fully started | Increase sleep in `_start_rhino_new_window` or start manually first |
| Checkpoint not found | Wrong filename passed to `--checkpoint-file` | Use exact basename shown in `models/PPO/checkpoints` |
| LR not changing | Using `--lr_schedule constant` | Switch to `linear` or `cosine` |
| Rewards very small / flat | Over-normalization | Inspect raw reward components in env; temporarily disable normalization (edit source) |
| Multiple PS windows | Re-running with old callback alive | Ensure previous training exited; kill stray `powershell.exe` processes |

---

## Extending / Modifying
Common tweaks inside `agent_training.py`:
* Adjust intrinsic reward weights (`str_wt`, `sun_wt`, etc.) in `make_single_env`
* Change rollout length (`num_steps`) & PPO hyperparams
* Add custom callbacks by extending `BaseCallback` and appending to `combined_callback`

---

## License
This project is licensed under the GNU General Public License v3.0 (GPL-3.0).

You are free to use, modify, and redistribute this software under the terms of the GPLv3. Any derivative work or redistribution must also be licensed under GPLv3 and include a copy of the license.

Key points (summary, not a substitute for the full text):
* Source must remain available under the same license when distributed.
* Modifications must be clearly indicated.
* No warranty is provided (AS IS).
* If you distribute binaries, you must provide access to the corresponding source.

See the full license text in the `COPYING` file or at: https://www.gnu.org/licenses/gpl-3.0.html

Copyright (C) 2025 ChristinaX153

---

Happy training!


