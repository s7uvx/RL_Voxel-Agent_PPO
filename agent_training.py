import os
import time
import argparse
import numpy as np
import torch
import math

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
# from sb3_contrib.common.maskable.evaluation import evaluate_policy  # optional

from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from gymnasium import wrappers

from voxel_env import VoxelEnv, export_voxel_grid

# ---------------------------
# Utility functions
# ---------------------------

# Parse grid argument to int or 3-tuple[int,int,int]
def parse_grid(grid_str: str):
    s = str(grid_str).strip()
    s = s.replace('[', '(').replace(']', ')')
    # pure int
    if s.isdigit():
        return int(s)
    # strip parens if present
    if s.startswith('(') and s.endswith(')'):
        s = s[1:-1]
    parts = [p.strip() for p in s.split(',') if p.strip()]
    if len(parts) == 1 and parts[0].isdigit():
        return int(parts[0])
    if len(parts) == 3 and all(p.lstrip('-').isdigit() for p in parts):
        return tuple(map(int, parts))
    raise ValueError(f"Invalid --grid value: {grid_str}. Use 5 or 5,5,10 or (5,5,10)")

# Enhanced export function with EPW and facade parameters
def export_voxel_grid_enhanced(grid, filename, epw_file=None, facade_params=None, action_info=None):
    """Export voxel grid with additional metadata"""
    data = {
        "voxels": grid.tolist(),
        "epw_file": os.path.basename(epw_file) if epw_file else None,
        "facade_params": {}
    }

    # Add facade parameters if available
    if facade_params:
        for key, value in facade_params.items():
            if hasattr(value, 'tolist'):
                data["facade_params"][key] = value.tolist()
            else:
                data["facade_params"][key] = value

    # Add action information if available
    if action_info:
        data["action_info"] = action_info

    with open(filename, 'w') as f:
        import json
        json.dump(data, f, indent=2)

# ---------------------------
# CLI args
# ---------------------------
parser = argparse.ArgumentParser(description='Train MaskablePPO agent for voxel environment')
parser.add_argument('--port', type=int, default=81, help='Base port number for VoxelEnv (default: 81)')
parser.add_argument('--new', action='store_true', default=False,
                    help='Start a brand new model (ignore checkpoints and saved models)')
parser.add_argument('--checkpoint', action='store_true', default=False,
                    help='Resume training from the most recent checkpoint in models/PPO/checkpoints/')
parser.add_argument('--n_envs', type=int, default=1, help='Number of parallel environments (default: 1)')
parser.add_argument('--grid', type=parse_grid, default='(5,5,5)',
                    help='Grid: int like 5 or tuple like 5,5,10 or (5,5,10)')
parser.add_argument('--chk_freq', type=int, default=500,
                    help='Frequency of saving checkpoints (default: 500)')
parser.add_argument('--timesteps', type=int, default=100000,
                    help='Total environment steps to train (default: 100000)')
parser.add_argument('--lr', type=float, default=3e-3, help='Initial learning rate')
parser.add_argument('--lr_final', type=float, default=3e-5, help='Final learning rate at end of training')
parser.add_argument('--lr_schedule', type=str, choices=['constant', 'linear', 'cosine'], default='cosine',
                    help='LR schedule over progress_remaining (1→0)')
parser.add_argument('--early-done', action='store_true', default=False,
                    help='Enable an explicit DONE action for early termination')
parser.add_argument('--early-done-bonus', type=float, default=0.0,
                    help='Bonus (or penalty if negative) when DONE is used meeting min occupancy')
parser.add_argument('--early-done-min-voxels', type=float, default=0.1,
                    help='Minimum fraction (<=1) or absolute count (>1) of active voxels required for DONE bonus')
args = parser.parse_args()

GRID_PARAM = parse_grid(args.grid)

# ---------------------------
# Basic info
# ---------------------------
print("Using CPU for PPO training (recommended for MLP policies)")

# ---------------------------
# Mask function for ActionMasker
# ---------------------------
def mask_fn(env):
    # return env.action_masks()
    return env.unwrapped.action_masks()

# ---------------------------
# Vec env factory
# ---------------------------
def make_single_env(rank: int = 0):
    def _thunk():
        e = VoxelEnv(
            port=args.port + rank,
            grid_size=GRID_PARAM,
            device='cpu',
            str_wt=2.0,
            sun_wt=0.4,
            wst_wt=0.005,
            cst_wt=0.05,
            day_wt=0.4,
            early_done=args.early_done,
            early_done_bonus=args.early_done_bonus,
            early_done_min_voxels=args.early_done_min_voxels
        )  # type: ignore[arg-type]
        e = wrappers.TimeLimit(env=e, max_episode_steps=256)
        e = ActionMasker(e, mask_fn)
        
        return e
    return _thunk

num_envs = max(1, args.n_envs)
env = DummyVecEnv([make_single_env(i) for i in range(num_envs)])
# Log raw (unnormalized) episode rewards
env = VecMonitor(env)
# Normalize rewards seen by the agent (zero-mean, unit-variance)
env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

# ---------------------------
# PPO hyperparams
# ---------------------------
num_steps = 128          # rollout length per env 
num_epochs = 5           # PPO epochs per update
batch_size = 32          # must divide (num_steps * num_envs); 1024*k is always divisible by 256

starting_step = 0
model_dir = os.path.join(os.getcwd(), 'models', 'PPO')
checkpoint_dir = os.path.join(model_dir, "checkpoints")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# ---------------------------
# Learning rate schedule
# ---------------------------
def _make_lr_schedule(kind: str, lr_start: float, lr_end: float):
    if kind == 'constant':
        return lambda progress_remaining: lr_start
    if kind == 'linear':
        # progress_remaining: 1 → 0, so lr: start → end
        return lambda progress_remaining: lr_end + (lr_start - lr_end) * progress_remaining
    if kind == 'cosine':
        # smooth decay start → end
        return lambda progress_remaining: lr_end + 0.5 * (lr_start - lr_end) * (1.0 + math.cos(math.pi * (1.0 - progress_remaining)))
    return lambda progress_remaining: lr_start

# Build schedule once
lr_schedule = _make_lr_schedule(args.lr_schedule, args.lr, args.lr_final)
norm_path = os.path.join(model_dir, "vecnormalize.pkl")
# If resuming, load running stats
if os.path.exists(norm_path):
    env = VecNormalize.load(norm_path, env)
    env.training = True
    env.norm_reward = True
# ---------------------------
# Model loading
# ---------------------------
if args.checkpoint:
    existing_checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".zip")]
    if not existing_checkpoints:
        raise FileNotFoundError("No checkpoints found in models/PPO/checkpoints/. Cannot resume.")
    latest_ckpt = max(existing_checkpoints,
                      key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)))
    print(f"Resuming from checkpoint: {latest_ckpt}")
    model_path = os.path.join(checkpoint_dir, latest_ckpt)
    model = MaskablePPO.load(model_path, env=env, device="cpu")
    # override lr schedule when resuming
    model.lr_schedule = lr_schedule
elif not args.new:
    existing_models = [f for f in os.listdir(model_dir) if f.endswith(".zip")]
    if existing_models:
        latest_model = os.path.splitext(sorted(existing_models, key=lambda x: os.path.getmtime(os.path.join(model_dir, x)))[-1])[0]
        print(f"Loading existing MaskablePPO model: {latest_model}")
        model = MaskablePPO.load(os.path.join(model_dir, latest_model), env=env, device='cpu')
        starting_step = model._total_timesteps if hasattr(model, "_total_timesteps") else 0
        model.lr_schedule = lr_schedule
    else:
        print("No saved models found. Starting fresh.")
        args.new = True

if args.new:
    print("Creating a new MaskablePPO model")
    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=lr_schedule,   # <-- schedule here
        gamma=0.99,
        n_steps=num_steps,
        batch_size=batch_size,
        n_epochs=num_epochs,
        gae_lambda=0.95,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        clip_range=0.2,
        policy_kwargs=dict(ortho_init=False),
        device="cpu",
        tensorboard_log="./ppo_voxel_tensorboard/"
    )
    starting_step = 0

# ---------------------------
# Training setup
# ---------------------------
print(f"Starting training from {starting_step} with {num_envs} parallel env(s) "
      f"({'early-done ON' if args.early_done else 'early-done OFF'}) on CPU...")
total_steps = args.timesteps
model_date_time = time.strftime("%Y%m%d-%H%M", time.localtime())

save_path = os.path.join(model_dir,
                         f"maskable_ppo_voxel_{model_date_time}_{starting_step+total_steps}")

checkpoint_callback = CheckpointCallback(
    save_freq=args.chk_freq,
    save_path=checkpoint_dir,
    name_prefix=f"maskable_ppo_voxel_{model_date_time}"
)

# ---------------------------
# Train
# ---------------------------
model.learn(
    total_timesteps=total_steps,
    progress_bar=True,
    callback=checkpoint_callback,
    reset_num_timesteps=False,
    tb_log_name=f"MaskablePPO_lr-{args.lr_schedule}"
)

# Save VecNormalize stats so evaluation uses the same scaling
env.save(norm_path)
print(f"Saved VecNormalize stats: {norm_path}")
print("Training completed")

# ---------------------------
# Save the trained model
# ---------------------------
final_step = int(getattr(model, "num_timesteps", starting_step+total_steps))
final_save_path = os.path.join(model_dir,
                               f"maskable_ppo_voxel_{model_date_time}_{final_step}")
model.save(final_save_path)
print(f"Model saved as: {final_save_path}.zip")

# ---------------------------
# Evaluation
# ---------------------------
print("Starting evaluation...")

max_eval_steps = num_steps
raw_eval_env = VoxelEnv(port=args.port, grid_size=GRID_PARAM, device='cpu')
raw_eval_env = wrappers.TimeLimit(env=raw_eval_env, max_episode_steps=max_eval_steps * 2)
raw_eval_env = ActionMasker(raw_eval_env, mask_fn)
eval_env = DummyVecEnv([lambda: raw_eval_env])
eval_env = VecMonitor(eval_env)
if os.path.exists(norm_path):
    eval_env = VecNormalize.load(norm_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False  # report raw rewards

output_folder = f"output_steps/model_{model_date_time}"
os.makedirs(output_folder, exist_ok=True)

obs, info = raw_eval_env.reset()
print("Exporting voxel states for visualization...")

for step in range(max_eval_steps):
    current_mask = raw_eval_env.unwrapped.action_masks()  # type: ignore[attr-defined]
    action_idx, _states = model.predict(obs, deterministic=True, action_masks=current_mask)

    obs, reward, terminated, truncated, info = raw_eval_env.step(action_idx)
    done = bool(terminated or truncated)

    voxel_part, facade_part = raw_eval_env.unwrapped._decode_combined_action(int(action_idx))  # <-- patched
    action_type, coords_or_params = voxel_part
    total_actions = raw_eval_env.action_space.n  # type: ignore[attr-defined]

    action_info = {
        "step": step,
        "action_idx": int(action_idx),
        "action_type": action_type,
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated)
    }

    if action_type in ["add", "delete"]:
        x, y, z = coords_or_params
        action_info["coordinates"] = [int(x), int(y), int(z)]
        print(f"Step {step}: Action Index: {action_idx} | Action: {action_type.upper()} "
              f"at ({x},{y},{z}) | Total Actions: {total_actions} | Reward: {reward}")
    elif action_type == "noop":
        print(f"Step {step}: Action Index: {action_idx} | Action: NOOP "
              f"| Total Actions: {total_actions} | Reward: {reward}")
    elif facade_part is not None:
        _, (param_name, direction_idx, value, direction_name) = facade_part
        action_info["facade_change"] = {
            "parameter": param_name,
            "direction": direction_name,
            "direction_idx": int(direction_idx),
            "value": int(value)
        }
        print(f"Step {step}: Action Index: {action_idx} | Action: FACADE {param_name}[{direction_name}] = {value} "
              f"| Total Actions: {total_actions} | Reward: {reward}")
    elif action_type == "done":
        print(f"Step {step}: Action Index: {action_idx} | Action: DONE | Reward: {reward}")
        action_info["early_done"] = True
    else:
        print(f"Step {step}: Action Index: {action_idx} | Action: INVALID "
              f"| Total Actions: {total_actions} | Reward: {reward}")

    if step % 10 == 0:
        raw_eval_env.unwrapped.render()

    filename = os.path.join(output_folder, f"step_{step:03}.json")
    export_voxel_grid_enhanced(
        raw_eval_env.unwrapped.grid,  # type: ignore[attr-defined]
        filename,
        epw_file=raw_eval_env.unwrapped.current_epw,  # type: ignore[attr-defined]
        facade_params=raw_eval_env.unwrapped.current_facade_params,  # type: ignore[attr-defined]
        action_info=action_info
    )

    if done:
        obs, info = raw_eval_env.reset()


print(f"Exported voxel states to: {output_folder}")
print(f"Tensorboard logs: ./ppo_voxel_tensorboard/")
print(f"Action space (masked at runtime, includes NO-OP): {env.action_space}")
