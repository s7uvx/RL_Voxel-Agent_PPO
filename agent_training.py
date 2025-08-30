import os
import time
import argparse
import numpy as np
import torch

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
# from sb3_contrib.common.maskable.evaluation import evaluate_policy  # optional

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from gymnasium import wrappers

from voxel_env import VoxelEnv, export_voxel_grid

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
        # Convert numpy arrays to lists for JSON serialization
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
parser.add_argument('--new', action='store_true', default=False, help='Start a new model instead of loading an existing one')
parser.add_argument('--n_envs', type=int, default=1, help='Number of parallel environments (default: 1)')
parser.add_argument('--grid', type=parse_grid, default='(5,5,5)', help='Grid: int like 5 or tuple like 5,5,10 or (5,5,10)')
parser.add_argument('--chk_freq', type=int, default=500, help='Frequency of saving checkpoints (default: 100)')
args = parser.parse_args()

GRID_PARAM = parse_grid(args.grid)

# ---------------------------
# Basic info
# ---------------------------
# gpu_available = torch.cuda.is_available()
# print(f"GPU available: {gpu_available}")
print("Using CPU for PPO training (recommended for MLP policies)")

# ---------------------------
# Mask function for ActionMasker
# ---------------------------
def mask_fn(env):
    # VoxelEnv implements env.action_masks()
    return env.action_masks()

# ---------------------------
# Vec env factory
# ---------------------------
def make_single_env(rank: int = 0):
    def _thunk():
        # give each env a unique port if you scale n_envs
        e = VoxelEnv(port=args.port + rank, grid_size=GRID_PARAM, device='cpu')  # type: ignore[arg-type]
        # wrap with the ActionMasker so the policy only samples valid actions
        return ActionMasker(e, mask_fn)
    return _thunk

num_envs = max(1, args.n_envs)
env = DummyVecEnv([make_single_env(i) for i in range(num_envs)])

# ---------------------------
# PPO hyperparams (stable defaults)
# ---------------------------
num_steps = 256          # rollout length per env 
num_epochs = 1          # PPO epochs per update
batch_size = 64          # must divide (num_steps * num_envs); 1024*k is always divisible by 256

starting_step = 0
model_dir = os.path.join(os.getcwd(), 'models', 'PPO')
os.makedirs(model_dir, exist_ok=True)
existing_models = [f for f in os.listdir(model_dir) if f.endswith('.zip')]

if len(existing_models) > 0 and not args.new:
    latest_model = os.path.splitext(sorted(existing_models, key=lambda x: os.path.getmtime(os.path.join(model_dir, x)))[-1])[0]
    print(f"Loading existing MaskablePPO model: {latest_model}")
    model_path = os.path.join(model_dir, latest_model)
    model = MaskablePPO.load(model_path, env=env, device='cpu')
    starting_step = getattr(model, "_total_timesteps", 0)
else:
    print("Creating a new MaskablePPO model")
    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        gamma=0.99,
        n_steps=num_steps,
        batch_size=batch_size,
        n_epochs=num_epochs,
        gae_lambda=0.95,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        clip_range=0.2,
        policy_kwargs=dict(ortho_init=False),  # helps stability with sparse/heterogeneous rewards
        device='cpu',
        tensorboard_log="./ppo_voxel_tensorboard/"
    )

print(f"Starting training from {starting_step} with {num_envs} parallel env(s) on CPU...")

# Keep the same "one-update" semantics you used before, just larger rollout & batch
new_steps = num_steps * num_epochs
model_date_time = time.strftime("%Y%m%d-%H%M", time.localtime())
save_path = os.path.join(model_dir, f"maskable_ppo_voxel_{model_date_time}_{new_steps+starting_step}")

# ---------------------------
# Checkpointing
# ---------------------------
checkpoint_callback = CheckpointCallback(
    save_freq=args.chk_freq,
    save_path=os.path.join(model_dir, 'checkpoints'),
    name_prefix=f"maskable_ppo_voxel_{model_date_time}_{new_steps+starting_step}"
)

# ---------------------------
# Train
# ---------------------------
model.learn(total_timesteps=new_steps, progress_bar=True, callback=checkpoint_callback)
print("Training completed")

# ---------------------------
# Save the trained model
# ---------------------------
model.save(save_path)
print(f"Model saved as: {save_path}.zip")

# ---------------------------
# Evaluation (mask-aware rollout)
# ---------------------------
print("Starting evaluation...")

max_eval_steps = 3 * num_steps

# Build a single eval env and keep masking on predict-time
raw_eval_env = VoxelEnv(port=args.port, grid_size=GRID_PARAM, device='cpu')  # type: ignore[arg-type]
raw_eval_env = ActionMasker(raw_eval_env, mask_fn)  # ensure masks are applied at sample-time
raw_eval_env = wrappers.TimeLimit(env=raw_eval_env, max_episode_steps=max_eval_steps * 2)
raw_eval_env = Monitor(raw_eval_env, allow_early_resets=True)

output_folder = f"output_steps/model_{model_date_time}"
os.makedirs(output_folder, exist_ok=True)

obs, info = raw_eval_env.reset()
print("Exporting voxel states for visualization...")

for step in range(max_eval_steps):
    # For predict-time, be explicit: recompute mask from the underlying env
    current_mask = raw_eval_env.unwrapped.action_masks()  # type: ignore[attr-defined]
    action_idx, _states = model.predict(obs, deterministic=True, action_masks=current_mask)

    obs, reward, terminated, truncated, info = raw_eval_env.step(action_idx)
    done = bool(terminated or truncated)

    # decode action for logging clarity (handles NO-OP and facade actions)
    action_type, coords_or_params = raw_eval_env.unwrapped._action_to_coords(int(action_idx))  # type: ignore[attr-defined]
    total_actions = raw_eval_env.action_space.n  # type: ignore[attr-defined]
    
    # Create action info for export
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
    elif action_type == "facade":
        param_name, direction_idx, value, direction_name = coords_or_params
        action_info["facade_change"] = {
            "parameter": param_name,
            "direction": direction_name,
            "direction_idx": int(direction_idx),
            "value": int(value)
        }
        print(f"Step {step}: Action Index: {action_idx} | Action: FACADE {param_name}[{direction_name}] = {value} "
              f"| Total Actions: {total_actions} | Reward: {reward}")
    else:
        print(f"Step {step}: Action Index: {action_idx} | Action: {action_type.upper()} "
              f"| Total Actions: {total_actions} | Reward: {reward}")

    if step % 10 == 0:
        raw_eval_env.unwrapped.render()

    # Export enhanced voxel grid with EPW and facade parameters
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
