from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from voxel_env import VoxelEnv, export_voxel_grid
import torch
import os
import time
import numpy as np
import argparse

# Add argument parser for port configuration
parser = argparse.ArgumentParser(description='Train PPO agent for voxel environment')
parser.add_argument('--port', type=int, default=81, help='Port number for VoxelEnv (default: 81)')
parser.add_argument('--new', action='store_true', default=False, help='Start training a new model instead of loading existing one (default: False)')
args = parser.parse_args()

# Check for GPU availability (for potential future CNN policies)
gpu_available = torch.cuda.is_available()
print(f"GPU available: {gpu_available}")
print("Using CPU for PPO training (recommended for MLP policies)")
# num_env = 0
# port_list = [6501,6001,6002,6003,6004,6005]
# Create vectorized environment
def make_env():
    # global num_env, port_list
    env = VoxelEnv(port=args.port, grid_size=5, device='cpu')  # Use CPU for environments too
    # num_env+=1
    return env

# Use vectorized environment for better performance
num_envs = 1  # Can use more envs on CPU since we're not GPU-limited
env = make_vec_env(make_env, n_envs=num_envs, vec_env_cls=DummyVecEnv, seed=np.random.randint(0,666))
num_steps = 74
num_epochs = 1000
# Configure PPO with CPU (optimal for MLP policies)

starting_step = 0
model_dir = os.path.join(os.getcwd(),'models','PPO')
os.makedirs(model_dir, exist_ok = True)
existing_models = os.listdir(model_dir)
if len(existing_models) > 0 and not args.new:
    latest_model = os.path.splitext(sorted(existing_models, key=lambda x: os.path.getmtime(os.path.join(model_dir, x)))[-1])[0]
    model = PPO.load(os.path.join(model_dir,latest_model), env=env, device='cpu')
    starting_step = model._total_timesteps
else:
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        gamma=0.95,
        n_steps=num_steps,
        batch_size=1,
        n_epochs=num_epochs,
        gae_lambda=0.9,
        ent_coef=0.1,
        vf_coef=0.4,
        max_grad_norm=0.5,
        clip_range=0.2,
        stats_window_size=10,
        device='cpu',  # Explicitly use CPU for better MLP performance
        tensorboard_log="./ppo_voxel_tensorboard/"  # Add tensorboard logging
    )

print(f"Starting training from {starting_step} with {num_envs} parallel environments on CPU...")

new_steps = num_steps*num_epochs
model.learn(total_timesteps=new_steps, progress_bar=True)
print("Training completed")

# Save the trained model
model_date_time = time.strftime("%Y%m%d-%H%M",time.localtime())
model.save(os.path.join(model_dir,f"ppo_voxel_model_{model_date_time}_{new_steps+starting_step}"))

# Post-training evaluation with single environment
print("Starting evaluation...")
eval_env = VoxelEnv(port=args.port, grid_size=5, device='cpu')  # Use CPU for evaluation too
output_folder = f"output_steps/model_{model_date_time}"
os.makedirs(output_folder, exist_ok=True)

obs, info = eval_env.reset()

print("Exporting voxel states for visualization...")
for step in range(148):  # Reduced steps for faster evaluation
    action_idx, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action_idx)
    done = terminated or truncated

    # Fixed: Remove reference to available_actions since we now have fixed action space
    action_type, (x, y, z) = eval_env._action_to_coords(action_idx)
    total_actions = eval_env.action_space.n
    print(f"Step {step}: Action Index: {action_idx} | Action: {action_type.upper()} at ({x},{y},{z}) | Total Actions: {total_actions} | Reward: {reward}")
    
    if step % 10 == 0:  # Print grid every 10 steps to reduce output
        eval_env.render()

    filename = os.path.join(output_folder, f"step_{step:03}.json")
    export_voxel_grid(eval_env.grid, filename)

    if done:
        obs, info = eval_env.reset()

print(f"Exported voxel states to: {output_folder}")
print("Model saved as: {}".format(os.path.join(model_dir,f"ppo_voxel_model_{model_date_time}_{new_steps+starting_step}")))
print(f"Tensorboard logs saved to: ./ppo_voxel_tensorboard/")

# The training code should work as-is since we simplified the action space
# The agent will now only choose between:
# Action 0: Add a voxel at a random valid location
# Action 1: Delete a random voxel (except root)

print(f"Action space: {env.action_space}")  # Should show Discrete(250) for 5x5x5 grid
