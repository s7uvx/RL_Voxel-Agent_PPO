from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from voxel_env import VoxelEnv, VectorizedVoxelEnv, export_voxel_grid
import torch
import os

# Check for GPU availability (for potential future CNN policies)
gpu_available = torch.cuda.is_available()
print(f"GPU available: {gpu_available}")
print("Using CPU for PPO training (recommended for MLP policies)")

# Create vectorized environment
def make_env():
    return VoxelEnv(grid_size=6, device='cpu')  # Use CPU for environments too

# Use vectorized environment for better performance
num_envs = 1  # Can use more envs on CPU since we're not GPU-limited
env = make_vec_env(make_env, n_envs=num_envs, vec_env_cls=DummyVecEnv)

# Configure PPO with CPU (optimal for MLP policies)
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    gamma=0.95,
    n_steps=2048,
    batch_size=128,
    gae_lambda=0.9,
    ent_coef=0.1,
    vf_coef=0.4,
    max_grad_norm=0.5,
    clip_range=0.2,
    device='cpu',  # Explicitly use CPU for better MLP performance
    tensorboard_log="./ppo_voxel_tensorboard/"  # Add tensorboard logging
)

print(f"Starting training with {num_envs} parallel environments on CPU...")
model.learn(total_timesteps=2000, progress_bar=True)
print("Training completed")

# Save the trained model
model.save("ppo_voxel_model")

# Post-training evaluation with single environment
print("Starting evaluation...")
eval_env = VoxelEnv(grid_size=6, device='cpu')  # Use CPU for evaluation too
output_folder = "output_steps_vec"
os.makedirs(output_folder, exist_ok=True)

obs, info = eval_env.reset()

print("Exporting voxel states for visualization...")
for step in range(101):  # Reduced steps for faster evaluation
    action_idx, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action_idx)
    done = terminated or truncated

    print(f"Step {step}: Action Index: {action_idx} | Available: {len(eval_env.available_actions)} | Reward: {reward}")
    if step % 10 == 0:  # Print grid every 10 steps to reduce output
        eval_env.render()

    filename = os.path.join(output_folder, f"step_{step:03}.json")
    export_voxel_grid(eval_env.grid, filename)

    if done:
        obs, info = eval_env.reset()

print(f"Exported voxel states to: {output_folder}")
print(f"Model saved as: ppo_voxel_model.zip")
print(f"Tensorboard logs saved to: ./ppo_voxel_tensorboard/")
