from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from voxel_env import VoxelEnv, export_voxel_grid
import os

# Create environment
env = VoxelEnv(grid_size=5)
check_env(env)

# PPO agent with tuned hyperparameters (no tensorboard)
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0001,
    gamma=0.95,
    n_steps=512,
    batch_size=64,
    gae_lambda=0.9,
    ent_coef=0.02,
    vf_coef=0.4,
    max_grad_norm=0.5,
    clip_range=0.2,
)

# Train agent
model.learn(total_timesteps=50000)

# Output folder for voxel states
output_folder = "output_steps"
os.makedirs(output_folder, exist_ok=True)

# Rollout for visualization
obs, info = env.reset()
for step in range(20):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    print(f"Step {step}: Action: {action} | Reward: {reward}")
    env.render()

    filename = os.path.join(output_folder, f"step_{step:03}.json")
    export_voxel_grid(env.grid, filename)

    if done:
        obs, info = env.reset()

print(f"Exported voxel states to folder: {output_folder}")
