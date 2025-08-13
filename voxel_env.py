import numpy as np
import os
import json
import torch
import time
from gymnasium import Env, spaces
import compute_rhino3d.Util
import compute_rhino3d.Grasshopper as gh
import gc

from gh_file_runner import get_reward_gh

class VoxelEnv(Env):
    def __init__(self, port, grid_size=5, device=None):
        super(VoxelEnv, self).__init__()
        self.grid_size = grid_size
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.int32)

        # Place root voxel and remember it (cannot be deleted)
        rx, ry = np.random.randint(0, grid_size, size=2)
        self.root_voxel = (rx, ry, 0)
        self.grid[rx, ry, 0] = 1

        # Expanded action space: 2 actions * (grid_size^3) locations
        # Action = action_type * grid_size^3 + location_index
        # action_type: 0=Add, 1=Delete
        # location_index: 0 to (grid_size^3 - 1)
        total_locations = grid_size ** 3
        self.action_space = spaces.Discrete(2 * total_locations)
        
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(grid_size * grid_size * grid_size,),
            dtype=np.float32
        )

        self.timeouts = 0
        self.port = port
        self.merged_gh_file = os.path.join(os.getcwd(), 'gh_files', 'RL_Voxel_V5_hops.ghx')
        compute_rhino3d.Util.url = f"http://localhost:{self.port}/"
        self.sun_wt = 0.3
        self.str_wt = 0.7

        self.epw_folder = os.path.join(os.getcwd(), 'gh_files', 'epw')
        self.epw_files = [
            os.path.join(self.epw_folder, f)
            for f in os.listdir(self.epw_folder)
            if f.lower().endswith(".epw")
        ]
        if not self.epw_files:
            raise FileNotFoundError(f"No EPW files found in {self.epw_folder}")

        self.current_epw = None

    def _action_to_coords(self, action_idx):
        """Convert action index to action type and 3D coordinates"""
        total_locations = self.grid_size ** 3
        
        if action_idx < total_locations:
            # Add action (0 to grid_size^3 - 1)
            action_type = "add"
            location_idx = action_idx
        else:
            # Delete action (grid_size^3 to 2*grid_size^3 - 1)
            action_type = "delete"
            location_idx = action_idx - total_locations
        
        # Convert location index to 3D coordinates
        z = location_idx // (self.grid_size ** 2)
        y = (location_idx % (self.grid_size ** 2)) // self.grid_size
        x = location_idx % self.grid_size
        
        return action_type, (x, y, z)

    def reset(self, seed=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        self.grid = np.zeros_like(self.grid, dtype=np.int32)

        # New root voxel each episode (cannot delete)
        rx = rng.integers(0, self.grid_size)
        ry = rng.integers(0, self.grid_size)
        self.root_voxel = (int(rx), int(ry), 0)
        self.grid[self.root_voxel] = 1

        # Pick EPW for this episode
        self.current_epw = np.random.choice(self.epw_files)
        normalized_epw = self.current_epw.replace("\\", "/")
        print(f"[EPISODE] Using EPW file: {os.path.basename(normalized_epw)}")

        observation = self.grid.flatten().astype(np.float32)
        info = {"epw_file": os.path.basename(self.current_epw)}
        return observation, info

    def step(self, action_idx):
        action_type, (x, y, z) = self._action_to_coords(action_idx)
        
        if action_type == "add":
            reward = self._add_voxel(x, y, z)
            if reward > -0.1:  # Successful
                print(f"[STEP] ✅ ADD at ({x},{y},{z}) - SUCCESS")
            else:  # Failed
                print(f"[STEP] ❌ ADD at ({x},{y},{z}) - FAILED (reward: {reward})")
        elif action_type == "delete":
            reward = self._delete_voxel(x, y, z)
            if reward > -0.1:  # Successful
                print(f"[STEP] ✅ DELETE at ({x},{y},{z}) - SUCCESS")
            else:  # Failed
                print(f"[STEP] ❌ DELETE at ({x},{y},{z}) - FAILED (reward: {reward})")
        else:
            reward = -0.5
            print(f"[STEP] ❌ INVALID action index: {action_idx}")

        # Episode termination logic
        total_voxels = self.grid_size ** 3
        active_voxels = int(np.sum(self.grid))
        terminated = bool(active_voxels >= 0.6 * total_voxels or active_voxels <= 1)
        truncated = False

        observation = self.grid.flatten().astype(np.float32)
        info = {"epw_file": os.path.basename(self.current_epw)}
        return observation, reward, terminated, truncated, info

    def _add_voxel(self, x, y, z):
        """Add a voxel at specific coordinates with validation"""
        # Check if coordinates are valid
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size and 0 <= z < self.grid_size):
            return -0.5  # Invalid coordinates
        
        # Check if location is already occupied
        if self.grid[x, y, z] == 1:
            return -0.3  # Already occupied
        
        # Check if location is adjacent to existing structure (connectivity rule)
        if not self._is_adjacent_to_structure(x, y, z):
            return -0.4  # Not connected to existing structure
        
        print(f"[ADD] Adding voxel at ({x}, {y}, {z})")
        self.grid[x, y, z] = 1
        
        # Get reward from Grasshopper
        epw_path = self.current_epw
        reward = get_reward_gh(self.grid, self.merged_gh_file, epw_path, self.sun_wt, self.str_wt)
        return reward

    def _delete_voxel(self, x, y, z):
        """Delete a voxel at specific coordinates with validation"""
        # Check if coordinates are valid
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size and 0 <= z < self.grid_size):
            return -0.5  # Invalid coordinates
        
        # Check if location is occupied
        if self.grid[x, y, z] == 0:
            return -0.3  # Nothing to delete
        
        # Check if it's the root voxel (cannot delete)
        if (x, y, z) == self.root_voxel:
            return -0.4  # Cannot delete root voxel
        
        print(f"[DELETE] Removing voxel at ({x}, {y}, {z})")
        self.grid[x, y, z] = 0
        
        # Get reward from Grasshopper
        epw_path = self.current_epw
        reward = get_reward_gh(self.grid, self.merged_gh_file, epw_path, self.sun_wt, self.str_wt)
        return reward

    def _is_adjacent_to_structure(self, x, y, z):
        """Check if position (x,y,z) is adjacent to existing structure"""
        neighbor_offsets = [
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1)
        ]
        
        for dx, dy, dz in neighbor_offsets:
            nx, ny, nz = x + dx, y + dy, z + dz
            if (0 <= nx < self.grid_size and 
                0 <= ny < self.grid_size and 
                0 <= nz < self.grid_size):
                if self.grid[nx, ny, nz] == 1:
                    return True
        return False

    # Remove the old random methods
    def _add_random_voxel(self):
        """DEPRECATED: Use _add_voxel instead"""
        pass

    def _delete_random_voxel(self):
        """DEPRECATED: Use _delete_voxel instead"""
        pass

    def render(self):
        print(self.grid)

    def _write_grid_to_temp_file(self):
        export_voxel_grid(self.grid, "temp_voxel_input.json")

    def _wait_for_external_reward(self, timeout=30):
        # Keep this method for compatibility if needed elsewhere
        global timeouts
        reward_file = "temp_reward.json"
        start_time = time.time()
        while True:
            if os.path.exists(reward_file):
                try:
                    with open(reward_file, 'r') as f:
                        content = f.read().strip()
                        if not content:
                            raise ValueError("Empty file")
                        data = json.loads(content)
                    for _ in range(10):
                        try:
                            os.remove(reward_file)
                            break
                        except PermissionError:
                            time.sleep(0.03)
                    return float(data.get("reward", 0.0))
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"⏳ Waiting for valid reward file... ({e})")
                    time.sleep(0.1)
            else:
                if time.time() - start_time > timeout:
                    self.timeouts += 1
                    print("❌ Timeout: No external reward received. {} times".format(self.timeouts))
                    return 0.0
                time.sleep(0.1)


def export_voxel_grid(grid, filename):
    data = {"voxels": grid.tolist()}
    with open(filename, 'w') as f:
        json.dump(data, f)


class VectorizedVoxelEnv:
    def __init__(self, num_envs=8, grid_size=5, device=None):
        self.num_envs = num_envs
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.envs = [VoxelEnv(port=6500 + i, grid_size=grid_size, device=self.device) for i in range(num_envs)]
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space

    def reset(self, seed=None):
        observations = []
        infos = []
        for i, env in enumerate(self.envs):
            env_seed = seed + i if seed is not None else None
            obs, info = env.reset(seed=env_seed)
            observations.append(obs)
            infos.append(info)
        return np.array(observations, dtype=np.float32), infos

    def step(self, actions):
        observations = []
        rewards = []
        terminated_list = []
        truncated_list = []
        infos = []
        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(action)
            observations.append(obs)
            rewards.append(reward)
            terminated_list.append(terminated)
            truncated_list.append(truncated)
            infos.append(info)
        return (np.array(observations, dtype=np.float32),
                np.array(rewards, dtype=np.float32),
                np.array(terminated_list, dtype=bool),
                np.array(truncated_list, dtype=bool),
                infos)

    def render(self):
        self.envs[0].render()
