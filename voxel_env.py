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

        # Action/obs spaces
        self.action_space = spaces.Discrete(1)  # will be reset dynamically
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(grid_size * grid_size * grid_size,),
            dtype=np.float32
        )

        # Actions are a list of tuples: ("add"|"del", x, y, z)
        self.available_actions = []
        self._update_available_actions()

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

        self.current_epw = None  # set during reset()

    def reset(self, seed=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        self.grid = np.zeros_like(self.grid, dtype=np.int32)

        # New root voxel each episode (cannot delete)
        rx = rng.integers(0, self.grid_size)
        ry = rng.integers(0, self.grid_size)
        self.root_voxel = (int(rx), int(ry), 0)
        self.grid[self.root_voxel] = 1

        self._update_available_actions()

        # Pick EPW for this episode
        self.current_epw = np.random.choice(self.epw_files)
        normalized_epw = self.current_epw.replace("\\", "/")
        print(f"[EPISODE] Using EPW file: {os.path.basename(normalized_epw)}")

        observation = self.grid.flatten().astype(np.float32)
        info = {"epw_file": os.path.basename(self.current_epw)}
        return observation, info

    def step(self, action_idx):
        # Decode action
        op, x, y, z = self.available_actions[action_idx]

        # Log the action
        print(f"[STEP] Action: {op.upper()} voxel at ({x}, {y}, {z})")

        # Execute
        if op == "add":
            reward = self._add_voxel(x, y, z)
        elif op == "del":
            reward = self._delete_voxel(x, y, z)
        else:
            reward = -0.2  # safety fallback

        # Refresh actions and dynamic action space
        self._update_available_actions()
        self.action_space = spaces.Discrete(max(1, len(self.available_actions)))

        # Episode termination logic
        total_voxels = self.grid_size ** 3
        active_voxels = int(np.sum(self.grid))
        terminated = bool(
            active_voxels >= 0.6 * total_voxels or len(self.available_actions) == 0
        )
        truncated = False

        observation = self.grid.flatten().astype(np.float32)
        info = {"epw_file": os.path.basename(self.current_epw)}
        return observation, reward, terminated, truncated, info


    def render(self):
        print(self.grid)

    def _count_neighbors(self, x, y, z):
        neighbor_offsets = np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ])
        neighbor_coords = np.array([x, y, z]) + neighbor_offsets
        valid_mask = (
            (neighbor_coords[:, 0] >= 0) & (neighbor_coords[:, 0] < self.grid_size) &
            (neighbor_coords[:, 1] >= 0) & (neighbor_coords[:, 1] < self.grid_size) &
            (neighbor_coords[:, 2] >= 0) & (neighbor_coords[:, 2] < self.grid_size)
        )
        valid_neighbors = neighbor_coords[valid_mask]
        if len(valid_neighbors) == 0:
            return 0
        neighbor_values = self.grid[valid_neighbors[:, 0], valid_neighbors[:, 1], valid_neighbors[:, 2]]
        return int(np.sum(neighbor_values == 1))

    def _update_available_actions(self):
        """Builds a combined list of add + delete actions."""
        occupied = np.argwhere(self.grid == 1)
        self.available_actions = []

        if len(occupied) == 0:
            return

        # --- ADD actions: empty neighbors of the current shape ---
        neighbor_offsets = np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ])
        potential_neighbors = occupied[:, None, :] + neighbor_offsets[None, :, :]
        potential_neighbors = potential_neighbors.reshape(-1, 3)
        valid_mask = (
            (potential_neighbors[:, 0] >= 0) & (potential_neighbors[:, 0] < self.grid_size) &
            (potential_neighbors[:, 1] >= 0) & (potential_neighbors[:, 1] < self.grid_size) &
            (potential_neighbors[:, 2] >= 0) & (potential_neighbors[:, 2] < self.grid_size)
        )
        valid_neighbors = potential_neighbors[valid_mask]
        if len(valid_neighbors) > 0:
            grid_values = self.grid[valid_neighbors[:, 0], valid_neighbors[:, 1], valid_neighbors[:, 2]]
            empty_mask = (grid_values == 0)
            empty_neighbors = valid_neighbors[empty_mask]
            for x, y, z in set(map(tuple, empty_neighbors)):
                self.available_actions.append(("add", int(x), int(y), int(z)))

        # --- DELETE actions: any occupied voxel except the root voxel ---
        root = self.root_voxel
        for x, y, z in map(tuple, occupied):
            if (x, y, z) != root:
                self.available_actions.append(("del", int(x), int(y), int(z)))

        # Keep a deterministic order (optional but helps reproducibility)
        self.available_actions.sort(key=lambda a: (a[0], a[1], a[2], a[3]))

    def _add_voxel(self, x, y, z):
        if self.grid[x, y, z] == 0:
            self.grid[x, y, z] = 1
            epw_path = self.current_epw
            reward = get_reward_gh(self.grid, self.merged_gh_file, epw_path, self.sun_wt, self.str_wt)
            return reward
        else:
            return -0.2  # invalid add

    def _delete_voxel(self, x, y, z):
        # Prevent deleting the root voxel
        if (x, y, z) == self.root_voxel:
            return -0.2

        if self.grid[x, y, z] == 1:
            self.grid[x, y, z] = 0
            epw_path = self.current_epw
            reward = get_reward_gh(self.grid, self.merged_gh_file, epw_path, self.sun_wt, self.str_wt)
            # Optional: add a small penalty to discourage excessive churn
            # reward -= 0.01
            return reward
        else:
            return -0.2  # invalid delete

    # (unchanged helpers below kept for completeness)
    def _write_grid_to_temp_file(self):
        export_voxel_grid(self.grid, "temp_voxel_input.json")

    def _wait_for_external_reward(self, timeout=30):
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
