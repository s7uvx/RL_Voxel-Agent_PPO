import numpy as np
from gymnasium import Env, spaces


class VoxelEnv(Env):
    def __init__(self, grid_size=5):
        super(VoxelEnv, self).__init__()
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.int32)

        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(grid_size * grid_size * grid_size,),
            dtype=np.int32
        )
        self.action_space = spaces.MultiDiscrete([grid_size, grid_size, grid_size])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros_like(self.grid, dtype=np.int32)
        observation = self.grid.flatten().astype(np.int32)
        info = {}
        return observation, info

    def step(self, action):
        x, y, z = action
        if self.grid[x, y, z] == 0:
            self.grid[x, y, z] = 1

        reward = self._calculate_reward()

        total_voxels = self.grid_size ** 3
        active_voxels = np.sum(self.grid)
        terminated = bool(active_voxels >= 0.6 * total_voxels)  # Ensure Python bool
        truncated = False
        info = {}

        observation = self.grid.flatten().astype(np.int32)
        return observation, reward, terminated, truncated, info

    def render(self):
        print(self.grid)

    def _calculate_reward(self):
        # Encourage more active voxels overall
        current_active_voxels = np.sum(self.grid)
        reward = float(current_active_voxels) * 0.1
        return reward


def export_voxel_grid(grid, filename):
    import json
    data = {"voxels": grid.tolist()}
    with open(filename, 'w') as f:
        json.dump(data, f)
