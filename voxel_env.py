import numpy as np
from gymnasium import Env, spaces


class VoxelEnv(Env):
    def __init__(self, grid_size=5):
        super(VoxelEnv, self).__init__()
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.int32)

        self.action_space = spaces.Discrete(1)  

        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(grid_size * grid_size * grid_size,),
            dtype=np.int32
        )

        self.available_actions = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros_like(self.grid, dtype=np.int32)

        center = self.grid_size // 2
        self.grid[center, center, center] = 1
        self._update_available_actions()

        observation = self.grid.flatten().astype(np.int32)
        info = {}
        return observation, info

    def step(self, action_idx):
        x, y, z = self.available_actions[action_idx]
        reward = self._calculate_reward(x, y, z)

        self._update_available_actions()

        total_voxels = self.grid_size ** 3
        active_voxels = np.sum(self.grid)

        terminated = bool(
            active_voxels >= 0.6 * total_voxels or len(self.available_actions) == 0
        )
        truncated = False
        info = {}

        observation = self.grid.flatten().astype(np.int32)
        self.action_space = spaces.Discrete(max(1, len(self.available_actions)))

        return observation, reward, terminated, truncated, info

    def render(self):
        print(self.grid)

    def _count_neighbors(self, x, y, z):
        count = 0
        neighbors = [
            (x + 1, y, z), (x - 1, y, z),
            (x, y + 1, z), (x, y - 1, z),
            (x, y, z + 1), (x, y, z - 1)
        ]
        for nx, ny, nz in neighbors:
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and 0 <= nz < self.grid_size:
                if self.grid[nx, ny, nz] == 1:
                    count += 1
        return count

    def _update_available_actions(self):
        possible = set()

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                for z in range(self.grid_size):
                    if self.grid[x, y, z] == 1:
                        neighbors = [
                            (x + 1, y, z), (x - 1, y, z),
                            (x, y + 1, z), (x, y - 1, z),
                            (x, y, z + 1), (x, y, z - 1)
                        ]
                        for nx, ny, nz in neighbors:
                            if (
                                0 <= nx < self.grid_size and
                                0 <= ny < self.grid_size and
                                0 <= nz < self.grid_size and
                                self.grid[nx, ny, nz] == 0
                            ):
                                possible.add((nx, ny, nz))

        self.available_actions = list(possible)

    def _calculate_reward(self, x, y, z):
        reward = 0.0

        if self.grid[x, y, z] == 0:
            self.grid[x, y, z] = 1
            num_neighbors = self._count_neighbors(x, y, z)

            if num_neighbors == 1:
                reward += 1.0
            elif 2 <= num_neighbors <= 3:
                reward += 1.5
            elif 4 <= num_neighbors <= 6:
                reward += 2.0
            else:
                reward -= 0.5
        else:
            reward -= 0.2

        total_voxels = self.grid_size ** 3
        if np.sum(self.grid) >= 0.5 * total_voxels:
            reward += 10.0

        return reward


def export_voxel_grid(grid, filename):
    import json
    data = {"voxels": grid.tolist()}
    with open(filename, 'w') as f:
        json.dump(data, f)
