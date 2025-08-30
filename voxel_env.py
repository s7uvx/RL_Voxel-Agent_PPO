import numpy as np
import os
import json
import torch
import time
from gymnasium import Env, spaces
import compute_rhino3d.Util
import compute_rhino3d.Grasshopper as gh
import gc
from typing import Deque, Tuple, cast, Callable, Iterable, Sequence, Optional

from gh_file_runner import get_reward_gh


class VoxelEnv(Env):
    """
    Voxel placement environment with action masking, safety NO-OP action, and facade options.

    Actions:
        0 .. (N-1):           ADD voxel at linearized location
        N .. (2N-1):          DELETE voxel at linearized location
        2N (NOOP_IDX):        NO-OP (always legal, small penalty)
        
        Facade actions (4 directions × 4 parameters × 7 values each):
        (2N+1) .. (2N+112):   Facade parameter adjustments
        - Actions 2N+1 to 2N+28:   min_cols for [east, north, west, south] (7 values each)
        - Actions 2N+29 to 2N+56:  max_cols for [east, north, west, south] (7 values each)
        - Actions 2N+57 to 2N+84:  min_rows for [east, north, west, south] (7 values each)
        - Actions 2N+85 to 2N+112: max_rows for [east, north, west, south] (7 values each)

    Where N = gx * gy * gz (grid_size if int, or product of a 3D tuple)
    """
    def __init__(self, port, grid_size=5, device=None, max_steps=200, cooldown_steps=3, step_penalty=0.0,
                 num_repulsors: int = 3, repulsor_wt: float = 0.2, repulsor_radius: float = 2.0,
                 repulsor_provider: Optional[Callable[["VoxelEnv", np.random.Generator], Iterable[Sequence[float]]]] = None):
        super(VoxelEnv, self).__init__()
        # Normalize grid_size to 3D dims (gx, gy, gz)
        if isinstance(grid_size, (list, tuple, np.ndarray)):
            dims = [int(v) for v in grid_size]
            if len(dims) == 1:
                dims = [dims[0], dims[0], dims[0]]
            if len(dims) != 3:
                raise ValueError("grid_size must be an int or a 3-sequence of ints")
            gx, gy, gz = dims
        else:
            gx = gy = gz = int(grid_size)

        if gx <= 0 or gy <= 0 or gz <= 0:
            raise ValueError("grid dimensions must be positive")

        self.gx, self.gy, self.gz = int(gx), int(gy), int(gz)
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.grid = np.zeros((self.gx, self.gy, self.gz), dtype=np.int32)

        # Place root voxel and remember it (cannot be deleted)
        rx = np.random.randint(0, self.gx)
        ry = np.random.randint(0, self.gy)
        self.root_voxel = (rx, ry, 0)
        self.grid[rx, ry, 0] = 1

        # Action space with NO-OP and facade actions
        total_locations = int(self.gx * self.gy * self.gz)
        self.total_locations = total_locations
        self.NOOP_IDX = 2 * total_locations
        
        # Facade parameters: 4 directions × 4 parameters × 7 values = 112 actions
        self.FACADE_ACTIONS = 4 * 4 * 7  # 112 facade actions
        self.action_space = spaces.Discrete(self.NOOP_IDX + 1 + self.FACADE_ACTIONS)
        
        # Facade parameter ranges and current values
        self.facade_min_val = 1
        self.facade_max_val = 7
        self.current_facade_params = {
            'min_cols': np.array([1, 1, 1, 1], dtype=np.int32),  # [east, north, west, south]
            'max_cols': np.array([7, 7, 7, 7], dtype=np.int32),
            'min_rows': np.array([1, 1, 1, 1], dtype=np.int32),
            'max_rows': np.array([7, 7, 7, 7], dtype=np.int32)
        }

        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(total_locations,),
            dtype=np.float32
        )

        # Episode/time-limit + anti-churn
        self.max_steps = max_steps
        self.step_count = 0
        self.cooldown_steps = cooldown_steps
        self._added_cooldown = {}    # forbid DELETE here for N steps
        self._deleted_cooldown = {}  # forbid ADD here for N steps
        self.step_penalty = step_penalty  # tiny cost per step (optional)

        self.timeouts = 0
        self.port = port
        self.merged_gh_file = os.path.join(os.getcwd(), 'gh_files', 'RL_Voxel_V6_hops.ghx')
        compute_rhino3d.Util.url = f"http://localhost:{self.port}/"
        self.sun_wt = 0.3
        self.str_wt = 0.7
        self.cst_wt = 0.1
        self.wst_wt = 0.1
        # Repulsor config
        self.num_repulsors = num_repulsors
        self.repulsor_wt = repulsor_wt
        self.repulsor_radius = repulsor_radius
        self.repulsor_provider = repulsor_provider

        self.epw_folder = os.path.join(os.getcwd(), 'gh_files', 'epw')
        self.epw_files = [
            os.path.join(self.epw_folder, f)
            for f in os.listdir(self.epw_folder)
            if f.lower().endswith(".epw")
        ]
        if not self.epw_files:
            raise FileNotFoundError(f"No EPW files found in {self.epw_folder}")

        self.current_epw = None
        self.repulsors = []

    # ---------------------------
    # Utilities
    # ---------------------------
    def _idx_to_coords(self, location_idx):
        area = self.gx * self.gy
        z = location_idx // area
        rem = location_idx % area
        y = rem // self.gx
        x = rem % self.gx
        return x, y, z

    def _coords_to_idx(self, x, y, z):
        return int(x + y * self.gx + z * (self.gx * self.gy))

    def _decode_facade_action(self, action_idx):
        """
        Decode facade action index to parameter type, direction, and value.
        
        Facade actions start at NOOP_IDX + 1
        Layout: 4 parameters × 4 directions × 7 values
        """
        facade_offset = action_idx - (self.NOOP_IDX + 1)
        
        # Which parameter (0=min_cols, 1=max_cols, 2=min_rows, 3=max_rows)
        param_idx = facade_offset // (4 * 7)
        remainder = facade_offset % (4 * 7)
        
        # Which direction (0=east, 1=north, 2=west, 3=south)
        direction_idx = remainder // 7
        
        # Which value (0-6 maps to values 1-7)
        value_idx = remainder % 7
        value = value_idx + 1  # Convert 0-6 to 1-7
        
        param_names = ['min_cols', 'max_cols', 'min_rows', 'max_rows']
        direction_names = ['east', 'north', 'west', 'south']
        
        return param_names[param_idx], direction_idx, value, direction_names[direction_idx]

    def _apply_facade_action(self, param_name, direction_idx, value):
        """Apply facade parameter change and return reward"""
        old_value = self.current_facade_params[param_name][direction_idx]
        self.current_facade_params[param_name][direction_idx] = value
        
        reward = get_reward_gh(
            self.grid, 
            self.merged_gh_file, 
            self.current_epw, 
            self.sun_wt, 
            self.str_wt, 
            self.cst_wt, 
            self.wst_wt, 
            repulsors=self.repulsors,
            facade_params=self.current_facade_params)
        
        return reward

    # ---------------------------
    # Action masking support
    # ---------------------------
    def action_masks(self) -> np.ndarray:
        """
        Boolean mask of shape (2 * total_locations + 1 + facade_actions,)
        True = allowed, False = masked (invalid).
        NOOP and all facade actions are always allowed.
        """
        total_actions = self.NOOP_IDX + 1 + self.FACADE_ACTIONS
        mask = np.zeros(total_actions, dtype=bool)

        # ADD actions in [0, total_locations)
        for loc in range(self.total_locations):
            x, y, z = self._idx_to_coords(loc)
            valid_add = (
                self.grid[x, y, z] == 0
                and self._is_adjacent_to_structure(x, y, z)
                and (x, y, z) not in self._deleted_cooldown  # anti ping-pong
            )
            mask[loc] = valid_add

        # DELETE actions in [total_locations, 2*total_locations)
        for loc in range(self.total_locations):
            x, y, z = self._idx_to_coords(loc)
            valid_del = (
                self.grid[x, y, z] == 1
                and (x, y, z) != self.root_voxel
                and (x, y, z) not in self._added_cooldown   # anti ping-pong
            )
            if valid_del:
                # Connectivity-aware masking: simulate delete; only allow if structure stays connected
                self.grid[x, y, z] = 0
                still_connected = self._is_structure_connected()
                self.grid[x, y, z] = 1
                valid_del = still_connected

            mask[self.total_locations + loc] = valid_del

        # Always allow NO-OP
        mask[self.NOOP_IDX] = True
        
        # Always allow all facade actions
        mask[self.NOOP_IDX + 1:] = True

        return mask

    def _action_to_coords(self, action_idx):
        """Convert action index to action type and coordinates/parameters"""
        if action_idx == self.NOOP_IDX:
            return "noop", (None, None, None)
        
        if action_idx > self.NOOP_IDX:
            # Facade action
            param_name, direction_idx, value, direction_name = self._decode_facade_action(action_idx)
            return "facade", (param_name, direction_idx, value, direction_name)

        total_locations = self.total_locations
        if action_idx < total_locations:
            action_type = "add"
            location_idx = action_idx
        else:
            action_type = "delete"
            location_idx = action_idx - total_locations

        x, y, z = self._idx_to_coords(location_idx)
        return action_type, (x, y, z)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        self.grid = np.zeros_like(self.grid, dtype=np.int32)

        # Episode bookkeeping
        self.step_count = 0
        self._added_cooldown.clear()
        self._deleted_cooldown.clear()

        # Reset facade parameters to defaults
        self.current_facade_params = {
            'min_cols': np.array([1, 1, 1, 1], dtype=np.int32),
            'max_cols': np.array([7, 7, 7, 7], dtype=np.int32),
            'min_rows': np.array([1, 1, 1, 1], dtype=np.int32),
            'max_rows': np.array([7, 7, 7, 7], dtype=np.int32)
        }

        # New root voxel each episode (cannot delete)
        rx = rng.integers(0, self.gx)
        ry = rng.integers(0, self.gy)
        self.root_voxel = (int(rx), int(ry), 0)
        self.grid[self.root_voxel] = 1

        # Generate per-episode repulsor points (use hook if provided; else random)
        self.repulsors = []
        def _random_repulsors() -> list[list[int]]:
            pts: list[list[int]] = []
            for _ in range(max(0, int(self.num_repulsors))):
                rx = int(rng.integers(0, self.gx))
                ry = int(rng.integers(0, self.gy))
                rz = int(rng.integers(0, self.gz))
                pts.append([rx, ry, rz])
            return pts

        if self.repulsor_provider is not None:
            try:
                provided = list(self.repulsor_provider(self, rng))  # expect iterable of 3D points
                arr = np.asarray(provided, dtype=float)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                if arr.size == 0 or arr.shape[1] < 3:
                    raise ValueError("Repulsor provider must yield points with at least 3 coordinates")
                coords = np.rint(arr[:, :3]).astype(int)
                coords[:, 0] = np.clip(coords[:, 0], 0, self.gx - 1)
                coords[:, 1] = np.clip(coords[:, 1], 0, self.gy - 1)
                coords[:, 2] = np.clip(coords[:, 2], 0, self.gz - 1)
                self.repulsors = coords.tolist()[: max(0, int(self.num_repulsors))]
            except Exception:
                # Fallback to random on any provider error
                self.repulsors = _random_repulsors()
        else:
            self.repulsors = _random_repulsors()

        # Pick EPW for this episode
        self.current_epw = np.random.choice(self.epw_files)
        normalized_epw = self.current_epw.replace("\\", "/")
        print(f"[EPISODE] ☀️ Using EPW file: {os.path.basename(normalized_epw)}", flush=True)

        observation = self.grid.flatten().astype(np.float32)
        info = {
            "epw_file": os.path.basename(self.current_epw or ""),
            "action_masks": self.action_masks(),
            "repulsors": np.array(self.repulsors, dtype=np.int32) if self.repulsors else np.empty((0,3), dtype=np.int32),
            "facade_params": self.current_facade_params.copy()
        }
        return observation, info

    def set_repulsor_provider(self, provider: Optional[Callable[["VoxelEnv", np.random.Generator], Iterable[Sequence[float]]]]):
        """Install or clear a per-episode repulsor provider hook.

        provider(env, rng) -> iterable of 3D points (x, y, z) in grid index space.
        If None, random repulsors will be used.
        """
        self.repulsor_provider = provider

    def _tick_cooldowns(self):
        # decrement, then purge zeros
        for d in (self._added_cooldown, self._deleted_cooldown):
            for k in list(d.keys()):
                d[k] -= 1
                if d[k] <= 0:
                    del d[k]

    def step(self, action_idx):
        # Time/bookkeeping
        self.step_count += 1
        self._tick_cooldowns()  # age cooldowns at each step

        action_type, coords_or_params = self._action_to_coords(action_idx)
        base_reward = 0.0

        if action_type == "noop":
            base_reward = -0.01  # small penalty so it's used only when truly blocked
            msg = "NOOP"

        elif action_type == "facade":
            param_name, direction_idx, value, direction_name = coords_or_params
            base_reward = self._apply_facade_action(param_name, direction_idx, value)
            msg = f"FACADE {param_name}[{direction_name}] = {value}"
            print(f"[STEP] 🏢 {msg}")

        elif action_type == "add":
            x, y, z = coords_or_params
            base_reward = self._add_voxel(x, y, z)
            msg = "ADD"

        elif action_type == "delete":
            x, y, z = coords_or_params
            base_reward = self._delete_voxel(x, y, z)
            msg = "DELETE"

        else:
            base_reward = -0.5
            msg = "INVALID"

        # Optional per-step penalty to discourage churn
        reward = float(base_reward) - float(self.step_penalty)

        if action_type in ["add", "delete"]:
            x, y, z = coords_or_params
            if reward > -0.1 and msg == "ADD":
                print(f"[STEP] ➕ {msg} at ({x},{y},{z}) - SUCCESS")
            elif reward > -0.1 and msg == "DELETE":
                print(f"[STEP] ➖ {msg} at ({x},{y},{z}) - SUCCESS")
            else:
                print(f"[STEP] ❌ {msg} at ({x},{y},{z}) - RESULT (reward: {reward})")

        # Episode termination logic
        total_voxels = self.total_locations
        active_voxels = int(np.sum(self.grid))
        terminated = bool(active_voxels >= 0.6 * total_voxels or active_voxels <= 1)

        # Terminate if only NO-OP and facade actions remain (no adds/deletes possible)
        mask_now = self.action_masks()
        no_voxel_actions = not np.any(mask_now[:self.NOOP_IDX])
        terminated = bool(terminated or no_voxel_actions)

        # Time-limit truncation
        truncated = self.step_count >= self.max_steps

        observation = self.grid.flatten().astype(np.float32)
        info = {
            "epw_file": os.path.basename(self.current_epw or ""),
            "action_masks": self.action_masks(),
            "facade_params": self.current_facade_params.copy()
        }
        return observation, reward, terminated, truncated, info

    def _safe_reward(self, r):
        """Coerce reward to finite float; fallback on invalid values."""
        try:
            r = float(r)
        except Exception:
            return -1.0
        if not np.isfinite(r):
            return -1.0
        return r

    def _add_voxel(self, x, y, z):
        """Add a voxel at specific coordinates with validation"""
        if not (0 <= x < self.gx and 0 <= y < self.gy and 0 <= z < self.gz):
            return -0.5  # Invalid coordinates
        if self.grid[x, y, z] == 1:
            return -0.3  # Already occupied
        if not self._is_adjacent_to_structure(x, y, z):
            return -0.4  # Not connected to existing structure

        print(f"[ADD] Adding voxel at ({x}, {y}, {z})")
        self.grid[x, y, z] = 1

        # start cooldown: forbid deleting this cell for a few steps
        if self.cooldown_steps > 0:
            self._added_cooldown[(x, y, z)] = self.cooldown_steps
            self._deleted_cooldown.pop((x, y, z), None)

        # Get reward from Grasshopper (now includes facade parameters)
        reward = get_reward_gh(
            self.grid,
            self.merged_gh_file,
            self.current_epw,
            self.sun_wt,
            self.str_wt,
            repulsors=self.repulsors,
            repulsor_wt=self.repulsor_wt,
            repulsor_radius=self.repulsor_radius,
            facade_params=self.current_facade_params,
        )
        return self._safe_reward(reward)

    def _delete_voxel(self, x, y, z):
        """Delete a voxel at specific coordinates with connectivity validation"""
        if not (0 <= x < self.gx and 0 <= y < self.gy and 0 <= z < self.gz):
            return -0.5  # Invalid coordinates
        if self.grid[x, y, z] == 0:
            return -0.3  # Nothing to delete
        if (x, y, z) == self.root_voxel:
            return -0.4  # Cannot delete root voxel

        # Tentative delete with connectivity check
        self.grid[x, y, z] = 0
        if not self._is_structure_connected():
            # revert if it disconnects the structure
            self.grid[x, y, z] = 1
            return -0.4

        print(f"[DELETE] Removing voxel at ({x}, {y}, {z})")

        # start cooldown: forbid adding this cell back for a few steps
        if self.cooldown_steps > 0:
            self._deleted_cooldown[(x, y, z)] = self.cooldown_steps
            self._added_cooldown.pop((x, y, z), None)

        # Get reward from Grasshopper (now includes facade parameters)
        reward = get_reward_gh(
            self.grid,
            self.merged_gh_file,
            self.current_epw,
            self.sun_wt,
            self.str_wt,
            repulsors=self.repulsors,
            repulsor_wt=self.repulsor_wt,
            repulsor_radius=self.repulsor_radius,
            facade_params=self.current_facade_params,
        )
        return self._safe_reward(reward)

    def _is_adjacent_to_structure(self, x, y, z):
        """Check if position (x,y,z) is adjacent to existing structure"""
        neighbor_offsets = [
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1)
        ]
        for dx, dy, dz in neighbor_offsets:
            nx, ny, nz = x + dx, y + dy, z + dz
            if (0 <= nx < self.gx and
                0 <= ny < self.gy and
                0 <= nz < self.gz):
                if self.grid[nx, ny, nz] == 1:
                    return True
        return False

    def _is_structure_connected(self):
        """
        BFS from root over voxels==1; returns True iff all active voxels are reachable.
        """
        from collections import deque

        rx, ry, rz = self.root_voxel
        if self.grid[rx, ry, rz] == 0:
            return False

        total_active = int(self.grid.sum())
        visited: set[tuple[int, int, int]] = set()
        q: Deque[tuple[int, int, int]] = deque([cast(tuple[int, int, int], self.root_voxel)])
        visited.add(self.root_voxel)

        neighbor_offsets = [
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1)
        ]

        while q:
            x, y, z = q.popleft()
            for dx, dy, dz in neighbor_offsets:
                nx, ny, nz = x + dx, y + dy, z + dz
                if (0 <= nx < self.gx and
                    0 <= ny < self.gy and
                    0 <= nz < self.gz):
                    if self.grid[nx, ny, nz] == 1 and (nx, ny, nz) not in visited:
                        visited.add((nx, ny, nz))
                        q.append((nx, ny, nz))

        return len(visited) == total_active

    # Deprecated random methods kept for compatibility
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
    """
    Simple Python-side vectorized wrapper (not a SB3 VecEnv).
    Kept for convenience; agent_training.py uses DummyVecEnv.
    """
    def __init__(self, num_envs=8, grid_size=5, device=None):
        self.num_envs = num_envs
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        # default params of VoxelEnv are fine here
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
