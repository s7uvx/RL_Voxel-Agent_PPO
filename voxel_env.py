import numpy as np
import os
import json
import torch
from gymnasium import Env, spaces
import compute_rhino3d.Util
from typing import Deque, cast, Callable, Iterable, Sequence, Optional

from gh_file_runner import get_reward_gh

class VoxelEnv(Env):
    """
    Voxel placement environment with action masking, safety NO-OP action, and facade options.
    + Optional early DONE action to let the agent terminate an episode.
    """
    def __init__(self, port, grid_size=5, device=None, max_steps=200, cooldown_steps=7, step_penalty=0.0,
                 num_repulsors: int = 2, sun_wt = 0.5, str_wt = 0.3, cst_wt = 0.1, wst_wt = 0.1, day_wt = 0.1,
                 repulsor_penalty_wt: float = 0.2, repulsor_radius: float = 2.0,
                 repulsor_provider: str = "edge",
                 early_done: bool = False,
                 early_done_bonus: float = 0.0,
                 early_done_min_voxels: float = 0.0,
                 save_actions: bool = False,
                 actions_output_dir: str = "episode_actions",
                 export_last_epoch_episode: bool = False,
                 model_name: str = "model",
                 log_actions_every: int = 100,
                 height_wt: float = 0.0,
                 repulsor_clear_wt: float = 0.0):
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
        self.height_wt = float(height_wt)
        self.repulsor_clear_wt = float(repulsor_clear_wt)
        self.repulsor_penalty_wt = float(repulsor_penalty_wt)
        self._max_repulsor_dist = np.linalg.norm([self.gx-1, self.gy-1, self.gz-1])
    
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.grid = np.zeros((self.gx, self.gy, self.gz), dtype=np.int32)

        # Place root voxel and remember it (cannot be deleted)
        rx = np.random.randint(0, self.gx)
        ry = np.random.randint(0, self.gy)
        self.root_voxel = (rx, ry, 0)
        self.grid[rx, ry, 0] = 1
        # Backbone (central tower) container
        self.central_tower_set: set[tuple[int,int,int]] = set()
        self._build_central_tower()  # ensure full-height tower at start

        # Action space with NO-OP and facade actions
        total_locations = int(self.gx * self.gy * self.gz)
        self.total_locations = total_locations
        self.NOOP_IDX = 2 * total_locations
        self.FACADE_ACTIONS = 2 * 4 * 7  # 56 facade actions
        self.voxel_actions_count = self.NOOP_IDX + 1
        self.facade_actions_count = self.FACADE_ACTIONS + 1
        self.NO_FACADE_IDX = self.FACADE_ACTIONS

        # Early-done config
        self.early_done = bool(early_done)
        self.early_done_bonus = float(early_done_bonus)
        # Allow user to give fraction (<=1) or absolute count (>1)
        if early_done_min_voxels <= 1.0:
            self.early_done_min_voxels = int(round(early_done_min_voxels * self.total_locations))
        else:
            self.early_done_min_voxels = int(early_done_min_voxels)

        self._combo_action_count = self.voxel_actions_count * self.facade_actions_count
        if self.early_done:
            self.DONE_IDX = self._combo_action_count  # last index
            self.action_space = spaces.Discrete(self._combo_action_count + 1)
        else:
            self.DONE_IDX = None
            self.action_space = spaces.Discrete(self._combo_action_count)

        self.episode_return = 0.0  # <-- INIT cumulative episode reward

        # Facade parameter ranges and current values
        self.facade_min_val = 1
        self.facade_max_val = 7
        self.current_facade_params = {
            'cols': np.array([3,3,3,3], dtype=np.int32),  # [east, north, west, south]
            'rows': np.array([4,4,4,4], dtype=np.int32),
        }

    # Observation space: Dict with voxel + repulsor grids
        self.observation_space = spaces.Dict({
            'vox': spaces.Box(low=0, high=1, shape=(self.gx, self.gy, self.gz), dtype=np.float32),
            'rep': spaces.Box(low=0, high=1, shape=(self.gx, self.gy, self.gz), dtype=np.float32)
        })

        # Episode/time-limit + anti-churn
        self.max_steps = max_steps
        self.step_count = 0
        self.cooldown_steps = cooldown_steps
        self._added_cooldown = {}    # forbid DELETE here for N steps
        self._deleted_cooldown = {}  # forbid ADD here for N steps
        self.step_penalty = step_penalty  # tiny cost per step (optional)

        # Loop-churn penalty settings
        self.loop_penalty_scale = 0.05      # strength of penalty per toggle (tune)
        self.loop_penalty_power = 1.0       # growth curve; >1 amplifies repeated flips
        self.loop_decay = 0.98              # per-step decay of toggle memory
        self._toggle_counts: dict[tuple[int, int, int], float] = {}  # voxel -> decaying toggle count
        # Unstick settings
        self.unstick_on_exhausted = True
        self.unstick_penalty = 0.05

        self.timeouts = 0
        self.port = port
        self.merged_gh_file = os.path.join(os.getcwd(), 'gh_files', 'RL_Voxel_V6_hops.ghx')
        compute_rhino3d.Util.url = f"http://localhost:{self.port}/"
        self.sun_wt = sun_wt
        self.str_wt = str_wt
        self.cst_wt = cst_wt
        self.wst_wt = wst_wt
        self.day_wt = day_wt
        # Repulsor config
        self.num_repulsors = num_repulsors
        # self.repulsor_wt = repulsor_wt
        self.repulsor_radius = repulsor_radius
        self.repulsor_provider = repulsor_provider
        self._set_repulsors()

        self.epw_folder = os.path.join(os.getcwd(), 'gh_files', 'epw')
        self.epw_files = [
            os.path.join(self.epw_folder, f)
            for f in os.listdir(self.epw_folder)
            if f.lower().endswith(".epw")
        ]
        if not self.epw_files:
            raise FileNotFoundError(f"No EPW files found in {self.epw_folder}")

        self.current_epw = None
        self.noop_count = 0
        # self.repulsors = []
        # (Was redefining observation_space with 'rpl' key; unify to 'rep')
        self.observation_space = spaces.Dict({
            'vox': spaces.Box(0, 1, (self.gx, self.gy, self.gz), dtype=np.float32),
            'rep': spaces.Box(0, 1, (self.gx, self.gy, self.gz), dtype=np.float32)
        })

        # Action tracking for episode output
        self.save_actions = save_actions
        self.actions_output_dir = actions_output_dir
        self.log_actions_every = log_actions_every
        self.current_episode_actions = []
        self.current_episode_rewards = []
        self.current_episode_infos = []
        self.episode_count = 0
        self.step_count_global = 0  # Global step counter across all episodes
        
        # Epoch-based step export settings
        self.export_last_epoch_episode = export_last_epoch_episode
        self.model_name = model_name
        self.current_epoch = 0
        self.is_last_episode_of_epoch = False
        self.epoch_step_export_dir = None
        # Stable run timestamp (so each episode of a run goes into same dated folder)
        # self.run_timestamp = time.strftime("%Y%m%d-%H%M", time.localtime())

        if self.save_actions:
            os.makedirs(self.actions_output_dir, exist_ok=True)

    # ---------------------------
    # Utilities
    # ---------------------------
    def _compute_height_reward(self):
        """
        Reward components:
          max_height_norm: tallest occupied voxel z / (gz-1)
          mean_height_norm: mean column height / gz
        Returns scalar (unweighted) combination = 0.6*max + 0.4*mean
        """
        if not np.any(self.grid):
            return 0.0
        # Column heights: highest z with voxel (or -1)
        occ = self.grid > 0
        heights = np.where(occ.any(axis=2),
                           occ.shape[2] - np.argmax(occ[:, :, ::-1] == 1, axis=2) - 1,
                           -1)
        valid = heights >= 0
        if not np.any(valid):
            return 0.0
        max_h = heights[valid].max()
        mean_h = heights[valid].mean()
        max_height_norm = (max_h / (self.gz - 1)) if self.gz > 1 else 0.0
        mean_height_norm = (mean_h / self.gz) if self.gz > 0 else 0.0
        return 0.6 * max_height_norm + 0.4 * mean_height_norm

    def _compute_repulsor_shaping(self):
        """
        Returns (positive_clear_reward, negative_intersection_penalty_unweighted)
        positive_clear_reward: average normalized distance of occupied voxels to nearest repulsor
        negative_intersection_penalty_unweighted: number of occupied voxels that exactly coincide with a repulsor point
        """
        if not hasattr(self, 'repulsors') or not self.repulsors:
            return 0.0, 0.0
        vox_idx = np.argwhere(self.grid == 1)
        if vox_idx.size == 0:
            return 0.0, 0.0
        reps = np.array(self.repulsors, dtype=np.float32)
        # Distance matrix: for efficiency compute min distances iteratively (repulsor count usually small)
        dmins = []
        intersect_count = 0
        rep_set = { (int(r[0]), int(r[1]), int(r[2])) for r in reps }
        for (x, y, z) in vox_idx:
            if (int(x), int(y), int(z)) in rep_set:
                intersect_count += 1
            diffs = reps - np.array([x, y, z], dtype=np.float32)
            dist = np.sqrt((diffs * diffs).sum(axis=1)).min()
            dmins.append(dist)
        dmins = np.array(dmins, dtype=np.float32)
        # Normalize by max possible distance to keep in [0,1]
        clear_reward = float(np.clip(dmins.mean() / max(1e-6, self._max_repulsor_dist), 0.0, 1.0))
        return clear_reward, float(intersect_count)
    
    def _build_central_tower(self):
        """
        Ensure a solid vertical tower of voxels from z=0..gz-1 at (root_x, root_y, z).
        Rebuilds (overwrites) any previous tower definition.
        """
        self.central_tower_set.clear()
        rx, ry, _ = self.root_voxel
        for z in range(self.gz):
            self.grid[rx, ry, z] = 1
            self.central_tower_set.add((rx, ry, z))
            
    def _idx_to_coords(self, location_idx):
        area = self.gx * self.gy
        z = location_idx // area
        rem = location_idx % area
        y = rem // self.gx
        x = rem % self.gx
        return x, y, z

    def _coords_to_idx(self, x, y, z):
        return int(x + y * self.gx + z * (self.gx * self.gy))

    def _repulsor_grid(self):
        """
        Return a (gx, gy, gz) float32 grid with 1.0 at repulsor positions, else 0.0.
        """
        grid = np.zeros((self.gx, self.gy, self.gz), dtype=np.float32)
        if hasattr(self, 'repulsors') and self.repulsors:
            for p in self.repulsors:
                try:
                    px, py, pz = int(p[0]), int(p[1]), int(p[2])
                except Exception:
                    continue
                if 0 <= px < self.gx and 0 <= py < self.gy and 0 <= pz < self.gz:
                    grid[px, py, pz] = 1.0
        return grid

    def _decode_facade_action(self, facade_action_idx: int):
        """
        Decode facade action index (0..FACADE_ACTIONS-1) to parameter type, direction, and value.
        """
        facade_offset = facade_action_idx  # already 0-based within facade actions
        param_idx = facade_offset // (4 * 7)
        remainder = facade_offset % (4 * 7)
        direction_idx = remainder // 7
        value_idx = remainder % 7
        value = value_idx + 1
        param_names = ['cols', 'rows']
        direction_names = ['east', 'north', 'west', 'south']
        return param_names[param_idx], direction_idx, value, direction_names[direction_idx]

    def _decode_combined_action(self, action_idx: int):
        """
        Extended: if early_done enabled and action_idx == DONE_IDX -> DONE action.
        """
        if self.early_done and action_idx == self.DONE_IDX:
            return ("done", (None, None, None)), None

        voxel_idx = action_idx // self.facade_actions_count
        facade_idx = action_idx % self.facade_actions_count

        # Decode voxel part
        if voxel_idx == self.NOOP_IDX:
            voxel_part = ("noop", (None, None, None))
        else:
            total_locations = self.total_locations
            if voxel_idx < total_locations:
                action_type = "add"
                location_idx = voxel_idx
            else:
                action_type = "delete"
                location_idx = voxel_idx - total_locations
            x, y, z = self._idx_to_coords(location_idx)
            voxel_part = (action_type, (x, y, z))

        # Decode facade part
        if facade_idx == self.NO_FACADE_IDX:
            facade_part = None
        else:
            param_name, direction_idx, value, direction_name = self._decode_facade_action(facade_idx)
            facade_part = ("facade", (param_name, direction_idx, value, direction_name))

        return voxel_part, facade_part

    def _apply_facade_action(self, param_name, direction_idx, value, compute_reward: bool = True):
        """Apply facade parameter change, optionally compute reward."""
        # old_value = self.current_facade_params[param_name][direction_idx]
        self.current_facade_params[param_name][direction_idx] = value

        if not compute_reward:
            # Return consistent format even when not computing reward
            empty_components = {
                'total_reward': 0.0,
                'cyclops_reward': 0.0,
                'karamba_reward': 0.0,
                'panel_cost_reward': 0.0,
                'panel_waste_reward': 0.0,
                'daylight_autonomy_reward': 0.0,
                'repulsor_penalty': 0.0,
                'weighted_cyclops': 0.0,
                'weighted_karamba': 0.0,
                'weighted_panel_cost': 0.0,
                'weighted_panel_waste': 0.0,
                'weighted_daylight_autonomy': 0.0
            }
            return 0.0, empty_components

        reward, reward_dict = get_reward_gh(
            self.grid, 
            self.merged_gh_file, 
            self.current_epw, 
            self.sun_wt, 
            self.str_wt, 
            self.cst_wt, 
            self.wst_wt,
            self.day_wt, 
            repulsors=self.repulsors,
            facade_params=self.current_facade_params)
        
        return reward, reward_dict
    
    def _set_repulsors(self):
        if self.repulsor_provider == "random":
            repulse_choice = np.random.choice(["edge", "floor", "volume"])
        else:
            repulse_choice = self.repulsor_provider
        match repulse_choice:
            case "edge":
                self.repulsors = self._edge_repulsors()
            case "floor":
                self.repulsors = self._floor_repulsors()
            case "volume":
                self.repulsors = self._volume_repulsors()
    
    def _edge_repulsors(self):
        """Generate edge repulsors with one-liner list comprehension"""
        edges = [
            lambda: [0, np.random.randint(0, self.gy), 0],              # Left
            lambda: [self.gx-1, np.random.randint(0, self.gy), 0],     # Right  
            lambda: [np.random.randint(0, self.gx), 0, 0],             # Front
            lambda: [np.random.randint(0, self.gx), self.gy-1, 0]      # Back
        ]

        return [edges[np.random.randint(0,4)]() for i in range(self.num_repulsors)]
    
    def _floor_repulsors(self):
        return [(np.random.randint(0,self.gx), np.random.randint(0,self.gy),0) for i in range(self.num_repulsors)]

    def _volume_repulsors(self):
        return [(np.random.randint(0,self.gx), np.random.randint(0,self.gy), np.random.randint(0,self.gz)) for i in range(self.num_repulsors)]
    # ---------------------------
    # Action masking support
    # ---------------------------
    def facade_action_masks(self) -> np.ndarray:
        """
        Boolean mask of length (FACADE_ACTIONS + 1) for facade actions.
        For now, all are allowed, including NO-FACADE (last index).
        """
        mask = np.ones(self.facade_actions_count, dtype=np.bool_)
        # Ensure NO-FACADE is always allowed
        mask[self.NO_FACADE_IDX] = True
        return mask

    def action_masks(self) -> np.ndarray:
        """
        Boolean mask for the combined action space (voxel × facade).
        Allowed iff both sub-actions are allowed. Guarantees at least one action.
        """
        voxel_mask = self.voxel_action_masks()
        facade_mask = self.facade_action_masks()
        
        # Create the base mask for combined actions
        base_mask = (voxel_mask[:, None] & facade_mask[None, :]).reshape(-1)

        # If early_done is enabled, we need to handle the extra action index
        if self.early_done:
            # The base mask corresponds to the first N actions, so we append a placeholder for the DONE action
            total_actions = self._combo_action_count + 1
            full_mask = np.zeros(total_actions, dtype=bool)
            full_mask[:self._combo_action_count] = base_mask
            # The DONE action is always available if enabled
            full_mask[self.DONE_IDX] = True
        else:
            full_mask = base_mask

        # Safeguard: If no actions are available at all, force the full NOOP action to be valid.
        # This prevents the agent from getting stuck and causing the value error.
        if not np.any(full_mask):
            noop_action_idx = self.NOOP_IDX * self.facade_actions_count + self.NO_FACADE_IDX
            if 0 <= noop_action_idx < len(full_mask):
                full_mask[noop_action_idx] = True
                print("[WARN] No valid actions found, forcing full NOOP to be available.")
            
        return full_mask

    def voxel_action_masks(self) -> np.ndarray:
        """
        Boolean mask of length (2 * total_locations + 1) for voxel actions only.
        True = allowed, False = masked (invalid).
        NOOP is always allowed.
        """
        mask = np.zeros(self.voxel_actions_count, dtype=np.bool_)
        # ADD actions in [0, total_locations)
        for loc in range(self.total_locations):
            x, y, z = self._idx_to_coords(loc)
            valid_add = (
                self.grid[x, y, z] == 0
                and self._is_adjacent_to_structure(x, y, z)
                and (x, y, z) not in self._deleted_cooldown
            )
            mask[loc] = valid_add

        # DELETE actions in [total_locations, 2*total_locations)
        for loc in range(self.total_locations):
            x, y, z = self._idx_to_coords(loc)
            # Prevent deleting any backbone voxel
            if (x, y, z) in getattr(self, "central_tower_set", ()):
                mask[self.total_locations + loc] = False
                continue
            valid_del = (
                self.grid[x, y, z] == 1
                and (x, y, z) != self.root_voxel
                and (x, y, z) not in self._added_cooldown
            )
            if valid_del:
                # simulate deletion and ensure structure stays connected
                self.grid[x, y, z] = 0
                still_connected = self._is_structure_connected()
                self.grid[x, y, z] = 1
                valid_del = still_connected
            mask[self.total_locations + loc] = valid_del

        # Always allow NO-OP
        mask[self.NOOP_IDX] = True
        return mask

    def reset(self, seed=None, options=None):
        # Save previous episode actions if we have any and save_actions is enabled
        if self.save_actions and self.current_episode_actions:
            self._save_episode_actions()
        
        # Reset action tracking for new episode (always track if save_actions is enabled)
        if self.save_actions:
            self.current_episode_actions = []
            self.current_episode_rewards = []
            self.current_episode_infos = []
        self.episode_count += 1
        
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        self.grid = np.zeros_like(self.grid, dtype=np.int32)
        self.episode_return = 0.0  # <-- RESET cumulative reward

        # Episode bookkeeping
        self.step_count = 0
        self._added_cooldown.clear()
        self._deleted_cooldown.clear()
        self.noop_count = 0

        # Reset facade parameters to defaults
        self.current_facade_params = {
            'cols': np.array([3, 3, 3, 3], dtype=np.int32),
            'rows': np.array([4, 4, 4, 4], dtype=np.int32)
        }

        # New root voxel each episode (cannot delete)
        rx = rng.integers(0, self.gx)
        ry = rng.integers(0, self.gy)
        self.root_voxel = (int(rx), int(ry), 0)
        self.grid[self.root_voxel] = 1
        # Rebuild central tower for new episode
        self._build_central_tower()

        self._set_repulsors()
        # Pick EPW for this episode
        self.current_epw = np.random.choice(self.epw_files)
        normalized_epw = self.current_epw.replace("\\", "/")
        print(f"[EPISODE] ☀️ Using EPW file: {os.path.basename(normalized_epw)}", flush=True)

        # Build dict observation
        observation = {
            'vox': self.grid.astype(np.float32),
            'rep': self._repulsor_grid()
        }
        info = {
            "epw_file": os.path.basename(self.current_epw or ""),
            "action_masks": self.action_masks(),
            "repulsors": np.array(self.repulsors, dtype=np.int32) if self.repulsors else np.empty((0,3), dtype=np.int32),
            "facade_params": self.current_facade_params.copy()
        }
        if self.early_done:
            info["early_done_available"] = True
        return observation, info

    def set_repulsor_provider(self, provider: Optional[Callable[["VoxelEnv", np.random.Generator], Iterable[Sequence[float]]]]):
        """Install or clear a per-episode repulsor provider hook.

        provider(env, rng) -> iterable of 3D points (x, y, z) in grid index space.
        If None, random repulsors will be used.
        """
        self.repulsor_provider = provider

    def set_epoch_info(self, epoch: int, is_last_episode: bool = False):
        """Set current epoch information for step export tracking"""
        self.current_epoch = epoch
        self.is_last_episode_of_epoch = is_last_episode
        
        if self.export_last_epoch_episode and is_last_episode:
            import time
            model_date_time = time.strftime("%Y%m%d-%H%M", time.localtime())
            self.epoch_step_export_dir = f"output_steps/{self.model_name}_{model_date_time}_epoch_{epoch}"
            os.makedirs(self.epoch_step_export_dir, exist_ok=True)
            print(f"[EPOCH_EXPORT] 📁 Prepared step export directory: {self.epoch_step_export_dir}")
        else:
            self.epoch_step_export_dir = None

    def enable_epoch_step_export(self, model_name: str = "model"):
        """Enable step export for last episode of each epoch"""
        self.export_last_epoch_episode = True
        self.model_name = model_name
        try:
            model_id = getattr(self, 'model_name', 'model')
        except Exception:
            model_id = 'model'
        print(f"[EPOCH_EXPORT] Enabled epoch step export for model: {model_id}")

    def _tick_cooldowns(self):
        # decrement, then purge zeros
        for d in (self._added_cooldown, self._deleted_cooldown):
            for k in list(d.keys()):
                d[k] -= 1
                if d[k] <= 0:
                    del d[k]
        # Decay loop-churn memory
        if self._toggle_counts:
            for k in list(self._toggle_counts.keys()):
                self._toggle_counts[k] *= self.loop_decay
                if self._toggle_counts[k] < 0.05:
                    del self._toggle_counts[k]

    def _reset_action_restrictions(self, *, clear_cooldowns: bool = True, clear_toggles: bool = True):
        """Clear cooldown and/or loop memory to restore valid actions."""
        if clear_cooldowns:
            self._added_cooldown.clear()
            self._deleted_cooldown.clear()
        if clear_toggles:
            self._toggle_counts.clear()

    def reset_action_mask(self, clear_cooldowns: bool = True, clear_toggles: bool = True):
        """
        Public helper to clear internal restrictions so the next call to action_masks()
        recomputes from current grid without cooldown/toggle penalties.
        """
        self._reset_action_restrictions(clear_cooldowns=clear_cooldowns, clear_toggles=clear_toggles)

    def _toggle_penalty(self, x: int, y: int, z: int) -> float:
        """Increase per-voxel toggle counter and return penalty."""
        key = (int(x), int(y), int(z))
        c = float(self._toggle_counts.get(key, 0.0))
        c = c * self.loop_decay + 1.0  # decayed memory + this flip
        self._toggle_counts[key] = c
        return float(self.loop_penalty_scale * (c ** self.loop_penalty_power))

    def step(self, action_idx):
        # Store the action at the beginning of step
        if self.save_actions:
            self.current_episode_actions.append(int(action_idx))
        
        # Early-done intercept
        if self.early_done and action_idx == self.DONE_IDX:
            # Optional bonus if structural occupancy condition met
            active_voxels = int(np.sum(self.grid))
            condition_met = active_voxels >= self.early_done_min_voxels
            bonus = self.early_done_bonus if condition_met else -abs(self.early_done_bonus)
            reward = float(bonus)
            self.episode_return += reward
            observation = self.grid.flatten().astype(np.float32)
            #TODO: add repulsor positions to observation
            info = {
                "early_done": True,
                "episode_return": self.episode_return,
                "active_voxels": active_voxels,
                "bonus_applied": bonus,
                "condition_met": condition_met
            }
            
            # Store reward and info for action tracking
            if self.save_actions:
                self.current_episode_rewards.append(float(reward))
                self.current_episode_infos.append(info.copy())
                # Save actions when episode ends
                self._save_episode_actions()
            
            terminated = True
            truncated = False
            return observation, reward, terminated, truncated, info

        # Time/bookkeeping
        self.step_count += 1
        self._tick_cooldowns()

        (voxel_type, voxel_payload), facade_part = self._decode_combined_action(int(action_idx))
        reward = 0.0
        step_msgs = []
        
        # Initialize reward_dict with empty components (will be updated by GH calls)
        reward_dict = {
            'total_reward': 0.0,
            'cyclops_reward': 0.0,
            'karamba_reward': 0.0,
            'panel_cost_reward': 0.0,
            'panel_waste_reward': 0.0,
            'daylight_autonomy_reward': 0.0,
            'repulsor_penalty': 0.0,
            'weighted_cyclops': 0.0,
            'weighted_karamba': 0.0,
            'weighted_panel_cost': 0.0,
            'weighted_panel_waste': 0.0,
            'weighted_daylight_autonomy': 0.0,
            'toggle_penalty': 0.0
        }
        # Apply facade first (no reward yet if a voxel op will follow)
        if facade_part is not None:
            _, (param_name, direction_idx, value, direction_name) = facade_part
            # If a voxel op happens, skip reward here to avoid double GH calls
            compute_reward = (voxel_type == "noop")
            facade_r, reward_dict = self._apply_facade_action(param_name, direction_idx, value, compute_reward=compute_reward)
            step_msgs.append(f"FACADE {param_name}[{direction_name}] = {value}")
            reward += float(facade_r)

        # Apply voxel op
        if voxel_type == "noop":
            self.noop_count += 1
            # Penalize early NOOP
            if self.step_count < 20:
                reward -= 0.05
            # Penalize NOOP
            reward -= float(self.step_penalty)*float(self.noop_count)
            # If no facade change happened either, apply small penalty
            if facade_part is None:
                reward += -0.05
            msg = "NOOP"
        elif voxel_type == "add":
            x, y, z = voxel_payload
            base_reward, reward_dict = self._add_voxel(x, y, z)  # calls GH with current (possibly updated) facade params
            reward += float(base_reward)
            msg = f"ADD at ({x},{y},{z})"
        elif voxel_type == "delete":
            x, y, z = voxel_payload
            base_reward, reward_dict = self._delete_voxel(x, y, z)  # calls GH with current (possibly updated) facade params
            reward += float(base_reward)
            msg = f"DELETE at ({x},{y},{z})"
        else:
            reward += -0.5
            msg = "INVALID"

        height_component = 0.0
        rep_clear_component = 0.0
        rep_intersections = 0.0
        rep_penalty_component = 0.0
        if self.height_wt > 0.0:
            height_component = self._compute_height_reward()
            reward += self.height_wt * height_component
        if (self.repulsor_clear_wt > 0.0) or (self.repulsor_penalty_wt > 0.0):
            rep_clear_component, rep_intersections = self._compute_repulsor_shaping()
            if self.repulsor_clear_wt > 0.0:
                reward += self.repulsor_clear_wt * rep_clear_component
            if self.repulsor_penalty_wt > 0.0 and rep_intersections > 0:
                rep_penalty_component = - self.repulsor_penalty_wt * rep_intersections
                reward += rep_penalty_component
        # Logging
        if step_msgs:
            print(f"[STEP] {' | '.join(step_msgs)}")
        print(f"[STEP] ▶ {msg} -> reward: {reward:.3f}")

        # Episode termination logic
        total_voxels = self.total_locations
        active_voxels = int(np.sum(self.grid))
        terminated = bool(active_voxels >= 0.6 * total_voxels)

        # Terminate or unstick if no add/delete actions remain (ignoring NOOP)
        full_mask = self.action_masks()
        # Check if any actions other than NOOP or DONE are available
        noop_action_idx = self.NOOP_IDX * self.facade_actions_count + self.NO_FACADE_IDX
        other_actions_available = False
        for i, is_available in enumerate(full_mask):
            if is_available:
                is_noop = (i == noop_action_idx)
                is_done = self.early_done and (i == self.DONE_IDX)
                if not is_noop and not is_done:
                    other_actions_available = True
                    break
        
        unstuck = False  # kept for backward compatibility in logs
        exhausted_actions = False
        if not other_actions_available and not terminated:
            # No valid actions (besides NOOP/DONE). Reset internal restrictions and terminate.
            self.reset_action_mask(clear_cooldowns=True, clear_toggles=True)
            terminated = True
            exhausted_actions = True
            print("[TERMINATE] No valid actions remaining -> terminating episode and resetting action mask.")

        # Time-limit truncation
        truncated = self.step_count >= self.max_steps
        
        observation = {
            'vox': self.grid.astype(np.float32),
            'rep': self._repulsor_grid()
        }
        self.episode_return += reward
        info = {
            "epw_file": os.path.basename(self.current_epw or ""),
            "action_masks": self.action_masks(),
            "facade_params": self.current_facade_params.copy(),
            "unstuck": unstuck,
            "exhausted_actions": exhausted_actions,
            "episode_return": self.episode_return,
            "height_component": height_component,
            "rep_clear_component": rep_clear_component,
            "rep_intersections": rep_intersections,
            "rep_penalty_component": rep_penalty_component,
            "reward_dict" : reward_dict
        }
        
        # Store reward and info after processing
        if self.save_actions:
            self.current_episode_rewards.append(float(reward))
            self.current_episode_infos.append(info.copy())
            # Save individual step file showing what was passed to Grasshopper
            self._save_step_file(action_idx, reward, terminated or truncated)
        
        # If episode is ending (terminated or truncated), save actions
        if self.save_actions and (terminated or truncated):
            self._save_episode_actions()
        
        return observation, reward, terminated, truncated, info

    def _safe_reward(self, r):
        """Coerce reward to finite float; fallback on invalid values."""
        try:
            r = float(r)
        except Exception:
            print('float issue in safe reward')
            return -0.0
        if not np.isfinite(r):
            print('isfinite issue in safe reward')
            return -0.0
        return r

    def _error_reward_with_components(self, reward_value: float):
        """Return error reward with empty components in proper format."""
        empty_components = {
            'total_reward': reward_value,
            'cyclops_reward': 0.0,
            'karamba_reward': 0.0,
            'panel_cost_reward': 0.0,
            'panel_waste_reward': 0.0,
            'daylight_autonomy_reward': 0.0,
            'repulsor_penalty': 0.0,
            'weighted_cyclops': 0.0,
            'weighted_karamba': 0.0,
            'weighted_panel_cost': 0.0,
            'weighted_panel_waste': 0.0,
            'weighted_daylight_autonomy': 0.0
        }
        return reward_value, empty_components

    def _add_voxel(self, x, y, z):
        """Add a voxel at specific coordinates with validation"""
        if not (0 <= x < self.gx and 0 <= y < self.gy and 0 <= z < self.gz):
            return self._error_reward_with_components(-0.5)  # Invalid coordinates
        if self.grid[x, y, z] == 1:
            return self._error_reward_with_components(-0.3)  # Already occupied
        if not self._is_adjacent_to_structure(x, y, z):
            return self._error_reward_with_components(-0.4)  # Not connected to existing structure

        print(f"[ADD] Adding voxel at ({x}, {y}, {z})")
        self.grid[x, y, z] = 1

        # start cooldown: forbid deleting this cell for a few steps
        if self.cooldown_steps > 0:
            self._added_cooldown[(x, y, z)] = self.cooldown_steps
            self._deleted_cooldown.pop((x, y, z), None)

        # Get reward from Grasshopper (now includes facade parameters)
        reward, reward_dict = get_reward_gh(
            self.grid,
            self.merged_gh_file,
            self.current_epw,
            self.sun_wt, 
            self.str_wt, 
            self.cst_wt, 
            self.wst_wt,
            self.day_wt, 
            repulsors=self.repulsors,
            repulsor_wt=self.repulsor_penalty_wt,
            repulsor_radius=self.repulsor_radius,
            facade_params=self.current_facade_params,
        )
        # Apply loop-churn penalty for flipping this voxel
        toggle_penalty = self._toggle_penalty(x, y, z)
        reward -= toggle_penalty
        reward_dict['toggle_penalty'] = toggle_penalty
        return self._safe_reward(reward), reward_dict

    def _delete_voxel(self, x, y, z):
        """Delete a voxel at specific coordinates with connectivity validation"""
        # Block deletion if voxel is part of the enforced central tower
        if (x, y, z) in getattr(self, "central_tower_set", ()):
            return self._error_reward_with_components(-0.4)  # Protected backbone voxel

        if not (0 <= x < self.gx and 0 <= y < self.gy and 0 <= z < self.gz):
            return self._error_reward_with_components(-0.5)  # Invalid coordinates
        if self.grid[x, y, z] == 0:
            return self._error_reward_with_components(-0.3)  # Nothing to delete
        if (x, y, z) == self.root_voxel:
            return self._error_reward_with_components(-0.4)  # Cannot delete root voxel

        # Tentative delete with connectivity check
        self.grid[x, y, z] = 0
        if not self._is_structure_connected():
            self.grid[x, y, z] = 1
            return self._error_reward_with_components(-0.4)

        print(f"[DELETE] Removing voxel at ({x}, {y}, {z})")

        # start cooldown: forbid adding this cell back for a few steps
        if self.cooldown_steps > 0:
            self._deleted_cooldown[(x, y, z)] = self.cooldown_steps
            self._added_cooldown.pop((x, y, z), None)

        # Get reward from Grasshopper (now includes facade parameters)
        reward, reward_dict = get_reward_gh(
            self.grid,
            self.merged_gh_file,
            self.current_epw,
            self.sun_wt,
            self.str_wt,
            self.cst_wt,
            self.wst_wt,
            self.day_wt,
            repulsors=self.repulsors,
            repulsor_wt=self.repulsor_penalty_wt,
            repulsor_radius=self.repulsor_radius,
            facade_params=self.current_facade_params,
        )
        # Apply loop-churn penalty for flipping this voxel
        toggle_penalty = self._toggle_penalty(x, y, z)
        reward -= toggle_penalty
        reward_dict['toggle_penalty'] = toggle_penalty
        return self._safe_reward(reward), reward_dict

    def _is_adjacent_to_structure(self, x, y, z):
        """Check if position (x,y,z) is adjacent to existing structure"""
        if not (0 <= x < self.gx and 0 <= y < self.gy and 0 <= z < self.gz):
            return False
        if self.grid[x, y, z] == 1:
            return False

        neighbor_offsets = [
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1)
        ]
        for dx, dy, dz in neighbor_offsets:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < self.gx and 0 <= ny < self.gy and 0 <= nz < self.gz:
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

    def _save_step_file(self, action_idx, reward, done):
        """Save individual step file showing what was passed to Grasshopper"""
        if not self.save_actions and not (self.export_last_epoch_episode and self.is_last_episode_of_epoch):
            return
        
        # Check if we should save this episode based on frequency (same as _save_episode_actions)
        if self.save_actions and self.log_actions_every > 0 and (self.episode_count % self.log_actions_every != 0):
            # Skip saving step files for episodes that won't be logged
            pass  # Don't return here, we still want epoch export to work
        elif not self.save_actions:
            # Only proceed if epoch export is enabled
            pass
        else:
            # We should save this episode's steps
            pass
        
        def _to_serializable(obj):
            """Convert numpy types and other non-serializable objects to Python native types"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: _to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_to_serializable(x) for x in obj]
            else:
                return obj
        
        try:
            # Increment global step counter
            self.step_count_global += 1
            
            # Decode the action to get details
            voxel_part, facade_part = self._decode_combined_action(action_idx)
            action_type, coords_or_params = voxel_part
            
            # Create action info similar to output_steps format
            action_info = {
                "step": int(self.step_count_global),
                "action_idx": int(action_idx),
                "action_type": str(action_type),
                "reward": float(reward),
                "done": bool(done)
            }
            
            # Add specific action details
            if action_type in ["add", "delete"] and coords_or_params is not None:
                try:
                    if isinstance(coords_or_params, (list, tuple)) and len(coords_or_params) >= 3:
                        x, y, z = coords_or_params[:3]
                        if x is not None and y is not None and z is not None:
                            action_info["coordinates"] = [int(x), int(y), int(z)]
                except (ValueError, TypeError, IndexError):
                    pass  # Skip if coordinates can't be converted
            elif action_type == "done":
                action_info["early_termination"] = True
            
            # Add facade information if present
            if facade_part is not None:
                _, (param_name, direction_idx, value, direction_name) = facade_part
                action_info["facade_change"] = {
                    "parameter": str(param_name),
                    "direction": str(direction_name),
                    "direction_idx": int(direction_idx),
                    "value": int(value)
                }
            
            # Create data structure matching output_steps format
            step_data = {
                "voxels": self.grid.tolist(),
                "epw_file": os.path.basename(self.current_epw) if self.current_epw else None,
                "facade_params": {
                    "cols": self.current_facade_params["cols"].tolist(),
                    "rows": self.current_facade_params["rows"].tolist()
                },
                "repulsors": _to_serializable(self.repulsors) if hasattr(self, 'repulsors') and self.repulsors else [],
                "action_info": _to_serializable(action_info)
            }
            
            # Save to regular actions output directory with episode-specific folder
            # Only save if this episode should be logged
            if self.save_actions and (self.episode_count % self.log_actions_every == 0):
                             
                # Create episode-specific subfolder
                episode_step_dir = os.path.join(self.actions_output_dir, f"episode_{self.episode_count:04d}")
                os.makedirs(episode_step_dir, exist_ok=True)
                
                step_filename = f"step_{self.step_count:03d}.json"  # Use episode step count
                step_filepath = os.path.join(episode_step_dir, step_filename)
                
                with open(step_filepath, 'w') as f:
                    json.dump(_to_serializable(step_data), f, indent=2)
                
                print(f"[STEP_SAVE] 💾 Saved episode {self.episode_count} step {self.step_count}: {action_type} -> {step_filepath}")
            
            # Save to epoch export directory if this is the last episode of an epoch
            if self.export_last_epoch_episode and self.is_last_episode_of_epoch and self.epoch_step_export_dir:
                epoch_step_filename = f"step_{self.step_count:03d}.json"  # Use episode step count for epoch export
                epoch_step_filepath = os.path.join(self.epoch_step_export_dir, epoch_step_filename)
                
                with open(epoch_step_filepath, 'w') as f:
                    json.dump(_to_serializable(step_data), f, indent=2)
                
                print(f"[EPOCH_EXPORT] 📊 Saved epoch step {self.step_count}: {action_type} -> {epoch_step_filepath}")
            
        except Exception as e:
            print(f"[ERROR] Failed to save step file: {e}")

    def _save_episode_actions(self):
        """Save the current episode's actions to a JSON file"""
        if not self.current_episode_actions:
            return
        
        # Frequency gating semantics:
        # 0 => disabled, 1 => every episode, N>1 => every N episodes
        if self.log_actions_every == 0:
            return
        if self.log_actions_every > 1 and (self.episode_count % self.log_actions_every != 0):
            if self.log_actions_every > 0:
                print(f"[EPISODE] 📝 Episode {self.episode_count} - skipping action log (saving every {self.log_actions_every} episodes)")
            return
        
        try:
            def _to_serializable(obj):
                """Convert numpy types and other non-serializable objects to Python native types"""
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: _to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [_to_serializable(x) for x in obj]
                else:
                    return obj
            
            # Calculate aggregate reward statistics
            reward_component_sums = {
                'total_reward': 0.0,
                'cyclops_reward': 0.0,
                'karamba_reward': 0.0,
                'panel_cost_reward': 0.0,
                'panel_waste_reward': 0.0,
                'daylight_autonomy_reward': 0.0,
                'repulsor_penalty': 0.0,
                'weighted_cyclops': 0.0,
                'weighted_karamba': 0.0,
                'weighted_panel_cost': 0.0,
                'weighted_panel_waste': 0.0,
                'weighted_daylight_autonomy': 0.0,
                'toggle_penalty': 0.0
            }
            
            # Sum up reward components across all steps
            for step_info in self.current_episode_infos:
                if "reward_dict" in step_info:
                    for key, value in step_info["reward_dict"].items():
                        if key in reward_component_sums:
                            reward_component_sums[key] += float(value)
            
            # Create episode summary
            episode_data = {
                "episode_number": self.episode_count,
                "total_steps": len(self.current_episode_actions),
                "total_reward": sum(self.current_episode_rewards),
                "final_voxels": int(np.sum(self.grid)),
                "epw_file": os.path.basename(self.current_epw) if self.current_epw else None,
                "grid_size": [self.gx, self.gy, self.gz],
                "root_voxel": list(self.root_voxel),
                "final_facade_params": {k: v.tolist() for k, v in self.current_facade_params.items()},
                "reward_component_totals": reward_component_sums,
                "actions": []
            }
            
            # Process each action
            for step, (action_idx, reward) in enumerate(zip(self.current_episode_actions, self.current_episode_rewards)):
                voxel_part, facade_part = self._decode_combined_action(action_idx)
                action_type, coords_or_params = voxel_part
                
                action_info = {
                    "step": step,
                    "action_idx": action_idx,
                    "action_type": action_type,
                    "reward": reward,
                    "voxels": voxel_part,
                    "facade": facade_part
                }
                
                # Add specific action details
                if action_type in ["add", "delete"] and coords_or_params is not None:
                    try:
                        x, y, z = coords_or_params
                        if x is not None and y is not None and z is not None:
                            action_info["coordinates"] = [int(x), int(y), int(z)]
                    except (ValueError, TypeError, IndexError):
                        pass  # Skip if coordinates can't be converted
                elif action_type == "done":
                    action_info["early_termination"] = True
                
                # Add facade information if present
                if facade_part is not None:
                    _, (param_name, direction_idx, value, direction_name) = facade_part
                    action_info["facade_change"] = {
                        "parameter": param_name,
                        "direction": direction_name,
                        "direction_idx": int(direction_idx),
                        "value": int(value)
                    }
                
                # Add any additional info from the step
                if step < len(self.current_episode_infos):
                    step_info = self.current_episode_infos[step]
                    if "unstuck" in step_info:
                        action_info["unstuck"] = step_info["unstuck"]
                    # Add reward components if available
                    if "reward_dict" in step_info:
                        action_info["reward_components"] = step_info["reward_dict"]
                
                episode_data["actions"].append(action_info)
            
            # Create subfolder with model name and epoch, and episode-specific subfolder
            os.makedirs(self.actions_output_dir, exist_ok=True)
            
            # Create episode-specific subfolder within the main folder
            episode_subfolder = os.path.join(self.actions_output_dir, f"episode_{self.episode_count:04d}")
            os.makedirs(episode_subfolder, exist_ok=True)
            
            # Save episode summary to episode subfolder
            filename = f"episode_summary_port_{self.port}.json"
            filepath = os.path.join(episode_subfolder, filename)
            
            with open(filepath, 'w') as f:
                json.dump(_to_serializable(episode_data), f, indent=2)
            
            print(f"[EPISODE]  Saved episode {self.episode_count} actions to: {filepath}")
            print(f"[EPISODE]  {len(self.current_episode_actions)} steps, total reward: {sum(self.current_episode_rewards):.4f}")
            
            # Also save the final grid state in the same episode subfolder
            grid_filename = f"final_grid_port_{self.port}.json"
            grid_filepath = os.path.join(episode_subfolder, grid_filename)
            export_voxel_grid(self.grid, grid_filepath)
            
        except Exception as e:
            print(f"[ERROR] Failed to save episode actions: {e}")

    def render(self):
        print(self.grid)

    def _write_grid_to_temp_file(self):
        export_voxel_grid(self.grid, "temp_voxel_input.json")

    # def _wait_for_external_reward(self, timeout=30):
    #     # Keep this method for compatibility if needed elsewhere
    #     global timeouts
    #     reward_file = "temp_reward.json"
    #     start_time = time.time()
    #     while True:
    #         if os.path.exists(reward_file):
    #             try:
    #                 with open(reward_file, 'r') as f:
    #                     content = f.read().strip()
    #                     if not content:
    #                         raise ValueError("Empty file")
    #                     data = json.loads(content)
    #                 for _ in range(10):
    #                     try:
    #                         os.remove(reward_file)
    #                         break
    #                     except PermissionError:
    #                         time.sleep(0.03)
    #                 return float(data.get("reward", 0.0))
    #             except (json.JSONDecodeError, ValueError) as e:
    #                 print(f"Waiting for valid reward file... ({e})")
    #                 time.sleep(0.1)
    #         else:
    #             if time.time() - start_time > timeout:
    #                 self.timeouts += 1
    #                 print("Timeout: No external reward received. {} times".format(self.timeouts))
    #                 return 0.0
    #             time.sleep(0.1)


def export_voxel_grid(grid, filename):
    def _to_serializable(obj):
        """Convert numpy types and other non-serializable objects to Python native types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [_to_serializable(x) for x in obj]
        else:
            return obj
    
    data = {"voxels": grid.tolist()}
    with open(filename, 'w') as f:
        json.dump(_to_serializable(data), f)


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
