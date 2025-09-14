import compute_rhino3d.Grasshopper as gh
import json
import gc
import os
import numpy as np

def weird_str_to_float(input_string: str) -> float:
    s = ''.join(ch for ch in str(input_string) if ch.isdigit() or ch in '.-')
    try:
        v = float(s)
        if not np.isfinite(v):
            return 0.0
        return v
    except Exception:
        return 0.0  # neutral fallback

def get_reward_gh(
    grid,
    merged_gh_file,
    epw_file,
    sun_wt: float = 0.3,
    str_wt: float = 0.5,
    cst_wt: float = 0.1,
    wst_wt: float = 0.1,
    day_wt: float = 0.1,
    repulsors: list | None = None,
    repulsor_wt: float = 0.0,
    repulsor_radius: float = 2.0,
    facade_params: dict | None = None,
) -> tuple[float, dict[str, float]]:
    if grid is None:
        return -0.2, {
            'total_reward': -0.2,
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

    # ✅ Convert EPW path to forward slashes
    epw_file = epw_file.replace("\\", "/")

    # Prepare voxel grid as JSON
    voxel_grid = json.dumps({"voxels": grid.tolist()})
    voxel_in = gh.DataTree("voxel_json")
    voxel_in.Append([0], [voxel_grid])

    # Send EPW path to Hops
    epw_path = gh.DataTree("epw_path")
    epw_path.Append([0], [epw_file])

    # Prepare facade parameters
    if facade_params is None:
        # Default facade parameters if none provided
        facade_params = {
            'cols': np.array([3,3,3,3], dtype=np.int32),
            'rows': np.array([4,4,4,4], dtype=np.int32)
        }

    # Ensure min <= max constraints
    cols = np.array(facade_params['cols'], dtype=int)
    rows = np.array(facade_params['rows'], dtype=int)
     

    # Convert to JSON lists and create DataTrees for Grasshopper
    # cols_min_json = json.dumps(min_cols.tolist())
    cols_tree = gh.DataTree("cols")
    cols_tree.Append([0], cols.tolist())

    rows_tree = gh.DataTree("rows")
    rows_tree.Append([0], rows.tolist())

    trees = [epw_path, voxel_in, cols_tree, rows_tree]

    # Evaluate Grasshopper definition with all input trees
    output = gh.EvaluateDefinition(
        merged_gh_file, 
        trees
    )
    # print(output)

    # Extract reward values

    cyclops_reward_str = output['values'][0]['InnerTree']['{0;0}'][0]['data']
    cyclops_reward = weird_str_to_float(cyclops_reward_str)

    karamba_reward_str = output['values'][1]['InnerTree']['{0}'][0]['data']
    karamba_reward = weird_str_to_float(karamba_reward_str)

    panel_cost_reward_str = output['values'][2]['InnerTree']['{0}'][0]['data']
    panel_cost_reward = weird_str_to_float(panel_cost_reward_str)

    panel_waste_reward_str = output['values'][3]['InnerTree']['{0;0}'][0]['data']
    panel_waste_reward = weird_str_to_float(panel_waste_reward_str)

    daylight_autonomty_reward_str = output['values'][4]['InnerTree']['{0;0;0}'][0]['data']
    daylight_autonomty_reward = weird_str_to_float(daylight_autonomty_reward_str)

    # Weighted reward from Grasshopper
    reward = (
        sun_wt * cyclops_reward
        + str_wt * karamba_reward
        + cst_wt * panel_cost_reward
        + wst_wt * panel_waste_reward
        + day_wt * daylight_autonomty_reward
    )

    # Optional: soft clamp to avoid extreme magnitudes that destabilize PPO
    reward = float(np.clip(reward, -100.0, 100.0))

    # Optional repulsor proximity penalty (0..repulsor_wt)
    repulsor_penalty = 0.0
    try:
        if repulsor_wt > 0 and repulsors:
            rep_arr = np.asarray(repulsors, dtype=float)
            if rep_arr.ndim == 1:
                rep_arr = rep_arr.reshape(1, -1)
            occ = np.argwhere(np.asarray(grid) == 1)
            if occ.size > 0 and rep_arr.size > 0:
                # pairwise distances (n_voxels x n_repulsors)
                diff = occ[:, None, :] - rep_arr[None, :, :]
                dists = np.linalg.norm(diff, axis=2)
                # linear falloff to 0 at radius
                with np.errstate(invalid='ignore'):
                    p = 1.0 - (dists / max(1e-6, float(repulsor_radius)))
                p = np.clip(p, 0.0, 1.0)
                proximity = float(p.mean())  # normalized 0..1
                repulsor_penalty = repulsor_wt * proximity
                reward -= repulsor_penalty
    except Exception as e:
        repulsor_penalty = 0.0

    print(
        f"[REWARD] Cyclops: {sun_wt * cyclops_reward:.2f}, "
        f"Karamba: {str_wt * karamba_reward:.2f}, "
        f"Cost: {cst_wt * panel_cost_reward:.2f}, "
        f"Waste: {wst_wt * panel_waste_reward:.2f}, "
        f"Repulsor: -{repulsor_penalty:.2f} - Total: {reward:.2f} | "
        f"Facade: cols {cols.tolist()}, rows {rows.tolist()}, daylight autonomy {daylight_autonomty_reward:.2f}"
    )

    # Return both total reward and individual components
    reward_components = {
        'total_reward': reward,
        'cyclops_reward': cyclops_reward,
        'karamba_reward': karamba_reward,
        'panel_cost_reward': panel_cost_reward,
        'panel_waste_reward': panel_waste_reward,
        'daylight_autonomy_reward': daylight_autonomty_reward,
        'repulsor_penalty': repulsor_penalty,
        'weighted_cyclops': sun_wt * cyclops_reward,
        'weighted_karamba': str_wt * karamba_reward,
        'weighted_panel_cost': cst_wt * panel_cost_reward,
        'weighted_panel_waste': wst_wt * panel_waste_reward,
        'weighted_daylight_autonomy': day_wt * daylight_autonomty_reward
    }

    return reward, reward_components
