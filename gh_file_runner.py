import compute_rhino3d.Grasshopper as gh
import json
import gc

def weird_str_to_float(input_string: str) -> float:
    numeric_part = ''.join(char for char in input_string if char.isdigit() or char == '.')
    try:
        return float(numeric_part)
    except ValueError:
        return -0.2  # fallback in case of error

def get_reward_gh(grid, merged_gh_file, epw_file, sun_wt, str_wt) -> float:
    if grid is None:
        return -0.2

    # Prepare voxel data
    data = json.dumps({"voxels": grid.tolist()})
    voxel_in = gh.DataTree("voxel_json")
    voxel_in.Append([0], [data])

    # Prepare EPW path (if still needed)
    epw_json = json.dumps(epw_file)
    epw_path = gh.DataTree('epw_path')
    epw_path.Append([0], [epw_json])

    try:
        # Send both inputs to the merged GH definition
        output = gh.EvaluateDefinition(merged_gh_file, [epw_path, voxel_in])

        # Adjust the InnerTree path below based on your GH output
        cyclops_reward_str = output['values'][0]['InnerTree']['{0;0}'][0]['data']
        karamba_reward_str = output['values'][1]['InnerTree']['{0}'][0]['data']
        cyclops_reward = weird_str_to_float(cyclops_reward_str)
        karamba_reward = weird_str_to_float(karamba_reward_str)
        reward = sun_wt * cyclops_reward + str_wt * karamba_reward
        print('cyclops reward {:.2f}, karamba reward {:.2f}, total reward {:.2f}'.format(sun_wt * cyclops_reward, str_wt * karamba_reward, reward))
    except Exception as e:
        print(f'[ERROR] Failed to evaluate GH definition: {e}')
        reward = -0.2

    gc.collect()
    return reward
