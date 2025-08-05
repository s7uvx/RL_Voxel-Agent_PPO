# import compute_rhino3d.Util
import compute_rhino3d.Grasshopper as gh
# import os
import json
import gc

def weird_str_to_float(input_string: str) -> float:
        numeric_part = ''.join(char for char in input_string if char.isdigit() or char == '.')
        return float(numeric_part)

def get_reward_gh(grid, cyclops_gh_file, karamba_gh_file, cyclops_wt, karamba_wt) -> float:
        if grid is None:
            return -0.2
        else:
            
            data = json.dumps({"voxels": grid.tolist()})
            voxel_in = gh.DataTree("voxel_json")
            voxel_in.Append([0], [data])
            try:
                cyclops_output = gh.EvaluateDefinition(cyclops_gh_file, [voxel_in], )
                cyclops_reward = weird_str_to_float(cyclops_output['values'][0]['InnerTree']['{0;0}'][0]['data'])
            except:
                cyclops_reward = -0.2
                print('error on cyclops')
            try:
                karamba_output = gh.EvaluateDefinition(karamba_gh_file, [voxel_in])
                karamba_reward = weird_str_to_float(karamba_output['values'][0]['InnerTree']['{0}'][0]['data'])
            except:
                karamba_reward = -0.2
                print('error on karamba')
            
            print('cyclops_reward: ',cyclops_reward*cyclops_wt)
            print('karamba reward: {}'.format(karamba_reward*karamba_wt))
            
            gc.collect()

            reward = cyclops_wt * cyclops_reward + karamba_wt * karamba_reward
            print('reward: ',reward)
            return reward