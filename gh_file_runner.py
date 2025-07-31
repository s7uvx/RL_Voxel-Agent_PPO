import compute_rhino3d.Util
import compute_rhino3d.Grasshopper as gh
import os

def get_reward_gh(voxel_grid: str, port: int) -> float:
    if voxel_grid is None:
        return -0.2
    else:
        compute_rhino3d.Util.url = f"http://localhost:{port}/"
        gh_file = os.path.join(os.getcwd(),'gh_files','RL_Voxel_V4_hops.gh')
        voxel_in = gh.DataTree("RH_IN:voxel_json")
        voxel_in.Append([0], [voxel_grid])
        output = gh.EvaluateDefinition(gh_file, [voxel_in])
        return float(str(output['values'][0]['InnerTree']['{0}'][0]['data']))
