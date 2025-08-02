import json
import os
import rhinoinside
rhinoinside.load(8)
import System
import Rhino.Geometry as rg

path = os.path.join(os.getcwd(), 'temp_voxel_input.json')
boxes = []

if os.path.exists(path):
    with open(path, 'r') as f:
        voxels = json.load(f)["voxels"]

    size = len(voxels)
    boxes = []  # Pre-allocate if you know approximate size
    
    # Cache frequently used constructors
    Point3d = rg.Point3d
    BoundingBox = rg.BoundingBox
    Box = rg.Box
    
    for x in range(size):
        voxels_x = voxels[x]  # Cache row access
        for y in range(size):
            voxels_xy = voxels_x[y]  # Cache column access
            for z in range(size):
                if voxels_xy[z] == 1:
                    # Direct box creation without intermediate variables
                    boxes.append(Box(BoundingBox(Point3d(x, y, z), Point3d(x + 1, y + 1, z + 1))))

# Output: list of voxel Boxes
a = boxes