import numpy as np
import os
import json
import requests
import zipfile
import tempfile
from typing import List, Tuple, Dict, Optional

from ladybug_geometry.geometry3d.pointvector import Point3D, Vector3D
from ladybug_geometry.geometry3d.face import Face3D
from ladybug_geometry.geometry3d.polyface import Polyface3D
from honeybee.room import Room
from honeybee.model import Model
from honeybee_radiance.sensorgrid import SensorGrid
from honeybee_radiance.modifier.material import Plastic
from ladybug.wea import Wea
from ladybug.epw import EPW
from honeybee_radiance.lightsource.sky import CertainIrradiance
from lbt_recipes.recipe import Recipe
print("✅ All honeybee and ladybug dependencies imported successfully")


class VoxelDaylightAnalyzer:
    """
    Analyzes daylight performance of voxel-based geometries using honeybee-radiance
    """
    
    def __init__(self, voxel_size: float = 1.0, epw_url: str = "https://energyplus-weather.s3.amazonaws.com/europe_wmo_region_6/ESP/ESP_Barcelona.081810_SWEC/ESP_Barcelona.081810_SWEC.zip"):
        """
        Initialize the daylight analyzer
        
        Args:
            voxel_size: Size of each voxel in meters
            epw_url: URL to EPW weather file (defaults to Barcelona)
        """
        self.voxel_size = voxel_size
        self.epw_url = epw_url
        self.epw_file = None
        self.wea = None
        self._setup_weather_data()
        
    def _setup_weather_data(self):
        """Download and setup weather data from EPW file"""
        # Create temp directory for weather data
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "weather.zip")
        
        # Download EPW file
        print("📥 Downloading Barcelona weather data...")
        response = requests.get(self.epw_url)
        response.raise_for_status()
        
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        
        # Extract EPW file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_name in zip_ref.namelist():
                if file_name.endswith('.epw'):
                    zip_ref.extract(file_name, temp_dir)
                    self.epw_file = os.path.join(temp_dir, file_name)
                    break
        
        if self.epw_file and os.path.exists(self.epw_file):
            # Create WEA object for sky conditions
            epw = EPW(self.epw_file)
            self.wea = Wea.from_epw_file(self.epw_file)
            print(f"✅ Weather data loaded: {epw.location}")
        else:
            print("❌ Could not find EPW file in archive")
    
    def voxels_to_points(self, voxel_grid: np.ndarray) -> List[Point3D]:
        """
        Convert voxel grid to Point3D objects
        
        Args:
            voxel_grid: 3D numpy array where 1 indicates occupied voxel
            
        Returns:
            List of Point3D objects for occupied voxels
        """
        points = []
        occupied_indices = np.argwhere(voxel_grid == 1)
        
        for idx in occupied_indices:
            x, y, z = idx
            # Center the point in the voxel
            point = Point3D(x,y,z)
            points.append(point)
            
        return points
    
    def create_voxel_room(self, center_point: Point3D, identifier: str) -> Room:
        """
        Create a honeybee Room from a voxel center point
        
        Args:
            center_point: Center point of the voxel
            identifier: Unique identifier for the room
            
        Returns:
            Honeybee Room object
        """
        return Room.from_box(identifier, 1.0, 1.0, 1.0, 0, origin=center_point)
    
    def create_honeybee_model(self, voxel_grid: np.ndarray) -> Model:
        """
        Create a honeybee Model from voxel grid
        
        Args:
            voxel_grid: 3D numpy array where 1 indicates occupied voxel
            
        Returns:
            Honeybee Model object
        """
        points = self.voxels_to_points(voxel_grid)
        rooms = []
        
        for i, point in enumerate(points):
            room_id = f"VoxelRoom_{i:03d}"
            room = self.create_voxel_room(point, room_id)
            rooms.append(room)
        
        # Create model
        model = Model("VoxelModel", rooms)
        
        # Add default materials
        default_material = Plastic.from_single_reflectance("DefaultMaterial", 0.4)
        for room in model.rooms:
            for face in room.faces:
                face.properties.radiance.modifier = default_material
        
        return model
    
    def setup_sensor_grid(self, model: Model, grid_size: float = 0.5) -> SensorGrid:
        """
        Create sensor grid for radiation analysis
        
        Args:
            model: Honeybee model
            grid_size: Size of sensor grid spacing
            
        Returns:
            SensorGrid object
        """
        grid = model.generate_exterior_face_grid(grid_size)
        sensor_grid = SensorGrid.from_mesh3d('VoxelSensors', grid)
        model.properties.radiance.add_sensor_grid(sensor_grid)
        
        return sensor_grid
    
    def analyze_cumulative_radiation(self, voxel_grid: np.ndarray, analysis_period: Optional[Tuple] = None) -> Dict:
        """
        Perform cumulative radiation analysis on voxel geometry
        
        Args:
            voxel_grid: 3D numpy array where 1 indicates occupied voxel
            analysis_period: Tuple of (start_month, start_day, start_hour, end_month, end_day, end_hour)
                           If None, analyzes full year
            
        Returns:
            Dictionary containing analysis results
        """
        print("🔄 Creating honeybee model from voxels...")
        model = self.create_honeybee_model(voxel_grid)

        print("🔄 Setting up sensor grid...")
        sensor_grid = self.setup_sensor_grid(model)
        
        print("🔄 Preparing radiation analysis...")
        
        # Create sky for analysis
        if self.wea:
            # Use actual weather data
            from honeybee_radiance.lightsource.sky import SkyMatrix
            sky = SkyMatrix(self.wea.filter_by_hoys(list(range(8759))))
            print("☀️ Using Barcelona weather data for sky conditions")
        else:
            # Use default clear sky
            sky = CertainIrradiance(1000)  # 1000 W/m2 irradiance
            print("☀️ Using default clear sky conditions")
        
        # Setup annual irradiance recipe
        recipe = Recipe('annual-irradiance')

        recipe.input_value_by_name('wea', self.wea)
        recipe.input_value_by_name('model', model)

        # Run analysis
        print("🔄 Running radiation analysis...")
        recipe.run(radiance_check=True, silent=True)

        results = recipe.outputs
        simulation_folder = os.path.join(recipe.default_project_folder, recipe.simulation_id)
        # Process results
        if results and len(results) > 0:
            average_irradiance = results[0].value(simulation_folder) 
            cumulative_radiation = results[1].value(simulation_folder)
            peak_irradiance = results[2].value(simulation_folder)

            # Calculate radiation on model surfaces
            surface_radiation = self._calculate_surface_radiation(model, cumulative_radiation)
            
            analysis_results = {
                'total_radiation_kwh_m2': (np.array(cumulative_radiation) / 1000.0).tolist(),  # Convert Wh to kWh
                'average_irradiance' : (np.array(average_irradiance) / 1000.0).tolist(),
                'peak_irradiance' : (np.array(peak_irradiance) / 1000.0).tolist(),
                'surface_radiation': surface_radiation,
                'sensor_count': sensor_grid.count,
                'voxel_count': np.sum(voxel_grid),
                'model_volume_m3': np.sum(voxel_grid) * (self.voxel_size ** 3)
            }
            
            print("✅ Radiation analysis completed successfully")
            return analysis_results
        else:
            print("❌ No results returned from analysis")
            return {'error': 'Analysis failed to return results'}
    
    def _calculate_surface_radiation(self, model: Model, irradiance_values: np.ndarray) -> Dict:
        """Calculate radiation received by each surface of the model"""
        surface_radiation = {}
        
        for room in model.rooms:
            for face in room.faces:
                # Simple approximation: assign average radiation to each face
                # In a full implementation, this would use proper ray tracing
                face_area = face.area
                avg_radiation = np.mean(irradiance_values)
                
                surface_radiation[face.identifier] = {
                    'area_m2': face_area,
                    'radiation_kwh_m2': avg_radiation / 1000.0,
                    'total_radiation_kwh': (avg_radiation * face_area) / 1000.0
                }
        
        return surface_radiation
    
    def export_results(self, results: Dict, filename: str):
        """Export analysis results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📄 Results exported to: {filename}")


# Example usage function
def analyze_voxel_daylight(voxel_grid: np.ndarray, output_file: str = "daylight_analysis.json") -> float:
    """
    Convenience function to analyze daylight performance of a voxel grid
    
    Args:
        voxel_grid: 3D numpy array where 1 indicates occupied voxel
        output_file: Filename for results export
        
    Returns:
        Total radiation score (higher is better for daylight access)
    """
    analyzer = VoxelDaylightAnalyzer(voxel_size=1.0)
    results = analyzer.analyze_cumulative_radiation(voxel_grid)
    
    if 'error' not in results:
        analyzer.export_results(results, output_file)
        # Return normalized radiation score (0-1 range)
        total_radiation = results.get('total_radiation_kwh_m2', 0)
        return min(total_radiation / 1000.0, 1.0)  # Normalize to 0-1 range
    else:
        print(f"Analysis failed: {results['error']}")
        return 0.0