import numpy as np
import os
import sys
import traceback
from daylight_analyzer import VoxelDaylightAnalyzer, analyze_voxel_daylight

def test_single_voxel():
    """Test VoxelDaylightAnalyzer with a single 1x1x1 voxel"""
    print("🧪 Testing VoxelDaylightAnalyzer with 1x1x1 voxel")
    print("=" * 50)
    
    # Create a minimal 1x1x1 voxel grid with one occupied voxel
    voxel_grid = np.array([[[1]]], dtype=np.int32)
    print(f"📦 Voxel grid shape: {voxel_grid.shape}")
    print(f"📦 Occupied voxels: {np.sum(voxel_grid)}")
    print(f"📦 Grid contents:\n{voxel_grid}")
    
    # Test the analyzer directly
    print("\n🔬 Testing VoxelDaylightAnalyzer class...")
    analyzer = VoxelDaylightAnalyzer(voxel_size=1.0)
    
    # Test point conversion
    print("🔄 Testing voxel to points conversion...")
    points = analyzer.voxels_to_points(voxel_grid)
    print(f"✅ Generated {len(points)} points from voxels")
    for i, point in enumerate(points):
        print(f"   Point {i}: ({point.x}, {point.y}, {point.z})")
    
    # Test model creation
    print("🔄 Testing honeybee model creation...")
    model = analyzer.create_honeybee_model(voxel_grid)
    print(f"✅ Created model with {len(model.rooms)} rooms")
    print(f"   Model identifier: {model.identifier}")
    
    # Test sensor grid setup
    print("🔄 Testing sensor grid setup...")
    sensor_grid = analyzer.setup_sensor_grid(model, grid_size=1.0)
    print(f"✅ Created sensor grid with {len(sensor_grid.sensors)} sensors")
    
    # Test full radiation analysis
    print("🔄 Testing full radiation analysis...")
    results = analyzer.analyze_cumulative_radiation(voxel_grid)
    
    if 'error' in results:
        print(f"❌ Analysis failed: {results['error']}")
        return False
    else:
        print("✅ Radiation analysis completed successfully!")
        print("\n📊 Analysis Results:")
        for key, value in results.items():
            if isinstance(value, dict):
                print(f"   {key}:")
                for sub_key, sub_value in value.items():
                    print(f"     {sub_key}: {sub_value}")
            else:
                print(f"   {key}: {value}")
        
        # Export results
        output_file = "test_single_voxel_results.json"
        analyzer.export_results(results, output_file)
        
        return True

def test_convenience_function():
    """Test the convenience function with a single voxel"""
    print("\n🧪 Testing convenience function analyze_voxel_daylight")
    print("=" * 50)
    
    # Create a minimal 1x1x1 voxel grid
    voxel_grid = np.array([[[1]]], dtype=np.int32)
    
    score = analyze_voxel_daylight(voxel_grid, "test_convenience_results.json")
    print(f"✅ Daylight score: {score}")
    print(f"📄 Results saved to: test_convenience_results.json")
    return score > 0

def test_multiple_voxels():
    """Test with a slightly larger 2x2x2 grid"""
    print("\n🧪 Testing with 2x2x2 voxel grid")
    print("=" * 50)
    
    # Create a 2x2x2 grid with 4 occupied voxels in a pattern
    voxel_grid = np.zeros((2, 2, 2), dtype=np.int32)
    voxel_grid[0, 0, 0] = 1  # Corner voxel
    voxel_grid[1, 1, 1] = 1  # Opposite corner voxel
    voxel_grid[0, 1, 0] = 1  # Additional voxel
    voxel_grid[1, 0, 1] = 1  # Additional voxel
    
    print(f"📦 Voxel grid shape: {voxel_grid.shape}")
    print(f"📦 Occupied voxels: {np.sum(voxel_grid)}")
    print(f"📦 Grid pattern:")
    for z in range(voxel_grid.shape[2]):
        print(f"   Z-level {z}:")
        print(f"   {voxel_grid[:, :, z]}")
    
    score = analyze_voxel_daylight(voxel_grid, "test_multiple_voxels_results.json")
    print(f"✅ Daylight score for 4-voxel pattern: {score}")
    return score > 0

def test_voxel_env_integration():
    """Test integration with VoxelEnv from the RL training"""
    print("\n🧪 Testing integration with VoxelEnv")
    print("=" * 50)
    
    from voxel_env import VoxelEnv
    
    # Create a VoxelEnv and take some steps
    env = VoxelEnv(port=6500, grid_size=3, device='cpu')
    obs, info = env.reset()
    
    # Take a few random steps to build some geometry
    for _ in range(3):
        if len(env.available_actions) > 0:
            action_idx = 0  # Take first available action
            obs, reward, terminated, truncated, info = env.step(action_idx)
    
    print(f"📦 VoxelEnv grid shape: {env.grid.shape}")
    print(f"📦 Occupied voxels: {np.sum(env.grid)}")
    print("📦 Final grid state:")
    print(env.grid)
    
    # Test daylight analysis on the env grid
    score = analyze_voxel_daylight(env.grid, "test_voxel_env_results.json")
    print(f"✅ Daylight score for VoxelEnv grid: {score}")
    return score >= 0  # Accept any non-negative score

def run_all_tests():
    """Run all tests and summarize results"""
    print("🚀 Starting VoxelDaylightAnalyzer Test Suite")
    print("=" * 60)
    
    test_results = []
    
    # Test 1: Single voxel
    test_results.append(("Single Voxel Test", test_single_voxel()))
    
    # Test 2: Convenience function
    test_results.append(("Convenience Function Test", test_convenience_function()))
    
    # Test 3: Multiple Voxels
    test_results.append(("Multiple Voxels Test", test_multiple_voxels()))
    
    # Test 4: VoxelEnv integration
    test_results.append(("VoxelEnv Integration Test", test_voxel_env_integration()))
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 30)
    passed = 0
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall Result: {passed}/{len(test_results)} tests passed")
    
    if passed == len(test_results):
        print("🎉 All tests passed! VoxelDaylightAnalyzer is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == len(test_results)

if __name__ == "__main__":
    # Check if dependencies are available
    from daylight_analyzer import VoxelDaylightAnalyzer
    print("✅ VoxelDaylightAnalyzer imported successfully")
    
    # Run tests
    success = run_all_tests()
    
    # Clean up test files (optional)
    test_files = [
        "test_single_voxel_results.json",
        "test_convenience_results.json", 
        "test_multiple_voxels_results.json",
        "test_voxel_env_results.json"
    ]
    
    print(f"\n🧹 Test files created: {', '.join(test_files)}")
    print("   (You can delete these files if not needed)")
    
    sys.exit(0 if success else 1)
