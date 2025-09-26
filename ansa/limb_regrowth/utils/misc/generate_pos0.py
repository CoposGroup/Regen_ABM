"""
Generate new cell initialization file for limb regeneration simulation
Places cells within epithelium boundary while avoiding bone region
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
import os

def point_in_polygon(point, polygon):
    """
    Ray casting algorithm to determine if point is inside polygon
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def generate_random_points_in_polygon(polygon, n_points, min_distance=0.05, max_attempts=50000):
    """
    Generate exactly n_points random points inside a polygon with adaptive distance constraint
    """
    polygon = np.array(polygon)
    x_min, x_max = polygon[:, 0].min(), polygon[:, 0].max()
    y_min, y_max = polygon[:, 1].min(), polygon[:, 1].max()
    
    points = []
    attempts = 0
    current_min_distance = min_distance
    last_progress = 0
    
    print(f"Generating {n_points} points with initial min_distance = {min_distance:.3f}")
    
    while len(points) < n_points and attempts < max_attempts:
        # Generate random point in bounding box
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        candidate = np.array([x, y])
        
        # Check if point is inside polygon
        if point_in_polygon(candidate, polygon):
            # Check minimum distance to existing points
            if len(points) == 0:
                points.append(candidate)
                print(f"Progress: {len(points)}/{n_points} points placed")
            else:
                distances = np.array([np.linalg.norm(candidate - p) for p in points])
                if np.all(distances >= current_min_distance):
                    points.append(candidate)
                    if len(points) % 50 == 0:  # Progress update every 50 points
                        print(f"Progress: {len(points)}/{n_points} points placed")
        
        attempts += 1
        
        # Adaptive strategy: reduce min_distance if we're stuck
        if attempts % 10000 == 0 and len(points) == last_progress:
            current_min_distance *= 0.9  # Reduce by 10%
            print(f"Reducing min_distance to {current_min_distance:.3f} (attempt {attempts})")
        
        if attempts % 10000 == 0:
            last_progress = len(points)
    
    # If we still don't have enough points, use a more aggressive approach
    if len(points) < n_points:
        print(f"Switching to aggressive placement for remaining {n_points - len(points)} points...")
        remaining_attempts = 0
        while len(points) < n_points and remaining_attempts < 20000:
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            candidate = np.array([x, y])
            
            if point_in_polygon(candidate, polygon):
                # Much more relaxed distance constraint
                if len(points) == 0:
                    points.append(candidate)
                else:
                    distances = np.array([np.linalg.norm(candidate - p) for p in points])
                    if np.all(distances >= current_min_distance * 0.5):  # Half the distance
                        points.append(candidate)
            
            remaining_attempts += 1
    
    return np.array(points)

def generate_grid_points_in_polygon(polygon, n_points_target=1500):
    """
    Generate approximately n_points_target grid points inside polygon
    Automatically adjusts spacing to hit target
    """
    polygon = np.array(polygon)
    x_min, x_max = polygon[:, 0].min(), polygon[:, 0].max()
    y_min, y_max = polygon[:, 1].min(), polygon[:, 1].max()
    
    # Estimate area of polygon (rough approximation)
    area_bbox = (x_max - x_min) * (y_max - y_min)
    
    # Estimate spacing needed for target number of points
    estimated_spacing = np.sqrt(area_bbox / n_points_target)
    
    print(f"Estimating grid spacing: {estimated_spacing:.3f}")
    
    # Try different spacings to get close to target
    best_points = []
    best_count_diff = float('inf')
    
    for spacing_factor in [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]:
        spacing = estimated_spacing * spacing_factor
        
        # Create grid
        x_grid = np.arange(x_min, x_max + spacing, spacing)
        y_grid = np.arange(y_min, y_max + spacing, spacing)
        
        points = []
        for x in x_grid:
            for y in y_grid:
                candidate = np.array([x, y])
                if point_in_polygon(candidate, polygon):
                    points.append(candidate)
        
        count_diff = abs(len(points) - n_points_target)
        print(f"Spacing {spacing:.3f}: {len(points)} points (diff: {count_diff})")
        
        if count_diff < best_count_diff:
            best_count_diff = count_diff
            best_points = points
    
    return np.array(best_points)

def ensure_exact_count(points, bone_polygon, epithelium_polygon, target_count=1500):
    """
    Ensure we have exactly the target number of points
    Add or remove points as needed
    """
    current_count = len(points)
    
    if current_count == target_count:
        print(f"Perfect! Generated exactly {target_count} cells")
        return points
    
    elif current_count > target_count:
        # Too many points - randomly remove excess
        excess = current_count - target_count
        print(f"Removing {excess} excess points randomly...")
        indices_to_keep = np.random.choice(current_count, target_count, replace=False)
        return points[indices_to_keep]
    
    else:
        # Too few points - add more with very relaxed constraints
        deficit = target_count - current_count
        print(f"Adding {deficit} more points with relaxed constraints...")
        
        epithelium = np.array(epithelium_polygon)
        x_min, x_max = epithelium[:, 0].min(), epithelium[:, 0].max()
        y_min, y_max = epithelium[:, 1].min(), epithelium[:, 1].max()
        
        additional_points = []
        attempts = 0
        max_attempts = deficit * 1000
        
        while len(additional_points) < deficit and attempts < max_attempts:
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            candidate = np.array([x, y])
            
            # Check if inside epithelium and outside bone
            if point_in_polygon(candidate, epithelium_polygon):
                if len(bone_polygon) == 0 or not point_in_polygon(candidate, bone_polygon):
                    # Very relaxed distance constraint - just avoid overlap
                    if len(points) == 0:
                        additional_points.append(candidate)
                    else:
                        all_existing = np.vstack([points, additional_points]) if len(additional_points) > 0 else points
                        distances = np.array([np.linalg.norm(candidate - p) for p in all_existing])
                        if np.all(distances >= 0.02):  # Very small minimum distance
                            additional_points.append(candidate)
            
            attempts += 1
        
        if len(additional_points) == deficit:
            print(f"Successfully added {len(additional_points)} points")
            return np.vstack([points, additional_points])
        else:
            print(f"Could only add {len(additional_points)} out of {deficit} needed points")
            return np.vstack([points, additional_points]) if len(additional_points) > 0 else points

def filter_points_outside_bone(points, bone_polygon):
    """
    Remove points that are inside the bone region
    """
    if len(bone_polygon) == 0:
        return points
    
    filtered_points = []
    for point in points:
        if not point_in_polygon(point, bone_polygon):
            filtered_points.append(point)
    
    return np.array(filtered_points)

def create_cell_initialization(epithelium_file, bone_file, output_file, 
                             n_cells_exact=1500, method='random', 
                             min_distance=0.05, include_bone=True):
    """
    Create new cell initialization file with EXACTLY n_cells_exact cells
    
    Parameters:
    ----------
    epithelium_file : str
        Path to epithelium boundary CSV file
    bone_file : str  
        Path to bone boundary CSV file
    output_file : str
        Path for output MATLAB file
    n_cells_exact : int
        Exact number of cells to create
    method : str
        'random' or 'grid' - method for generating points
    min_distance : float
        Initial minimum distance between cells (may be reduced if needed)
    include_bone : bool
        Whether to include the bone region (default: True)
    """
    # Load boundaries
    print("Loading boundary files...")
    try:
        epithelium = np.loadtxt(epithelium_file, delimiter=',')
        print(f"Loaded epithelium boundary: {len(epithelium)} points")
    except Exception as e:
        print(f"Error loading epithelium file: {e}")
        return
    bone = np.array([])
    if include_bone:
        try:
            bone = np.loadtxt(bone_file, delimiter=',')
            print(f"Loaded bone boundary: {len(bone)} points")
        except Exception as e:
            print(f"Warning: Could not load bone file: {e}")
            bone = np.array([])  # Empty array if no bone file
    # Generate initial points based on method
    print(f"Generating cell positions using {method} method...")
    if method == 'random':
        points = generate_random_points_in_polygon(
            epithelium, int(n_cells_exact * 1.2), min_distance  # 20% extra initially
        )
    elif method == 'grid':
        points = generate_grid_points_in_polygon(epithelium, int(n_cells_exact * 1.2))
    else:
        raise ValueError("Method must be 'random' or 'grid'")
    print(f"Generated {len(points)} initial points in epithelium")
    # Remove points inside bone region
    if include_bone and len(bone) > 0:
        points_before = len(points)
        points = filter_points_outside_bone(points, bone)
        points_removed = points_before - len(points)
        print(f"Removed {points_removed} points inside bone region")
        print(f"Points after bone filtering: {len(points)}")
    # Ensure we have exactly the target number
    points = ensure_exact_count(points, bone, epithelium, n_cells_exact)
    final_count = len(points)
    print(f"Final cell count: {final_count}")
    if final_count != n_cells_exact:
        print(f"Warning: Could not achieve exactly {n_cells_exact} cells. Got {final_count}")
    if len(points) == 0:
        print("Error: No valid cell positions generated!")
        return
    # Save to MATLAB format
    print(f"Saving to {output_file}...")
    savemat(output_file, {
        'pos0': points,
        'Ncells': len(points)
    })
    print(f"Successfully created initialization file with {len(points)} cells")
    # Create visualization
    visualize_initialization(epithelium, bone, points, output_file.replace('.mat', '.png'))
    return points

def visualize_initialization(epithelium, bone, cell_positions, output_image):
    """
    Create visualization of the initialization
    """
    plt.figure(figsize=(12, 8))
    
    # Plot epithelium boundary
    epi_x, epi_y = epithelium[:, 0], epithelium[:, 1]
    plt.plot(np.append(epi_x, epi_x[0]), np.append(epi_y, epi_y[0]), 
             'r-', linewidth=2, label='Epithelium boundary')
    
    # Plot bone boundary if available
    if len(bone) > 0:
        bone_x, bone_y = bone[:, 0], bone[:, 1]
        plt.plot(np.append(bone_x, bone_x[0]), np.append(bone_y, bone_y[0]), 
                 'k-', linewidth=3, label='Bone boundary')
        plt.fill(bone_x, bone_y, color='gray', alpha=0.3, label='Bone region')
    
    # Plot cell positions
    if len(cell_positions) > 0:
        plt.scatter(cell_positions[:, 0], cell_positions[:, 1], 
                   c='blue', s=20, alpha=0.6, label=f'Cells ({len(cell_positions)})')
    
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('Cell Initialization Layout')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    
    # Save visualization
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {output_image}")

# Example usage
if __name__ == "__main__":
    # File paths - adjust these to match your file structure
    epithelium_file = "data/input/epi0.csv"
    bone_file = "data/input/bone.csv" 
    output_file = "cellinitialization_1500.mat"
    print("=== Creating exactly 1500 cells (random distribution) ===")
    create_cell_initialization(
        epithelium_file=epithelium_file,
        bone_file=bone_file,
        output_file=output_file,
        n_cells_exact=500,
        method='random',
        min_distance=0,  # Will be reduced automatically if needed: should be 0.06
        include_bone=False  # Set to False to ignore bone
    )
    print("\nDone! File contains exactly 1500 randomly distributed cells.")