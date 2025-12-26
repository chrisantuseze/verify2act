"""
Collision Checker for Points2Plans Integration

Implements 2D bounding box collision detection as used in Points2Plans.
This is used during planning to filter out infeasible actions that would
result in object-object collisions.

Based on Points2Plans base_RD.py collision checking logic.
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional


class CollisionChecker:
    """
    2D collision detection for object bounding boxes.
    
    Uses projected 2D bounding boxes in the XY plane (ignoring Z for simplicity)
    to detect potential collisions in predicted states.
    """
    
    def __init__(
        self,
        x_collision: float = 0.05,
        y_collision: float = 0.05,
        z_threshold: float = 0.01,
        verbose: bool = False
    ):
        """
        Initialize collision checker.
        
        Args:
            x_collision: Half-width of bounding box in X dimension (meters)
            y_collision: Half-width of bounding box in Y dimension (meters)
            z_threshold: Vertical threshold for considering objects at same level
            verbose: Whether to print collision detection details
        """
        self.x_collision = x_collision
        self.y_collision = y_collision
        self.z_threshold = z_threshold
        self.verbose = verbose
    
    def check_2d_collision(self, bbox1: List[List[float]], bbox2: List[List[float]]) -> bool:
        """
        Check if two 2D bounding boxes collide.
        
        Uses axis-aligned bounding box (AABB) collision detection.
        Each bbox is a list of 4 corner points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        
        Args:
            bbox1: First bounding box corners
            bbox2: Second bounding box corners
        
        Returns:
            True if boxes collide, False otherwise
        """
        # Extract min/max coordinates for each box
        bbox1_array = np.array(bbox1)
        bbox2_array = np.array(bbox2)
        
        x1_min, y1_min = bbox1_array[:, 0].min(), bbox1_array[:, 1].min()
        x1_max, y1_max = bbox1_array[:, 0].max(), bbox1_array[:, 1].max()
        
        x2_min, y2_min = bbox2_array[:, 0].min(), bbox2_array[:, 1].min()
        x2_max, y2_max = bbox2_array[:, 0].max(), bbox2_array[:, 1].max()
        
        # Check for overlap (AABB collision test)
        x_overlap = (x1_min <= x2_max) and (x1_max >= x2_min)
        y_overlap = (y1_min <= y2_max) and (y1_max >= y2_min)
        
        collision = x_overlap and y_overlap
        
        if self.verbose and collision:
            print(f"  Collision detected: Box1[{x1_min:.3f},{y1_min:.3f} to {x1_max:.3f},{y1_max:.3f}] "
                  f"overlaps Box2[{x2_min:.3f},{y2_min:.3f} to {x2_max:.3f},{y2_max:.3f}]")
        
        return collision
    
    def get_object_bbox(self, center_pos: np.ndarray) -> List[List[float]]:
        """
        Get 2D bounding box corners for an object given its center position.
        
        Args:
            center_pos: Object center position [x, y, z]
        
        Returns:
            List of 4 corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        x, y = center_pos[0], center_pos[1]
        
        # Four corners of axis-aligned box
        corners = [
            [x - self.x_collision, y - self.y_collision],
            [x + self.x_collision, y - self.y_collision],
            [x - self.x_collision, y + self.y_collision],
            [x + self.x_collision, y + self.y_collision]
        ]
        
        return corners
    
    def check_scene_collisions(
        self,
        object_positions: np.ndarray,
        object_heights: Optional[np.ndarray] = None,
        exclude_indices: Optional[List[int]] = None
    ) -> Tuple[bool, List[Tuple[int, int]]]:
        """
        Check for collisions between all objects in a scene.
        
        Args:
            object_positions: Array of object center positions [N, 3] or [N, 2]
            object_heights: Optional heights (Z coordinates) [N,]
            exclude_indices: Optional list of object indices to exclude from checking
        
        Returns:
            collision_detected: True if any collision found
            collision_pairs: List of (i, j) index pairs that collide
        """
        if exclude_indices is None:
            exclude_indices = []
        
        num_objects = object_positions.shape[0]
        collision_pairs = []
        
        # Extract heights if available
        if object_heights is None:
            if object_positions.shape[1] >= 3:
                object_heights = object_positions[:, 2]
            else:
                object_heights = np.zeros(num_objects)
        
        # Check all pairs
        for i in range(num_objects):
            if i in exclude_indices:
                continue
                
            for j in range(i + 1, num_objects):
                if j in exclude_indices:
                    continue
                
                # Only check objects at similar heights (avoid checking stacked objects)
                height_diff = abs(object_heights[i] - object_heights[j])
                if height_diff > self.z_threshold:
                    continue
                
                # Get bounding boxes
                bbox_i = self.get_object_bbox(object_positions[i])
                bbox_j = self.get_object_bbox(object_positions[j])
                
                # Check collision
                if self.check_2d_collision(bbox_i, bbox_j):
                    collision_pairs.append((i, j))
                    if self.verbose:
                        print(f"  Collision between object {i} and object {j}")
        
        collision_detected = len(collision_pairs) > 0
        
        return collision_detected, collision_pairs
    
    def check_predicted_state_collisions(
        self,
        predicted_point_clouds: torch.Tensor,
        predicted_poses: Optional[torch.Tensor] = None,
        target_object_id: Optional[int] = None,
        placement_height: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Check collisions in a predicted state from dynamics model.
        
        This implements the collision checking logic from Points2Plans base_RD.py
        where it checks objects above/below the placement target.
        
        Args:
            predicted_point_clouds: Predicted point clouds [N_objects, N_points, 3]
            predicted_poses: Predicted object poses [N_objects, 6] or [N_objects, 3]
            target_object_id: ID of target object for placement
            placement_height: Expected height of placement target
        
        Returns:
            is_feasible: True if no collisions, False if collisions detected
            reason: String describing why infeasible (empty if feasible)
        """
        # Convert tensors to numpy if needed
        if isinstance(predicted_point_clouds, torch.Tensor):
            predicted_point_clouds = predicted_point_clouds.detach().cpu().numpy()
        
        if predicted_poses is not None and isinstance(predicted_poses, torch.Tensor):
            predicted_poses = predicted_poses.detach().cpu().numpy()
        
        # Calculate object centers from point clouds if poses not provided
        if predicted_poses is None:
            object_positions = np.mean(predicted_point_clouds, axis=1)[:, :3]  # [N, 3]
        else:
            object_positions = predicted_poses[:, :3]
        
        num_objects = object_positions.shape[0]
        
        if placement_height is not None and target_object_id is not None:
            # Two-phase check from Points2Plans:
            # 1. Check objects above placement target
            # 2. Check objects below placement target
            
            # Phase 1: Objects above target
            above_indices = []
            above_positions = []
            for i in range(num_objects):
                if i == target_object_id:
                    continue
                if object_positions[i, 2] > placement_height:
                    above_indices.append(i)
                    above_positions.append(object_positions[i])
            
            if len(above_positions) > 1:
                above_positions = np.array(above_positions)
                collision, pairs = self.check_scene_collisions(
                    above_positions,
                    exclude_indices=[]
                )
                if collision:
                    return False, f"Collision detected among objects above target (indices: {above_indices})"
            
            # Phase 2: Objects below target
            below_indices = []
            below_positions = []
            for i in range(num_objects):
                if i == target_object_id:
                    continue
                if object_positions[i, 2] < placement_height:
                    below_indices.append(i)
                    below_positions.append(object_positions[i])
            
            if len(below_positions) > 1:
                below_positions = np.array(below_positions)
                collision, pairs = self.check_scene_collisions(
                    below_positions,
                    exclude_indices=[]
                )
                if collision:
                    return False, f"Collision detected among objects below target (indices: {below_indices})"
        
        else:
            # Simple check: all objects at same level
            collision, pairs = self.check_scene_collisions(object_positions)
            if collision:
                return False, f"Collision detected between objects: {pairs}"
        
        return True, ""
    
    def visualize_bboxes(
        self,
        object_positions: np.ndarray,
        collision_pairs: Optional[List[Tuple[int, int]]] = None
    ):
        """
        Visualize bounding boxes (optional, for debugging).
        
        Args:
            object_positions: Object positions [N, 3]
            collision_pairs: List of colliding pairs to highlight
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            
            fig, ax = plt.subplots(figsize=(8, 8))
            
            num_objects = object_positions.shape[0]
            
            for i in range(num_objects):
                bbox = self.get_object_bbox(object_positions[i])
                bbox_array = np.array(bbox)
                
                x_min, y_min = bbox_array[:, 0].min(), bbox_array[:, 1].min()
                width = bbox_array[:, 0].max() - x_min
                height = bbox_array[:, 1].max() - y_min
                
                # Check if this object is in a collision
                in_collision = False
                if collision_pairs:
                    for pair in collision_pairs:
                        if i in pair:
                            in_collision = True
                            break
                
                color = 'red' if in_collision else 'blue'
                alpha = 0.3 if in_collision else 0.2
                
                rect = Rectangle((x_min, y_min), width, height,
                               linewidth=2, edgecolor=color, facecolor=color, alpha=alpha)
                ax.add_patch(rect)
                
                # Add object label
                ax.text(object_positions[i, 0], object_positions[i, 1], 
                       f'Obj {i}', ha='center', va='center', fontsize=10)
            
            ax.set_xlabel('X (meters)')
            ax.set_ylabel('Y (meters)')
            ax.set_title('Object Bounding Boxes (2D)')
            ax.axis('equal')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for visualization")


def test_collision_checker():
    """Test the collision checker with simple scenarios."""
    print("\n=== Testing Collision Checker ===\n")
    
    checker = CollisionChecker(x_collision=0.05, y_collision=0.05, verbose=True)
    
    # Test 1: Non-colliding objects
    print("Test 1: Non-colliding objects")
    positions = np.array([
        [0.0, 0.0, 0.8],
        [0.2, 0.0, 0.8],
        [0.4, 0.0, 0.8]
    ])
    collision, pairs = checker.check_scene_collisions(positions)
    print(f"Result: collision={collision}, pairs={pairs}")
    assert not collision, "Should not detect collision"
    print("✓ Passed\n")
    
    # Test 2: Colliding objects
    print("Test 2: Colliding objects (close together)")
    positions = np.array([
        [0.0, 0.0, 0.8],
        [0.03, 0.0, 0.8],  # Within collision threshold
        [0.4, 0.0, 0.8]
    ])
    collision, pairs = checker.check_scene_collisions(positions)
    print(f"Result: collision={collision}, pairs={pairs}")
    assert collision, "Should detect collision between objects 0 and 1"
    print("✓ Passed\n")
    
    # Test 3: Different heights (should not collide)
    print("Test 3: Objects at different heights")
    positions = np.array([
        [0.0, 0.0, 0.8],
        [0.0, 0.0, 1.0],  # Same XY but different Z
    ])
    collision, pairs = checker.check_scene_collisions(positions)
    print(f"Result: collision={collision}, pairs={pairs}")
    assert not collision, "Should not detect collision (different heights)"
    print("✓ Passed\n")
    
    print("=== All tests passed! ===\n")


if __name__ == "__main__":
    test_collision_checker()
