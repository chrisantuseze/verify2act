"""
Metadata Extraction Utilities for Robosuite Environments

This module handles extraction of static object properties from MuJoCo models.
"""

import numpy as np
import mujoco
from typing import Dict, List


class MetadataExtractor:
    """Extracts and manages object metadata from MuJoCo simulation."""
    
    def __init__(self, sim):
        """
        Initialize metadata extractor.
        
        Args:
            sim: MuJoCo simulation instance
        """
        self.sim = sim
        self.model = sim.model
    
    def extract_all_objects(self) -> Dict[str, Dict]:
        """
        Extract metadata for all relevant objects in the scene.
        
        Returns:
            Dictionary mapping object names to their metadata
        """
        object_metadata = {}
        object_bodies = self._get_object_bodies()
        
        for body_name, body_id in object_bodies.items():
            metadata = {
                'body_id': body_id,
                'body_name': body_name,
                'geom_ids': [],
                'extents': None,
                'fix_base_link': False,
                'object_type': 'primitive',
                'asset_filename': None,
            }
            
            # Get associated geoms
            geom_ids = self._get_body_geoms(body_id)
            metadata['geom_ids'] = geom_ids
            
            # Extract extent information
            if len(geom_ids) > 0:
                metadata['extents'] = self._compute_extents(geom_ids[0])
            
            # Determine if object is static
            metadata['fix_base_link'] = self._is_body_static(body_id)
            
            object_metadata[body_name] = metadata
        
        return object_metadata
    
    def _get_object_bodies(self) -> Dict[str, int]:
        """
        Identify relevant object bodies in the scene.
        
        Returns:
            Dictionary mapping body names to body IDs
        """
        object_bodies = {}
        
        # Keywords to identify relevant objects
        object_keywords = ['cube', 'table', 'block', 'object', 'can', 'milk', 'peg', 'box']
        
        # Keywords to exclude (robot parts and non-objects)
        exclude_keywords = ['robot', 'gripper', 'link', 'joint', 'world', 'floor',
                           'mount', 'controller', 'base', 'pedestal']
        
        for body_id in range(self.model.nbody):
            body_name = self.model.body(body_id).name
            
            if not body_name:
                continue
            
            # Skip excluded bodies
            if any(kw in body_name.lower() for kw in exclude_keywords):
                continue
            
            # Include object bodies
            if any(kw in body_name.lower() for kw in object_keywords):
                object_bodies[body_name] = body_id
        
        return object_bodies
    
    def _get_body_geoms(self, body_id: int) -> List[int]:
        """
        Get all geom IDs associated with a body.
        
        Args:
            body_id: MuJoCo body ID
            
        Returns:
            List of geom IDs
        """
        geom_ids = []
        for geom_id in range(self.model.ngeom):
            if self.model.geom_bodyid[geom_id] == body_id:
                geom_ids.append(geom_id)
        return geom_ids
    
    def _compute_extents(self, geom_id: int) -> List[float]:
        """
        Compute object extents from geometry.
        
        Args:
            geom_id: Primary geom ID
            
        Returns:
            [x, y, z] extents
        """
        geom_size = self.model.geom_size[geom_id]
        geom_type = self.model.geom_type[geom_id]
        
        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            return (geom_size * 2).tolist()
        elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            radius = geom_size[0]
            return [radius * 2, radius * 2, radius * 2]
        elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            radius = geom_size[0]
            half_height = geom_size[1]
            return [radius * 2, radius * 2, half_height * 2]
        else:
            return (geom_size * 2).tolist()
    
    def _is_body_static(self, body_id: int) -> bool:
        """
        Determine if a body is static (welded to world).
        
        Args:
            body_id: MuJoCo body ID
            
        Returns:
            True if static, False if movable
        """
        # Check if body has any joints
        has_joints = False
        for jnt_id in range(self.model.njnt):
            if self.model.jnt_bodyid[jnt_id] == body_id:
                has_joints = True
                break
        
        if not has_joints:
            return True
        
        # Check if body has very large mass (kinematic)
        body_mass = self.model.body_mass[body_id]
        if body_mass > 1000:
            return True
        
        return False
