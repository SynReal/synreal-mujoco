
from dataclasses import dataclass, field
from typing import List
from typing import Callable

import numpy as np
import synreal_sim as sim
from synreal_mujoco import cloth_property

@dataclass
class s3d_scene:
    world: sim.World = None
    deformable_bodies: List[ sim.DeformableBody] = field(default_factory=list)
    deformable_body_names : List[str] = field(default_factory=list)

    rigid_bodies: List[ sim.RigidBody] = field(default_factory=list)

    sim_cloth: List[sim.Cloth] = field(default_factory=list)
    cloth_names: List[str] = field(default_factory=list)


@dataclass
class rigid_body_builder:
    with_convex_hull : bool = False
    is_fixed: bool = True
    attrib: sim.RigidBodyAttrib = cloth_property.get_rigid_body_property_default()


@dataclass
class cloth_builder:
    translate = np.array([0,0,0])
    quat = np.array([1,0,0, 0])
    attrib: sim.ClothAttrib = cloth_property.get_cloth_property_default()

class deformable_body_builder:
    def __init__(self, file:str, pos, rest_pos, tets, collision_faces):
        self.file = file
        self.pos = pos
        self.rest_pos = rest_pos
        self.tets = tets
        self.collision_faces = collision_faces
        self.attrib = None
