from dataclasses import dataclass
from typing import List
import synreal_sim as sim
from _deformable_data_helper import *

@dataclass
class s3d_scene:
    world: sim.World
    deformable_bodies: List[ sim.DeformableBody]


class deformable_body_builder:
    def __init__(self, file:str, pos, rest_pos, tets, collision_faces):
        self.file = file
        self.pos = pos
        self.rest_pos = rest_pos
        self.tets = tets
        self.collision_faces = collision_faces







class s3d_scene_builder:
    def __init__(self ):
        self.deformable_bodies : List[deformable_body_builder] = []


    def add_deformable_body_by_file(self, filename, collision_faces):
        self.deformable_bodies.append(deformable_body_builder(filename,[],[],[],[]))

    def add_deformable_body_by_file_with_boundary_collision_faces(self, filename ):
        self.deformable_bodies.append(deformable_body_builder(filename,[],[],[],[]))

    def add_deformable_body(self, pos, rest_pos,tets, collision_faces):
        self.deformable_bodies.append(deformable_body_builder('', pos, rest_pos, tets, collision_faces))

    def build(self,sim_world): #TODO: remove sim_world later
        scene = s3d_scene()
        scene.world = sim_world

        for dfm in self.deformable_bodies:
            if dfm.file =='':
                obj = sim.DeformableBody(dfm.pos , dfm.collision_faces, dfm.tets, dfm.rest_pos)
            elif len(dfm.collision_faces) == 0:
                pos, tets = load_tetrahedrons(dfm.file)
                collision_faces = compute_boundary_faces(tets)  # you may want to specify collision faces another way
                obj = sim.DeformableBody(pos , collision_faces, tets, pos)
            else:
                pos, tets = load_tetrahedrons(dfm.file)
                obj = sim.DeformableBody(pos , dfm.collision_faces, tets, pos)

            scene.deformable_bodies.append(obj)
            obj.attach(scene.world)

        return scene