from dataclasses import dataclass
from typing import List
import synreal_sim as sim
from synreal_mujoco._deformable_data_helper import *
import synreal_mujoco.s3d_mj as s3d_mj
from synreal_mujoco import cloth_property

@dataclass
class s3d_scene:
    world: sim.World = None
    deformable_bodies: List[ sim.DeformableBody] = None

    rigid_bodies: List[ sim.RigidBody] = None
    sim_cloth: List[sim.Cloth] = None
    cloth_names: List[str] = None



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
        self.mjcf_file =''
        self.cloth_files = []

    # mujoco mjcf
    def add_mjcf_rigidbodies(self, filename ):
        self.mjcf_file = filename

    # clothes
    def add_cloth_by_file(self, filename ):
        self.cloth_files.append(filename)

    # deformable body
    def add_deformable_body_by_file(self, filename, collision_faces):
        self.deformable_bodies.append(deformable_body_builder(filename,[],[],[],[]))

    def add_deformable_body_by_file_with_boundary_collision_faces(self, filename ):
        self.deformable_bodies.append(deformable_body_builder(filename,[],[],[],[]))

    def add_deformable_body(self, pos, rest_pos,tets, collision_faces):
        self.deformable_bodies.append(deformable_body_builder('', pos, rest_pos, tets, collision_faces))

    @staticmethod
    def _add_rigid_body_to_scene(s : s3d_scene,m,d):
        s.rigid_bodies = s3d_mj.add_rigid_body_to_sim(m, d, s.world, lambda name,attrib: cloth_property.set_rigid_body_property_default(attrib),False)


    @staticmethod
    def _add_cloth_to_scene(s : s3d_scene,m,d):
        s.sim_cloth, s.cloth_names = s3d_mj.add_cloth_to_sim(m, d, s.world, lambda nama, attrib: cloth_property.set_cloth_property_default( attrib))


    def _add_deformable_body_to_scene(self, scene : s3d_scene):
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

    # build
    def build(self ):

        scene = s3d_scene()

        m, d = s3d_mj.load_data(self.mjcf_file)

        scene.world = s3d_mj.get_a_sim_world(m)

        s3d_scene_builder._add_rigid_body_to_scene(scene, m,d)
        s3d_scene_builder._add_cloth_to_scene(scene, m,d)
        self._add_deformable_body_to_scene(scene)

        return m,d,scene