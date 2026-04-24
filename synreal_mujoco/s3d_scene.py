from copy import deepcopy
from dataclasses import dataclass
from typing import List

import numpy as np
import synreal_sim as sim
from synreal_mujoco._deformable_data_helper import *
import synreal_mujoco.s3d_mj as s3d_mj
from synreal_mujoco import cloth_property

import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class s3d_scene:
    world: sim.World = None
    deformable_bodies: List[ sim.DeformableBody] = field(default_factory=list)
    deformable_body_names : List[str] = field(default_factory=list)

    rigid_bodies: List[ sim.RigidBody] = field(default_factory=list)

    sim_cloth: List[sim.Cloth] = field(default_factory=list)
    cloth_names: List[str] = field(default_factory=list)



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
        self.deformable_bodies_ready : List[deformable_body_builder] = []
        self.mjcf_file =''
        self.cloth_files = []
        self.cloth_name_prefix = 'cloth'
        self.deformable_body_name_prefix = 'dfm'
        self._temp_files: List[str] = []

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
    def _add_cloth_to_scene(s : s3d_scene,m,d , name_start_with_will_considered_cloth):
        s.sim_cloth, s.cloth_names = s3d_mj.add_cloth_to_sim(m, d, s.world, name_start_with_will_considered_cloth, lambda nama, attrib: cloth_property.set_cloth_property_default( attrib))


    def _add_deformable_body_to_scene(self, scene : s3d_scene):
        scene.deformable_bodies = []
        for dfm in self.deformable_bodies_ready:
            obj = sim.DeformableBody(dfm.pos , dfm.collision_faces, dfm.tets, dfm.rest_pos)
            scene.deformable_bodies.append(obj)
            obj.attach(scene.world)


    @staticmethod
    def _export_surface_to_obj(pos, faces, obj_path: str) -> None:
        with open(obj_path, 'w') as f:
            for v in pos:
                f.write(f'v {v[0]} {v[1]} {v[2]}\n')
            for face in faces:
                f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')

    @staticmethod
    def add_flexcomp_to_worldbody(tree: ET.ElementTree, name:str, file: str, pos,quat, **attribs) -> None:
        """Inserts a <flexcomp> with the given file into <worldbody>. Extra keyword
        arguments are added as XML attributes (e.g. name, type, pos, radius, dim)."""
        worldbody = tree.getroot().find('worldbody')
        if worldbody is None:
            raise ValueError("No <worldbody> element found in the XML tree")

        attrs = {
            'name': name,
            'type': 'mesh',
            'pos': f'{pos[0]} {pos[1]} {pos[2]}',
            'quat': f'{quat[0]} {quat[1]} {quat[2]} {quat[3]}',
            'radius': '0.0005',
            'dim': '2',
            'custom': 'true',
            'file': file
        }
        attrs.update(attribs)
        elem = ET.SubElement(worldbody, 'flexcomp', attrs)
        elem.text = '\n        '  # forces explicit </flexcomp> closing tag instead of />
        elem.tail = '\n\n    '    # newline between </flexcomp> and </worldbody>

    def _add_flex_cloth(self,s:s3d_scene):
        import os
        tree = ET.parse(self.mjcf_file)

        base, ext = os.path.splitext(self.mjcf_file)

        # cloth
        for i,cloth_file in enumerate(self.cloth_files):
            name = self.cloth_name_prefix +'_' + str(i)
            s3d_scene_builder.add_flexcomp_to_worldbody(tree, self.cloth_name_prefix, cloth_file,[-0.8,-2.0,0.2],[1,0,0,0])

        # deformable body
        s.deformable_body_names =[]
        offset = np.array([0, 0, 0.5])
        rot_quat = np.array([1, 0, 0, 0])
        for i, dfm in enumerate(self.deformable_bodies):
            dfm_b = dfm

            if dfm.file == '':
                pos, faces = dfm.pos, dfm.collision_faces

            elif len(dfm.collision_faces) == 0:
                pos, tets = load_tetrahedrons(dfm.file)
                faces = compute_boundary_faces(tets)

                dfm_b.pos = pos
                dfm_b.collision_faces = faces
                dfm_b.rest_pos = deepcopy(pos)
                dfm_b.tets = tets

            else:
                pos, _ = load_tetrahedrons(dfm.file)
                faces = dfm.collision_faces

                dfm_b.collision_faces = faces

            obj_path = base + f'_{self.deformable_body_name_prefix}_{i}.obj'
            s3d_scene_builder._export_surface_to_obj(pos, faces, obj_path)  # export before offset mutates pos
            self._temp_files.append(obj_path)

            dfm_b.pos += offset
            dfm_b.rest_pos += offset

            self.deformable_bodies_ready.append(dfm_b)

            name = self.deformable_body_name_prefix +'_' + str(i)
            s3d_scene_builder.add_flexcomp_to_worldbody(
                tree, name, os.path.basename(obj_path), offset, rot_quat)
            s.deformable_body_names.append(name)

        # write .xml
        out_path = base + '_flex' + ext
        tree.write(out_path)
        self._temp_files.append(out_path)
        self.mjcf_file = out_path


    # build
    def build(self ):

        scene = s3d_scene()

        self._add_flex_cloth(scene)

        m, d = s3d_mj.load_data(self.mjcf_file)

        import os
        for path in self._temp_files:
            os.remove(path)
        self._temp_files.clear()

        scene.world = s3d_mj.get_a_sim_world(m)

        s3d_scene_builder._add_rigid_body_to_scene(scene, m, d)

        name_start_with_will_considered_cloth='cloth'
        s3d_scene_builder._add_cloth_to_scene(scene, m, d, name_start_with_will_considered_cloth)

        self._add_deformable_body_to_scene(scene)

        return m,d,scene