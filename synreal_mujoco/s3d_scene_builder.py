from copy import deepcopy
from dataclasses import dataclass
from typing import List
from typing import Callable
from typing import Dict

import numpy as np
import synreal_sim as sim
import synreal_mujoco._mj_data_helper as _mj_data_helper
from synreal_mujoco._deformable_data_helper import *
import synreal_mujoco.s3d_mj as s3d_mj
import synreal_mujoco.smj as smj
from synreal_mujoco import cloth_property
import synreal_mujoco.data_classes as dc
import synreal_mujoco.utility as utility
from synreal_mujoco.sim_rigidbody import sim_rigidbody

import xml.etree.ElementTree as ET
from pathlib import Path
import os
import json
import mujoco


xml_prefix_cloth = 'cloth'
xml_prefix_deformable_body = 'dfm'
xml_prefix_rigid_mesh = 'rigid_mesh'



def _name_2_xml_name(prefix, obj_file):
    file_base_name = str(Path(obj_file).stem)
    return prefix +'_' + file_base_name

def _xml_name_2_name(prefix, obj_file):
    file_base_name = str(Path(obj_file).stem)
    return file_base_name.removeprefix(prefix+'_')

class s3d_scene_builder:
    def __init__(self, world_attrib_setter = lambda x:x  ):

        # world
        self.world_attrib_setter = world_attrib_setter

        # deformable body
        self.deformable_body_files : List[str] = []
        self.deformable_body_buidlers : List[dc.deformable_body_builder] = []
        self._temp_files: List[str] = []

        #mjcf_scene
        self.mjcf_file =''
        self.flexed_mjcf_file = ''
        self.rigidbody_builder_fn : Callable[[str],dc.rigid_body_builder]

        #rigid mesh
        self.rigid_mesh_files = []
        self.rigid_mesh_builder_map: Dict[str, dc.rigid_mesh_builder] = {}

        #cloth
        self.cloth_files = []
        self.cloth_builder_map : Dict[str, dc.cloth_builder] = {}
        self.cloth_uv = {}
        self.use_uv = []

        # connects
        self.connect_files = []


    # mujoco mjcf
    def add_mjcf_rigidbodies( self, filename, rigidbody_builder_fn = lambda name,attrib:attrib ):
        self.mjcf_file = filename
        self.rigidbody_builder_fn = rigidbody_builder_fn

    # clothes
    def add_cloth_by_file(self, filename, use_uv = False):
        self.cloth_files.append(filename)
        builder = dc.cloth_builder()
        name = s3d_scene_builder._get_file_name(filename)
        self.cloth_builder_map[name] = builder
        self.use_uv.append(use_uv)
        return builder


    def add_rigid_mesh(self,filename):
        filename = Path(filename).resolve()
        name = self._get_file_name(filename)
        if name in self.rigid_mesh_builder_map:
            raise ValueError(f'Duplicate rigid mesh name: {name}')
        builder = dc.rigid_mesh_builder()
        self.rigid_mesh_files.append(filename)
        self.rigid_mesh_builder_map[name] = builder
        return builder

    # deformable body
    def add_deformable_body_by_file(self, filename ):
        self.deformable_body_files.append(filename)
        dfm_builder = dc.deformable_body_builder()
        dfm_builder.get_pos = lambda x : x
        dfm_builder.get_rest_pos = lambda x : x
        self.deformable_body_buidlers.append(dfm_builder)
        return dfm_builder

    def add_connect(self, filename):
        self.connect_files.append(filename)


    # build
    def build(self ):

        scene = dc.s3d_scene()

        dfm_bodies_param = self._add_flex_to_mjcf(scene)

        self._compute_cloth_uv()

        m, d = s3d_mj.load_data(self.flexed_mjcf_file)

        for path in self._temp_files:
            os.remove(path)
        self._temp_files.clear()

        def configure_world(attrib):
            if self.rigid_mesh_files:
                attrib.enable_rigid_self_collision = True
            self.world_attrib_setter(attrib)

        scene.world = s3d_mj.get_a_sim_world(m, configure_world)

        s3d_scene_builder._add_rigid_body_to_scene(scene, m, d, self.rigidbody_builder_fn )
        self._add_rigid_meshes_to_scene(scene, m, d)

        def get_cloth_uv(xml_name):
            name = _xml_name_2_name(xml_prefix_cloth,xml_name)
            return self.cloth_uv[name]

        s3d_scene_builder._add_cloth_to_scene(scene, m, d, self.cloth_builder_map, get_cloth_uv, xml_prefix_cloth)

        self._add_deformable_body_to_scene(scene, dfm_bodies_param)

        self._add_connects_to_scene(scene, m)

        collision_force = []
        scene.mapper = smj.s3d_mj_mapper (
            scene.world,
            scene.sim_cloth,
            scene.cloth_names,
            scene.rigid_bodies,
            scene.mj_rb_index,
            collision_force
        )

        return m, d, scene



    @staticmethod
    def _add_rigid_body_to_scene(s : dc.s3d_scene, m, d, rigidbody_builder_fn : Callable[[str,dc.rigid_body_builder], None ]):
        s.rigid_bodies, s.mj_rb_index, s.mj_geom_index, s.mj_mesh_index, s.rigid_body_names = s3d_mj._add_rigid_body_to_sim(m, d, s.world, rigidbody_builder_fn)


    @staticmethod
    def _get_file_name(obj_file):
        return str(Path(obj_file).stem)

    def _add_rigid_meshes_to_scene(self, scene, m, d):
        for filename in self.rigid_mesh_files:
            name = self._get_file_name(filename)
            builder = self.rigid_mesh_builder_map[name]
            flex_name = _name_2_xml_name(xml_prefix_rigid_mesh, name)
            flex_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_FLEX, flex_name)
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, builder.quat)
            positions = _mj_data_helper.get_local_coodinate(
                _mj_data_helper._get_flex_pos(flex_id, m, d),
                rotation.reshape(3, 3), builder.translate,
            )
            mesh = sim.Mesh(_mj_data_helper._get_flex_tri(flex_id, m), positions)
            transform = _mj_data_helper.to_sim_transfrom(rotation, builder.translate)
            body = sim_rigidbody(mesh, transform)
            body.set_attrib(builder.attrib)
            body.set_pin(builder.is_fixed)
            body.set_collision_group(1)
            body.set_collision_mask(1)
            body.attach(scene.world)
            scene.rigid_meshes.append(body)
            scene.rigid_mesh_flex_names.append(flex_name)


    @staticmethod
    def _add_cloth_to_scene(s : dc.s3d_scene, m, d , attrib_map, get_cloth_uv, name_start_with_will_considered_cloth):

        def __get_attrib (name ):
           return attrib_map[_xml_name_2_name(xml_prefix_cloth, name)].attrib

        s.sim_cloth, s.cloth_names = s3d_mj._add_cloth_to_sim_2( m, d, s.world,  __get_attrib , get_cloth_uv, name_start_with_will_considered_cloth )


    def _add_deformable_body_to_scene(self, scene : dc.s3d_scene, dfm_body_params):
        scene.deformable_bodies = []
        for dfm_body_param in dfm_body_params:
            dfm = sim.DeformableBody(dfm_body_param.pos , dfm_body_param.collision_faces, dfm_body_param.tets, dfm_body_param.rest_pos)
            dfm.set_attrib(dfm_body_param.attrib)
            scene.deformable_bodies.append(dfm)
            scene.used_vert_of_deformable_body_collision_faces.append(dfm_body_param.used_vert_of_deformable_body_collision_faces)
            scene.deformable_body_collision_faces.append(dfm_body_param.collision_faces)
            dfm.attach(scene.world)

    def _set_connect_dfm_fixed(self, connect_info, scene : dc.s3d_scene):
        dfm_name = connect_info.object1
        fixed_verts = connect_info.data1
        dfm_body = scene.deformable_bodies[scene.deformable_body_names.index(dfm_name)]
        flags = np.array([True for _ in range(len(fixed_verts))])
        dfm_body.set_pin(flags, fixed_verts)

    def _compute_connect_dfm_local_coord(self, connect_info, scene : dc.s3d_scene, m):
        rb_name = connect_info.object0
        dfm_name = connect_info.object1

        dfm_body = scene.deformable_bodies[scene.deformable_body_names.index(dfm_name)]
        dfm_x = dfm_body.get_positions()

        mesh_id = scene.mj_mesh_index[scene.rigid_body_names.index(rb_name)]
        mesh_quat = m.mesh_quat[mesh_id]
        mesh_center = m.mesh_pos[mesh_id]
        mesh_rot = np.empty(9)
        mujoco.mju_quat2Mat(mesh_rot, mesh_quat)
        mesh_rot = mesh_rot.reshape(3, 3)
        connect_info.data0 = _mj_data_helper.get_local_coodinate(dfm_x, mesh_rot, mesh_center)

    @staticmethod
    def _populate_connect_info(connect_info,object0,object1):
        connect_info.object0 = object0['name']
        connect_info.object1 = object1['name']
        connect_info.object_type0 = object0['object_type']
        connect_info.object_type1 = object1['object_type']
        connect_info.data_type0 = object0['data_type']
        connect_info.data_type1 = object1['data_type']
        connect_info.data0 = object0['data']
        connect_info.data1 = object1['data']

    def _add_connects_to_scene(self, scene : dc.s3d_scene, m):
        scene.connect_infos = []
        for connect_file in self.connect_files:
            with open(connect_file, 'r') as f:
                data = json.load(f)

            object0 = data['object0']
            object1 = data['object1']

            connect_info = dc.connect_info()
            s3d_scene_builder._populate_connect_info(connect_info,object0,object1)
            scene.connect_infos.append(connect_info)

            if connect_info.is_deformable_body_attatch_to_rigid_body():
                self._set_connect_dfm_fixed(connect_info, scene)
                self._compute_connect_dfm_local_coord(connect_info,scene,m)


    @staticmethod
    def _add_flexcomp_to_worldbody(tree: ET.ElementTree, name_prefix:str, name:str, file: str, pos,quat,rgba, **attribs) -> None:
        """Inserts a <flexcomp> with the given file into <worldbody>. Extra keyword
        arguments are added as XML attributes (e.g. name, type, pos, radius, dim)."""
        worldbody = tree.getroot().find('worldbody')
        if worldbody is None:
            raise ValueError("No <worldbody> element found in the XML tree")

        attrs = {
            'name': _name_2_xml_name(name_prefix, name),
            'type': 'mesh',
            'pos': f'{pos[0]} {pos[1]} {pos[2]}',
            'quat': f'{quat[0]} {quat[1]} {quat[2]} {quat[3]}',
            'radius': '0.0005',
            'dim': '2',
            'rgba': f'{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}',
            'custom': 'true',
            'file': str(file)
        }
        attrs.update(attribs)
        elem = ET.SubElement(worldbody, 'flexcomp', attrs)
        elem.text = '\n        '  # forces explicit </flexcomp> closing tag instead of />
        elem.tail = '\n\n    '    # newline between </flexcomp> and </worldbody>

    def _add_flex_cloth(self, tree):
        for cloth_file in self.cloth_files:
            name = s3d_scene_builder._get_file_name(cloth_file)
            cloth_builder = self.cloth_builder_map[name]
            s3d_scene_builder._add_flexcomp_to_worldbody(tree, xml_prefix_cloth, name, cloth_file,cloth_builder.translate,cloth_builder.quat,cloth_builder.rgba)

    def _add_flex_rigid_meshes(self, tree):
        for filename in self.rigid_mesh_files:
            name = self._get_file_name(filename)
            builder = self.rigid_mesh_builder_map[name]
            self._add_flexcomp_to_worldbody(
                tree, xml_prefix_rigid_mesh, name, filename,
                builder.translate, builder.quat, builder.rgba,
            )

    def _add_flex_deformable_body(self, tree, mjcf_name, s : dc.s3d_scene):
        deformable_bodies_param=[]
        s.deformable_body_names =[]
        for i, dfm_file in enumerate(self.deformable_body_files):
            dfm_builder = self.deformable_body_buidlers[i]
            pos, tets = load_tetrahedrons(dfm_file)
            if dfm_builder.collision_faces is None:
                faces, used_vert_of_deformable_body_collision_faces = compute_boundary_faces(tets)

            rest_pos = deepcopy(dfm_builder.get_pos(pos))
            curr_pos = deepcopy(dfm_builder.get_pos(pos))

            temp_obj_path = mjcf_name + f'_{xml_prefix_deformable_body}_{i}.obj'
            temp_obj_path = Path(temp_obj_path).as_posix()  
            utility.write_obj(curr_pos[used_vert_of_deformable_body_collision_faces], faces, temp_obj_path)  # export before offset mutates pos
            self._temp_files.append(temp_obj_path)

            name = s3d_scene_builder._get_file_name(dfm_file)
            s3d_scene_builder._add_flexcomp_to_worldbody(
                tree, xml_prefix_deformable_body, name, temp_obj_path, np.array([0,0,0]), np.array([1,0,0,0]),np.array([1,0.6,0.8,1]))

            s.deformable_body_names.append(name)
            deformable_bodies_param.append(dc.deformable_body_constructor_param(curr_pos, rest_pos, tets, faces, used_vert_of_deformable_body_collision_faces,dfm_builder.attrib))
        return deformable_bodies_param


    def _add_flex_to_mjcf(self, s: dc.s3d_scene):
        tree = ET.parse(self.mjcf_file)
        base, ext = os.path.splitext(self.mjcf_file)

        # cloth
        self._add_flex_cloth(tree)

        # rigid mesh
        self._add_flex_rigid_meshes(tree)

        # deformable body
        deformable_bodies_param = self._add_flex_deformable_body(tree,base,s)

        # write .xml
        out_path = base + '_flex' + ext
        tree.write(out_path)
        self._temp_files.append(out_path)
        self.flexed_mjcf_file = out_path

        return deformable_bodies_param


    def _compute_cloth_uv(self):
        for cloth_file, use_uv in zip(self.cloth_files, self.use_uv):
            name = s3d_scene_builder._get_file_name(cloth_file)
            if  not use_uv:
                self.cloth_uv[name] = None
                continue

            uv = []
            if str(cloth_file.suffix) == '.obj':
                uv = utility.read_uv_from_obj(cloth_file)
            if len(uv) == 0:
                self.cloth_uv[name] = None
            else:
                self.cloth_uv[name] = uv

