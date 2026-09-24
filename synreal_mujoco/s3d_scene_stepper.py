from dataclasses import dataclass

import numpy as np
import synreal_sim as sim

import synreal_mujoco._mj_data_helper as _mj_data_helper
import synreal_mujoco.s3d_scene_builder as s3d_scene_builder
import synreal_mujoco.smj as smj


@dataclass(frozen=True)
class _DeformableRigidAttachment:
    deformable_body: object
    rigid_geom_id: int
    local_coords: np.ndarray
    vertex_indices: np.ndarray


class s3d_scene_stepper:
    def __init__(self, mujoco_model, mujoco_data, scene):
        self.mujoco_model = mujoco_model
        self.mujoco_data = mujoco_data
        self.scene: s3d_scene_builder.s3d_scene = scene

        self._cache_scene_lookups()
        self._attachments = self._build_deformable_rigid_attachments()
        self._last_rigid_body_transforms = [
            rigid_body.get_transform() for rigid_body in scene.rigid_bodies
        ]

    def set_mocap_pos(self, mocap_name, pos):
        smj.set_mocap_pos(self.mujoco_model, self.mujoco_data, mocap_name, pos)

    def update_force_2_mujoco(self):
        smj.update_rigidbody_cloth_collision_force(
            self.mujoco_model,
            self.mujoco_data,
            self.scene.mapper,
        )
        smj.apply_collision_force_to_rigidbody(
            self.mujoco_model,
            self.mujoco_data,
            self.scene.mapper,
        )

    def set_rigidbody_pos_mj_2_s3d(self):
        geom_xmat = self.mujoco_data.geom_xmat
        geom_xpos = self.mujoco_data.geom_xpos

        for index, rigid_body, geom_id in self._iter_rigid_body_geoms():
            current_transform = _mj_data_helper.to_sim_transfrom(
                geom_xmat[geom_id],
                geom_xpos[geom_id],
            )
            rigid_body.move(self._last_rigid_body_transforms[index], current_transform)
            self._last_rigid_body_transforms[index] = current_transform

        self._update_attached_deformable_bodies()

    def step_s3d(self):
        self.scene.world.step_sim()

    def set_cloth_pos_s3d_2_mj(self, include_stress=True):
        """Fetch SynReal outputs and update all render-only flex meshes."""
        self._fetch_s3d_outputs(include_stress)
        self._copy_cloth_positions_to_mujoco()
        self._copy_deformable_body_positions_to_mujoco()
        self.set_rigidbody_pos_s3d_2_mj()

    def set_rigidbody_pos_s3d_2_mj(self):
        """Copy rigid mesh poses after fetching SynReal's Transforms output."""
        for body, flex_name in zip(self.scene.rigid_meshes, self.scene.rigid_mesh_flex_names):
            _mj_data_helper.set_flex_positions(
                self.mujoco_model, self.mujoco_data, flex_name, body.get_positions(),
            )

    #################  deformable body stress related methods
    def get_deformable_body_positions_in_rigidbody_frame(self, dfm_name, rigidbody_name):
        positions = self._deformable_body_by_name[dfm_name].get_positions()
        geom_id = self._rigid_body_geom_by_name[rigidbody_name]

        if geom_id is None:
            return positions

        return _mj_data_helper.get_local_coodinate(
            positions,
            self.mujoco_data.geom_xmat[geom_id].reshape(3, 3),
            self.mujoco_data.geom_xpos[geom_id],
        )

    def get_deformable_body_stress(self, dfm_name):
        return self._deformable_body_by_name[dfm_name].get_stress_map()

    def reset_deformable_body_to_connected_pos(self, rigid_body_name, dfm_name):
        deformable_body = self._deformable_body_by_name[dfm_name]

        for connect_info in self.scene.connect_infos:
            if connect_info.object0 != rigid_body_name or connect_info.object1 != dfm_name:
                continue

            geom_id = self._rigid_body_geom_by_name[rigid_body_name]
            world_pos = self._rigid_local_to_world(connect_info.data0, geom_id)
            deformable_body.set_positions(world_pos, np.arange(len(world_pos)))

    #################  end deformable body stress related methods 

    def _cache_scene_lookups(self):
        scene = self.scene

        self._deformable_body_by_name = dict(
            zip(scene.deformable_body_names, scene.deformable_bodies)
        )
        self._deformable_body_used_verts_by_name = dict(
            zip(
                scene.deformable_body_names,
                scene.used_vert_of_deformable_body_collision_faces,
            )
        )
        self._rigid_body_geom_by_name = dict(zip(scene.rigid_body_names, scene.mj_geom_index))
        self._deformable_body_flex_name_by_name = {
            name: self._deformable_body_xml_name(name)
            for name in scene.deformable_body_names
        }

    def _build_deformable_rigid_attachments(self):
        attachments = []

        for connect_info in self.scene.connect_infos:
            if not connect_info.is_deformable_body_attatch_to_rigid_body():
                continue

            attachments.append(
                _DeformableRigidAttachment(
                    deformable_body=self._deformable_body_by_name[connect_info.object1],
                    rigid_geom_id=self._rigid_body_geom_by_name[connect_info.object0],
                    local_coords=np.asarray(connect_info.data0),
                    vertex_indices=np.asarray(connect_info.data1),
                )
            )

        return attachments

    def _iter_rigid_body_geoms(self):
        for index, (rigid_body, geom_id) in enumerate(
            zip(self.scene.rigid_bodies, self.scene.mj_geom_index)
        ):
            if geom_id is not None:
                yield index, rigid_body, geom_id

    def _fetch_s3d_outputs(self, include_stress):
        output_vars = [
            sim.OutputVars.Positions,
            sim.OutputVars.Transforms,
            sim.OutputVars.CollisionForceRigidBoydCloth,
        ]
        if include_stress:
            output_vars.extend(
                [
                    sim.OutputVars.AvatarStressMapObstacle,
                    sim.OutputVars.AvatarStressMapCloth,
                ]
            )

        self.scene.world.fetch_sim(0, output_vars)

    def _copy_cloth_positions_to_mujoco(self):
        for cloth, cloth_name in zip(self.scene.sim_cloth, self.scene.cloth_names):
            _mj_data_helper.set_flex_positions(
                self.mujoco_model,
                self.mujoco_data,
                cloth_name,
                cloth.get_positions(),
            )

    def _copy_deformable_body_positions_to_mujoco(self):
        for dfm_name, deformable_body in zip(
            self.scene.deformable_body_names,
            self.scene.deformable_bodies,
        ):
            used_verts = self._deformable_body_used_verts_by_name[dfm_name]
            flex_name = self._deformable_body_flex_name_by_name[dfm_name]
            positions = deformable_body.get_positions()[used_verts]

            _mj_data_helper.set_flex_positions(
                self.mujoco_model,
                self.mujoco_data,
                flex_name,
                positions,
            )

    def _update_attached_deformable_bodies(self):
        for attachment in self._attachments:
            world_pos = self._rigid_local_to_world(
                attachment.local_coords,
                attachment.rigid_geom_id,
            )
            attachment.deformable_body.set_positions(
                world_pos[attachment.vertex_indices],
                attachment.vertex_indices,
            )

    def _rigid_local_to_world(self, local_coords, geom_id):
        return _mj_data_helper.get_world_coodinate(
            local_coords,
            self.mujoco_data.geom_xmat[geom_id].reshape(3, 3),
            self.mujoco_data.geom_xpos[geom_id],
        )

    @staticmethod
    def _deformable_body_xml_name(dfm_name):
        return s3d_scene_builder._name_2_xml_name(
            s3d_scene_builder.xml_prefix_deformable_body,
            dfm_name,
        )
