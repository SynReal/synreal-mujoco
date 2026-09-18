
import mujoco.viewer
import numpy as np
import synreal_sim as sim

from pathlib import Path
import sys

if __name__ == '__main__' and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synreal_mujoco import project_io
from synreal_mujoco import project_data_classes as pdc

import synreal_mujoco.s3d_mj as s3d_mj
import synreal_mujoco.s3d_scene_builder as s3d_scene_builder
import synreal_mujoco.s3d_scene_stepper as s3d_scene_stepper


class project_runner:
    def __init__(self):
        pass

    def run(self,project_path:Path):
        l_project_io = project_io.project_io()
        l_project_io.read(project_path)
        l_project_io.save(project_path/'auto_save.json')
        l_project_data = l_project_io.get_data()
        s3d_scene_builder = self._config_project(l_project_data, project_path)
        self._start_simulation_loop(s3d_scene_builder)
        print("done without nothing!")

    def _config_project(self, p_project_data : project_io.dc.project_data, project_path: Path = Path('.')):
        self._login_sim()

        scene_builder = s3d_scene_builder.s3d_scene_builder()
        for entity in p_project_data.entities:
            self._dispatch_entity(entity, scene_builder, project_path)
        return scene_builder

    def _dispatch_entity(self, entity, s3d_scene_builder, project_path: Path = Path('.')):
        if isinstance(entity, pdc.mjcf_rigidbody):
            s3d_scene_builder.add_mjcf_rigidbodies((project_path / entity.path).resolve())
        elif isinstance(entity, pdc.deformable_body): 
            dfm_attrib = s3d_scene_builder.add_deformable_body_by_file((project_path / entity.path).resolve())
            dfm_attrib.attrib.density = entity.attrib.density
            dfm_attrib.attrib.dynamicFriction = entity.attrib.dynamic_friction
            dfm_attrib.attrib.poissonRatio = entity.attrib.poisson_ratio
            dfm_attrib.attrib.staticFriction = entity.attrib.static_friction
            dfm_attrib.attrib.surfaceOffsets = entity.attrib.surface_offsets
            dfm_attrib.attrib.youngsModulus = entity.attrib.youngs_modulus

            dfm_attrib.get_pos = lambda  x: x + entity.trans.translation # alter current pos
        elif isinstance(entity, pdc.cloth): 
            cloth_builder = s3d_scene_builder.add_cloth_by_file((project_path / entity.path).resolve())
            cloth_builder.translate = np.array(entity.trans.translation, dtype=float, copy=True)
            cloth_builder.quat = np.empty(4)
            # Fixed-axis XYZ Euler angles in radians; MuJoCo uses wxyz quaternions.
            mujoco.mju_euler2Quat(cloth_builder.quat, np.asarray(entity.trans.euler_xyz, dtype=float), 'XYZ')

            attrib = entity.attrib
            cloth_builder.attrib = sim.ClothAttrib()
            cloth_builder.attrib.stretch_stiff = sim.Vec3f(*attrib.stretch_stiffness)
            cloth_builder.attrib.bend_stiff = sim.Vec3f(*attrib.bend_stiffness)
            cloth_builder.attrib.thickness = attrib.thickness
            cloth_builder.attrib.density = attrib.density
            cloth_builder.attrib.pressure = attrib.pressure
            cloth_builder.attrib.static_friction = attrib.static_friction
            cloth_builder.attrib.dynamic_friction = attrib.dynamic_friction
            cloth_builder.attrib.yield_curvature = attrib.yield_curvature
            cloth_builder.attrib.volume_conserve_strength = attrib.volume_conserve_strength
            cloth_builder.attrib.frozen = attrib.frozen



    def _login_sim(self):
        curr_folder = Path(__file__).parent
        login_file = curr_folder.parent/ 'simulation_login.json'
        s3d_mj.log_in_simulation(login_file=login_file) # this line is optional, but a login prompt will pop up latter

    def _start_simulation_loop(self,s3d_scene_builder):
        m, d, s = s3d_scene_builder.build()

        l_s3d_scene_stepper = s3d_scene_stepper.s3d_scene_stepper(m,d,s)

        with mujoco.viewer.launch_passive(m, d) as viewer:

            while viewer.is_running():

                mujoco.mj_step(m, d)

                l_s3d_scene_stepper.set_rigidbody_pos_mj_2_s3d()
                l_s3d_scene_stepper.step_s3d()
                l_s3d_scene_stepper.set_cloth_pos_s3d_2_mj()

                viewer.sync()
        


if __name__ == '__main__':
    l_project_runner = project_runner()
    cwd = Path(__file__).parent.resolve()
    #project_path = cwd / 'projects' / 'deformable_body'
    project_path = cwd / 'projects' / 'piper_cloth'
    l_project_runner.run(project_path)
