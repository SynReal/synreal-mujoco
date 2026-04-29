import mujoco.viewer

import synreal_mujoco.s3d_mj as s3d_mj
import synreal_mujoco.s3d_scene as s3d_scene
import synreal_mujoco.s3d_scene_stepper as s3d_scene_stepper
import synreal_mujoco.data_classes as dc
from pathlib import Path

s3d_mj.log_in_simulation(login_file='../../simulation_login.json') # this line is optional, but a login prompt will pop up latter

asset_dir = Path(__file__).parent.resolve() / 'xml_projects'

s3d_scene_builder = s3d_scene.s3d_scene_builder()

def rb_builder(name):
    ret = dc.rigid_body_builder()
    if name == 'link8': #TODO: link8/0 should be ok
        ret.attrib.mass = 3e-2
    return ret


s3d_scene_builder.add_mjcf_rigidbodies( asset_dir/ 'piper_secription'/'piper_description.xml', rb_builder)

######### cloth
cloth_attrib = s3d_scene_builder.add_cloth_by_file( asset_dir / 'clothes'/ '50k_plane.obj')
cloth_attrib.stretch_stiff.x = 200
cloth_attrib.stretch_stiff.y = 200
cloth_attrib.stretch_stiff.z = 200

m,d,s = s3d_scene_builder.build()

l_s3d_scene_stepper = s3d_scene_stepper.s3d_scene_stepper(m,d,s)

with mujoco.viewer. launch_passive(m, d) as viewer:

    while viewer. is_running():

        mujoco. mj_step(m, d)

        l_s3d_scene_stepper. set_rigid_body_pos_to_scene()
        l_s3d_scene_stepper. step_sim()
        l_s3d_scene_stepper. set_render_pos_to_mujoco()

        viewer. sync()

