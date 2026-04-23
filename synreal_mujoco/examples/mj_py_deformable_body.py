import mujoco.viewer

import synreal_mujoco.s3d_mj as s3d_mj
import synreal_mujoco.s3d_scene as s3d_scene
import synreal_mujoco.s3d_scene_stepper as s3d_scene_stepper

s3d_mj.log_in_simulation(login_file='../../simulation_login.json') # this line is optional, but a login prompt will pop up latter

s3d_scene_builder = s3d_scene.s3d_scene_builder()
s3d_scene_builder.add_mjcf_rigidbodies('xml_projects/piper_secription/piper_description.xml')
s3d_scene_builder.add_cloth_by_file('xml_projects/piper_secription/50k_plane.obj')
#s3d_scene_builder.add_deformable_body_by_file_with_boundary_collision_faces('xml_projects/piper_secription/tets.vtk')

m,d,s = s3d_scene_builder.build()

l_s3d_scene_stepper = s3d_scene_stepper.s3d_scene_stepper(m,d,s)

with mujoco.viewer. launch_passive(m, d) as viewer:

    while viewer. is_running():

        mujoco. mj_step(m, d)

        l_s3d_scene_stepper.set_rigid_body_pos_to_scene()
        l_s3d_scene_stepper.step_sim()
        l_s3d_scene_stepper. set_render_pos_to_mujoco()

        viewer. sync()

