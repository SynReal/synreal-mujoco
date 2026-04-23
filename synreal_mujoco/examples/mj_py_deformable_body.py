import mujoco.viewer
import numpy as np

import synreal_mujoco.s3d_mj as s3d_mj
from synreal_mujoco import cloth_property
import synreal_mujoco.s3d_scene as s3d_scene



s3d_mj.log_in_simulation(login_file='../../simulation_login.json') # this line is optional, but a login prompt will pop up latter

m , d = s3d_mj.load_data('xml_projects/piper_secription/piper_description.xml')

world = s3d_mj.get_a_sim_world(m)

s3d_scene_builder = s3d_scene.s3d_scene_builder()
s3d_scene_builder.add_deformable_body_by_file_with_boundary_collision_faces('xml_projects/piper_secription/tets.vtk')

l_s3d_scene = s3d_scene_builder.build(world)

rigid_bodies = s3d_mj.add_rigid_body_to_sim(m, d, world, lambda name,attrib: cloth_property.set_rigid_body_property_default(attrib),False)

with mujoco.viewer. launch_passive(m, d) as viewer:

    while viewer.is_running():

        mujoco. mj_step(m, d)

        s3d_mj. set_rigid_body_pos_to_sim(m, d, rigid_bodies)
        world. step_sim()

        world. fetch_sim(0)

        #s3d_mj. set_cloth_pos_to_mujoco(m, d, sim_cloth, cloth_names)
        #s3d_mj. set_s3d_scene_to_mujoco(m, d, )

        viewer. sync()

