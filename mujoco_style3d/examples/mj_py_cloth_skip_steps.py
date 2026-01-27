import time

import mujoco.viewer

import mujoco_style3d.s3d_mj as s3d_mj
from mujoco_style3d import cloth_property
from mujoco_style3d import step_skipper

s3d_mj.log_in_simulation(login_file='../../simulation_login.json') # this line is optional, but a login prompt will pop up latter

m , d = s3d_mj.load_data('xml_projects/piper_secription_with_cloth/piper_description.xml')

world = s3d_mj.get_a_sim_world(m)

sim_cloth, cloth_names = s3d_mj.add_cloth_to_sim(m, d, world, lambda nama , attrib: cloth_property.set_cloth_property_default(attrib))
rigid_bodies = s3d_mj.add_rigid_body_to_sim(m, d, world)

sync_rate = 1

l_step_skipper = step_skipper.step_skipper()
rb_x, _ = s3d_mj.get_rigid_body_mesh(m, d)
l_step_skipper.set_rigidbody_refpos(rb_x)

with mujoco.viewer. launch_passive(m, d) as viewer:

    fi = 0

    while viewer.is_running():

        begin0_t = time. time()

        rb_mat , rb_pos = s3d_mj. get_rigid_body_transform(m,d)
        l_step_skipper. set_pos( s3d_mj.get_cloth_pos(sim_cloth, cloth_names), rb_mat, rb_pos, m.opt.timestep )

        mujoco. mj_step(m, d)

        begin1_t = time. time()

        if not l_step_skipper. safe_to_skip(): # skip simulation if possible
            s3d_mj. set_rigid_body_pos_with_velocity( rigid_bodies, l_step_skipper.get_last_rigid_body_transform(),l_step_skipper.get_curr_rigid_body_transform() )
            world. step_sim()

        end1_t = time. time()
        duration1 = end1_t - begin1_t

        world. fetch_sim(0)

        s3d_mj. set_cloth_pos_to_mujoco(m, d, sim_cloth, cloth_names)

        if fi % sync_rate == 0:
            viewer. sync()
        fi += 1

        end0_t = time. time()
        duration0 = end0_t - begin0_t
        print("fps = ", 1. / duration0,'\t', 1. / duration1)


