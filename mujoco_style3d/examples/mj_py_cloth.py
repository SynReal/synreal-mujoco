import time

import style3dsim as sim

import mujoco.viewer

import mujoco_style3d.s3d_mj as s3d_mj

def log_in(usr,pw):
    sim.login(usr, pw, True, None)

log_in('simsdk001','xSXiaCMd')

m , d = s3d_mj.load_data('xml_projects/piper_secription_with_cloth/piper_description.xml')

world = s3d_mj.get_a_sim_world()

sim_pieces, piece_names = s3d_mj.add_piece_to_sim(m, d, world)
rigid_bodies = s3d_mj.add_rigid_body_to_sim(m, d, world)

sync_rate = 1

with mujoco.viewer.launch_passive(m, d) as viewer:

    fi=0
    while viewer.is_running():

        begin0_t = time.time()
        mujoco.mj_step(m, d)

        begin1_t = time.time()
        s3d_mj.set_rigid_body_pos_to_sim(m, d,  rigid_bodies)
        world.step_sim()
        end1_t = time.time()
        duration1 = end1_t - begin1_t

        world.fetch_sim(0)

        s3d_mj.set_piece_pos_to_mujoco(m, d, sim_pieces, piece_names)

        if fi % sync_rate == 0:
            viewer.sync()
        fi += 1

        end0_t = time.time()
        duration0 = end0_t - begin0_t
        print("fps = ", 1. / duration0,'\t', 1. / duration1)


