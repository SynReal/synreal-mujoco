import time
import os

import style3dsim as sim

import mujoco.viewer

import helper

def log_in(usr,pw):
    sim.login(usr, pw, True, None)

log_in('simsdk001','xSXiaCMd')

#'F:/mujoco_proj/piper_description/mujoco_model/piper_bimanual_description_act_tmp.xml'
#m, d = helper.load_data('F:/mujoco_proj/4_grid/scene.xml')
#m , d = helper.load_data('F:/mujoco_proj/cloth_drop_to_arm/main.xml')
#m , d = helper.load_data('F:/mujoco_proj/piper_description/mujoco_model/piper_bimanual_description_act_tmp_no_plugin.xml')
m , d = helper.load_data('F:/mujoco_proj/piper_secription_with_cloth/piper_description.xml')

world = helper.get_a_sim_world()

sim_pieces, piece_names = helper.add_piece_to_sim(m,d,world)
rigid_bodies = helper.add_rigid_body_to_sim(m,d,world)

with mujoco.viewer.launch_passive(m, d) as viewer:

    while True:
        begin0_t = time.time()
        mujoco.mj_step(m, d)

        begin1_t = time.time()
        helper.set_rigid_body_pos_to_sim(m,d,world,rigid_bodies)
        world.step_sim()
        end1_t = time.time()
        duration1 = end1_t - begin1_t

        world.fetch_sim(0)

        helper.set_piece_pos_to_mujoco(m, d, sim_pieces, piece_names)

        viewer.sync()

        end0_t = time.time()
        duration0 = end0_t - begin0_t
        print("fps = ", 1. / duration0, 1. / duration1)
