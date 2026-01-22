import time

import mujoco.viewer
import numpy as np

import mujoco_style3d.s3d_mj as s3d_mj
from mujoco_style3d import cloth_property

s3d_mj.log_in_simulation(login_file='../../simulation_login.json') # this line is optional, but a login prompt will pop up latter

m , d = s3d_mj.load_data('xml_projects/wonik_allegro/left_hand.xml')

world = s3d_mj.get_a_sim_world(m)

sim_cloth, cloth_names = s3d_mj.add_cloth_to_sim(m, d, world, lambda nama : cloth_property.get_cloth_property_default())
rigid_bodies = s3d_mj.add_rigid_body_to_sim(m, d, world)

mocap_id = m.body('palm').mocapid[0]

sync_rate = 1

with mujoco.viewer. launch_passive(m, d) as viewer:

    fi = 0

    while viewer.is_running():

        begin0_t = time. time()

        d.mocap_pos[mocap_id]=np.array([ 0.01 * fi , 0.0, 0.3 ])

        mujoco. mj_step(m, d)

        begin1_t = time. time()

        s3d_mj. set_rigid_body_pos_to_sim(m, d, rigid_bodies)
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


