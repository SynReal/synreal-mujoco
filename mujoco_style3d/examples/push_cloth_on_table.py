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
rb_id = s3d_mj.get_geom_parent(m,d)

mocap_id = m.body('palm').mocapid[0]

id1 = m.actuator('ffa1').id
id2 = m.actuator('ffa2').id

force_rb = []

sync_rate = 1

with mujoco.viewer. launch_passive(m, d) as viewer:

    fi = 0

    while viewer.is_running():

        begin0_t = time. time()

        #palm_pos = d. mocap_pos[mocap_id]
        x = 0.004 * fi
        if  x < 1.0:

            z = np. clip( 0.3 - 0.001 * float(fi), 0.21 , 1 )

            d. mocap_pos[mocap_id] = np. array([ x , 0.5 , z])

            target_angle = 1.0
            d.ctrl[id1] = target_angle
            d.ctrl[id2] = target_angle
            #d.ctrl[3] = target_angle

            #force set to  ctrl

            if len(force_rb) > 0:
                for i in range( len(rigid_bodies) ):
                    l_rb_id =  rb_id[i]
                    rb_force = force_rb[i]
                    added = False
                    for f, bary in zip(*rb_force): # force and bary

                        orientation = d. xmat[l_rb_id]
                        orientation = orientation. reshape(3,3)
                        torque = orientation @ bary
                        #torque = np.array([0,0,0])
                        d. xfrc_applied[l_rb_id] += [ f[0], f[1], f[2], torque[0], torque[1], torque[2] ]
                        added = True
                    if added :
                        pass
                        #ft = d. xfrc_applied[l_rb_id]
                        #print(f'force torque: {l_rb_id}, {ft[0]:.2e} {ft[1]:.2e} {ft[2]:.2e} {ft[3]:.2e} {ft[4]:.2e} {ft[5]:.2e}')

        mujoco. mj_step(m, d)

        begin1_t = time. time()

        s3d_mj. set_rigid_body_pos_to_sim(m, d, rigid_bodies)
        world. step_sim()

        force_rb = []
        for i in range(len(rigid_bodies)):
            rb = rigid_bodies[i]
            f_rb = s3d_mj. get_collision_force_from_piece( rb )
            force_rb.append(f_rb)

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


