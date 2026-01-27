import time

import mujoco.viewer
import numpy as np

import mujoco_style3d.smj as smj

def rigid_body_property_fn(rb_name,attrib):
    if  rb_name == 'table_mesh':
        attrib. dynamic_friction = 0.007
        attrib. static_friction = 0.007
        attrib. mass = 3e-2
    else:
        attrib. dynamic_friction = 0.03
        attrib. static_friction = 0.03
        attrib. mass = 3e-2

m , d, mp = smj. smj_load_data('xml_projects/wonik_allegro/left_hand.xml', rb_property_fn = rigid_body_property_fn)

def set_finger_target_pos(m,d):

    target_angle = 1.0

    # first finger
    smj. set_actuator_target_pos(m, d, 'ffa1', target_angle)
    smj. set_actuator_target_pos(m, d, 'ffa2', target_angle)

    # middle finger
    smj. set_actuator_target_pos(m, d, 'mfa1', target_angle)
    smj. set_actuator_target_pos(m, d, 'mfa2', target_angle)

    # ring finger (wu ming zhi)
    smj. set_actuator_target_pos(m, d, 'rfa1', target_angle)
    smj. set_actuator_target_pos(m, d, 'rfa2', target_angle)


with mujoco.viewer. launch_passive(m, d) as viewer:

    fi = 0

    while viewer. is_running():

        begin0_t = time. time()

        x = 0.004 * fi

        if  x < 1.2:

            z = np. clip( 0.3 - 0.001 * float(fi), 0.210 , 1 )

            smj. set_mocap_pos(m, d,'palm', np. array([ x , 0.5 , z]))

            set_finger_target_pos(m,d)

            smj. update_rigidbody_cloth_collision_force(m, d, mp)
            smj. apply_collision_force_to_rigidbody(m, d, mp) ## cloth affacts rigid body

        smj. smj_rigid_body_step(m, d)

        begin1_t = time. time()

        smj. update_rigidbody_to_cloth(m,d,mp)   ## rigid body affacts cloth

        smj. smj_cloth_step(mp)

        end1_t = time. time()

        duration1 = end1_t - begin1_t

        smj. update_cloth_to_rigid_body(m,d,mp)  ## fetch cloth position back , for mujoco visual, no force apply to rigidbody yet

        viewer. sync()

        fi += 1

        end0_t = time. time()
        duration0 = end0_t - begin0_t
        print(f'fps: {1. / duration0:.2f}, sim fps: {1. / duration1:.2f}')

