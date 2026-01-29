import time

import mujoco.viewer
import numpy as np

import mujoco_style3d.smj as smj

import json
from pathlib import Path

def rigid_body_property_fn(geo_name,attrib):
    if  geo_name == 'table_box' or geo_name == 'table_mesh':
        print(f' set {geo_name} rigid body property')
        attrib. dynamic_friction = 0.007
        attrib. static_friction = 0.007
        attrib. mass = 3e-2
    else:
        attrib. dynamic_friction = 0.03
        attrib. static_friction = 0.03
        attrib. mass = 3e-2



mjcf_file = 'xml_projects/test/some_hand/left_hand.xml'
#mjcf_file = 'xml_projects/wonik_allegro/left_hand.xml'

mjcf_dir = Path(mjcf_file).parent.resolve()
trajectory_file = mjcf_dir / 'trajectory_param.json'

m , d, mp = smj. smj_load_data(mjcf_file, rb_property_fn = rigid_body_property_fn)

with open(trajectory_file,'r') as fin:
    data=json.load(fin)

drop_rate = data['drop_rate']
hand_z_min = data['hand_z_min']

with mujoco. viewer. launch_passive(m, d) as viewer:

    fi = 0

    while viewer. is_running():

        begin0_t = time. time()

        x = 0.004 * fi

        if  x < 1.2:

            z = np. clip( 0.3 - drop_rate * float(fi), hand_z_min , 1 )

            smj. set_mocap_pos(m, d,'palm', np. array([ x , 0.5 , z]))

            smj. update_rigidbody_cloth_collision_force(m, d, mp)
            smj. apply_collision_force_to_rigidbody(m, d, mp) ## cloth affacts rigid body

        smj. smj_rigid_body_step(m, d)

        smj. update_rigidbody_to_cloth(m,d,mp)   ## rigid body affacts cloth

        begin1_t = time. time()

        smj. smj_cloth_step(mp)

        end1_t = time. time()

        duration1 = end1_t - begin1_t

        smj. update_cloth_to_rigid_body(m, d, mp)  ## fetch cloth position back , for mujoco visual, no force apply to rigidbody yet

        viewer. sync()

        fi += 1

        end0_t = time. time()
        duration0 = end0_t - begin0_t
        print(f'fps: {1. / duration0:.2f}, sim fps: {1. / duration1:.2f}')

