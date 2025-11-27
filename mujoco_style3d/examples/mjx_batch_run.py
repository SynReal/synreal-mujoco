
import mujoco.viewer

import mujoco_style3d.s3d_mjx as s3d_mjx
import mujoco_style3d.s3d_mj as s3d_mj


s3d_mj.log_in_simulation(login_file='../../simulation_login.json') # this line is optional, but a login prompt will pop up latter

batch_size = 2
mjx_mng = s3d_mjx.mjx_data_manager('xml_projects/piper_secription_with_cloth/piper_description1.xml', batch_size)
#mjx_mng = s3d_mjx.mjx_data_manager('xml_projects/hanging_cube/hanging_cube.xml', batch_size)

batch_to_show = 0

mj_model, mj_data = mjx_mng.get_mj_data(batch_to_show)

with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:

    fi = 0

    while viewer.is_running() and fi < 500 :

        # set action_batch here
        #mjx_mng.set_rigidbody_action(act_batch)

        mjx_mng.step()

        mjx_mng.set_rigidbody_pos_to_mujoco( batch_to_show)
        mjx_mng.set_cloth_pos_to_mujoco(batch_to_show)

        print(f"frame {fi} ")
        viewer.sync()
        fi += 1
