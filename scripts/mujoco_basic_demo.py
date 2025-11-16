import time

import mujoco
import mujoco.viewer
import os

import style3dsim as sim
import mujoco_style3d.s3d_mj as s3d_mj

try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    PROJECT_ROOT = os.getcwd()


# login to style3d
def log_in(usr,pw):
    sim.login(usr, pw, True, None)
log_in('SHJD_test01_en','YpCVTFAK')

# load xml
xml_path = os.path.join(PROJECT_ROOT, 'assets', 'mujoco_model', 'dual_piper_with_cloth_simple.xml')
m , d = s3d_mj.load_data(xml_path)

# cloth_params
from mujoco_style3d.s3d_mj import ClothParams
cloth_params = ClothParams()
cloth_params.stretch_stiff = sim.Vec3f(75000,200000, 20000)
cloth_params.bend_stiff = sim.Vec3f(1e3, 2e3, 1.5e3)
cloth_params.density = 220
cloth_params.static_friction = 0.6
cloth_params.dynamic_friction = 0.6



# create sim world
world = s3d_mj.get_a_sim_world()
sim_pieces, piece_names = s3d_mj.add_piece_to_sim(m, d, world,cloth_params=cloth_params) # deformable 
rigid_bodies = s3d_mj.add_rigid_body_to_sim(m, d, world) # rigid body

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


