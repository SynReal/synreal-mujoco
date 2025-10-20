import time

import style3dsim as sim
import numpy as np

import mujoco
import mujoco.viewer

import mj_to_sim

def log_callback(file_name: str, func_name: str, line: int, level: sim.LogLevel, message: str):
	if level == sim.LogLevel.INFO:
		print("[info]: ", message)
	elif level == sim.LogLevel.ERROR:
		print("[error]: ", message)
	elif level == sim.LogLevel.WARNING:
		print("[warning]: ", message)
	elif level == sim.LogLevel.DEBUG:
		print("[debug]: ", message)


def load_data(xml_path):
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)  # so that d is populated by m
    return m,d


def add_piece_to_sim(x,t,name):

    cloth = sim.Cloth(t, x, np.array([], dtype=float), False)

    cloth_attrib = sim.ClothAttrib()
    cloth_attrib.stretch_stiff = sim.Vec3f(120, 100, 80)
    cloth_attrib.bend_stiff = sim.Vec3f(1e-6, 1e-6, 1e-6)
    cloth_attrib.density = 0.2
    cloth_attrib.static_friction = 0.03
    cloth_attrib.dynamic_friction = 0.03

    cloth.set_attrib(cloth_attrib)

    global world

    cloth.attach(world)

    sim_pieces.append(cloth)
    piece_names.append(name)

def log_in(usr,pw):
    sim.login(usr, pw, True, None)


def sim_init():
    sim.set_log_callback(log_callback)

    world = sim.World()
    world_attrib = sim.WorldAttrib()
    world_attrib.enable_gpu = True
    world_attrib.gravity = sim.Vec3f(0, 0, -9.81)
    world_attrib.time_step = 0.01
    world.set_attrib(world_attrib)
    return world


log_in('simsdk001','xSXiaCMd')

#'F:/mujoco_proj/piper_description/mujoco_model/piper_bimanual_description_act_tmp.xml'
m,d=load_data('F:/mujoco_proj/4_grid/scene.xml')

world=sim_init()

sim_pieces=[]
piece_names=[]
mj_to_sim.for_each_piece(m, d, add_piece_to_sim)

with mujoco.viewer.launch_passive(m, d) as viewer:
    while True:
        begin0_t = time.time()
        mujoco.mj_step(m, d)

        begin1_t = time.time()
        world.step_sim()
        end1_t = time.time()
        duration1 = end1_t - begin1_t

        world.fetch_sim(0)
        for cloth,cloth_name in zip(sim_pieces,piece_names):
            x = cloth.get_positions()
            mj_to_sim.set_positions(m, d, cloth_name, x)
        viewer.sync()

        end0_t = time.time()
        duration0 = end0_t - begin0_t
        print("fps = ", 1. / duration0, 1. / duration1)
