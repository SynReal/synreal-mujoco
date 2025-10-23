import style3dsim as sim
import mujoco
import numpy as np

import mj_to_sim


def _add_piece_to_sim(x,t,name,world,sim_pieces,piece_names):

    cloth = sim.Cloth(t, x, np.array([], dtype=float), False)

    cloth_attrib = sim.ClothAttrib()
    cloth_attrib.stretch_stiff = sim.Vec3f(120, 100, 80)
    cloth_attrib.bend_stiff = sim.Vec3f(1e-6, 1e-6, 1e-6)
    cloth_attrib.density = 0.2
    cloth_attrib.static_friction = 0.03
    cloth_attrib.dynamic_friction = 0.03

    cloth.set_attrib(cloth_attrib)

    cloth.attach(world)

    sim_pieces.append(cloth)
    piece_names.append(name)


def _log_callback(file_name: str, func_name: str, line: int, level: sim.LogLevel, message: str):
    if level == sim.LogLevel.INFO:
        print("[info]: ", message)
    elif level == sim.LogLevel.ERROR:
        print("[error]: ", message)
    elif level == sim.LogLevel.WARNING:
        print("[warning]: ", message)
    elif level == sim.LogLevel.DEBUG:
        print("[debug]: ", message)

def get_a_sim_world():
    sim.set_log_callback(_log_callback)

    world = sim.World()
    world_attrib = sim.WorldAttrib()
    world_attrib.enable_gpu = True
    world_attrib.gravity = sim.Vec3f(0, 0, -9.81)
    world_attrib.ground_direction = sim.Vec3f(0., 0., 1.)
    world_attrib.time_step = 0.01
    world.set_attrib(world_attrib)
    return world


def load_data(xml_path):
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)  # so that d is populated by m
    return m,d

def add_piece_to_sim(m,d,world):
    sim_pieces = []
    piece_names = []
    add_piece = lambda x,t,name :_add_piece_to_sim(x,t,name,world,sim_pieces,piece_names)
    mj_to_sim.for_each_piece(m, d, add_piece)
    return sim_pieces,piece_names

def set_piece_to_mujoco(m,d,sim_pieces,piece_names):
    for cloth, cloth_name in zip(sim_pieces, piece_names):
        x = cloth.get_positions()
        mj_to_sim.set_positions(m, d, cloth_name, x)
