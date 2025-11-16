import style3dsim as sim
import mujoco
import numpy as np

import mujoco_style3d._mj_data_helper as _mj_data_helper

from dataclasses import dataclass

@dataclass
class ClothParams:
    """ 布料物理属性的数据类"""
    # stretch_stiff: sim.Vec3f = sim.Vec3f(7.5e4,2.0e5, 2.0e4)
    stretch_stiff: sim.Vec3f = sim.Vec3f(75000,200000, 20000)
    bend_stiff: sim.Vec3f = sim.Vec3f(1e3, 2e3, 1.5e3)
    density: float = 220
    static_friction: float = 0.6
    dynamic_friction: float = 0.6


def add_piece_to_sim(m,d,world, cloth_params: ClothParams = None):
    sim_pieces = []
    piece_names = []
    if cloth_params is None:
        cloth_params = ClothParams()
    add_piece = lambda x,t,name :_add_piece_to_sim(x,t,name,world,sim_pieces,piece_names, cloth_params)
    _mj_data_helper.for_each_piece(m, d, add_piece)
    return sim_pieces,piece_names


def _add_piece_to_sim(x, t, name, world, sim_pieces, piece_names, params: ClothParams):   
    cloth = sim.Cloth(t, x, np.array([], dtype=float), False)

    cloth_attrib = sim.ClothAttrib()
    cloth_attrib.stretch_stiff = params.stretch_stiff
    cloth_attrib.bend_stiff = params.bend_stiff
    cloth_attrib.density = params.density
    cloth_attrib.static_friction = params.static_friction
    cloth_attrib.dynamic_friction = params.dynamic_friction

    cloth.set_attrib(cloth_attrib)
    cloth.attach(world)

    sim_pieces.append(cloth)
    piece_names.append(name)

def _add_rigid_body_to_sim(i, x, t, xmat, xpos, world, rigid_bodies):

    transform = _mj_data_helper. to_sim_transfrom(xmat,xpos)

    mesh = sim.Mesh(t, x)

    rigid_body = sim.RigidBody(mesh,transform)

    rigid_body_attrib = sim.RigidBodyAttrib()
    rigid_body_attrib.dynamic_friction = 0.03
    rigid_body_attrib.static_friction = 0.03
    rigid_body_attrib.mass = 3e-2

    rigid_body.set_attrib(rigid_body_attrib)

    rigid_body.set_pin(True)

    rigid_body.attach(world)

    rigid_bodies.append(rigid_body)

def _set_rigid_body_to_sim(i, x, t, xmat,xpos, rigid_bodies,last_rigid_body_transform):
    rigid_bodies[i].move(last_rigid_body_transform[i],_mj_data_helper.to_sim_transfrom(xmat,xpos))


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
    world_attrib.time_step = 0.001
    world_attrib.enable_rigid_self_collision = False
    world.set_attrib(world_attrib)
    return world


def load_data(xml_path):
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)  # so that d is populated by m

    return m, d

# def add_piece_to_sim(m, d, world):
#     sim_pieces = []
#     piece_names = []
#     add_piece = lambda x, t, name :_add_piece_to_sim(x,t,name,world,sim_pieces,piece_names)
#     _mj_data_helper.for_each_piece(m, d, add_piece)
#     return sim_pieces,piece_names


def add_rigid_body_to_sim(m, d, world):
    objects = []

    _mj_data_helper.for_each_rigid_meshes(m, d, lambda i,x,t,xmat,xpos:_add_rigid_body_to_sim(i,x,t,xmat,xpos, world, objects))

    return  objects

def set_piece_pos_to_mujoco(m, d, sim_pieces,piece_names):
    for cloth, cloth_name in zip(sim_pieces, piece_names):
        x = cloth.get_positions()
        _mj_data_helper.set_piece_positions(m, d, cloth_name, x)

def set_rigid_body_pos_to_sim(m, d,  rigid_bodies):
    last_rigid_body_transform = [rb.get_transform() for rb in rigid_bodies]
    _mj_data_helper.for_each_rigid_meshes(m, d, lambda i, x, t, xmat,xpos : _set_rigid_body_to_sim(i, x, t, xmat,xpos, rigid_bodies, last_rigid_body_transform))

