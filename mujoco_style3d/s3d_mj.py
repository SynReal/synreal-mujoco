import os.path

import style3dsim as sim
import mujoco
import numpy as np
import json

import mujoco_style3d._mj_data_helper as _mj_data_helper

def _add_cloth_to_sim(x, t, name, world, sim_clothes, cloth_names, fabric_getter):

    cloth = sim.Cloth(t, x, np.array([], dtype=float), False)

    cloth_attrib = fabric_getter(name)
    cloth.set_attrib(cloth_attrib)

    cloth.attach(world)

    sim_clothes.append(cloth)
    cloth_names.append(name)

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

def log_in_simulation(**kwargs):
    name=''

    if not sim.is_login():
        login_file = None
        if 'login_file' in kwargs:
            login_file = kwargs['login_file']

        if login_file and os.path.exists(login_file):
            with open(login_file,'r') as f:
                login=json.load(f)
                name = login['name']
                pass_word = login['pass_word']
        else:
            name = input('Enter your name : ')
            pass_word = input('Enter your password : ')

        sim.login(name, pass_word, True, None)

    if sim.is_login():
        print(f'login successful {name}')
    else:
        print('login failed')

def get_a_sim_world(m):

    log_in_simulation()

    sim.set_log_callback(_log_callback)

    world = sim.World()
    world_attrib = sim.WorldAttrib()
    world_attrib.enable_gpu = True
    world_attrib.gravity = sim.Vec3f(m.opt.gravity[0], m.opt.gravity[1], m.opt.gravity[2])
    world_attrib.ground_direction = sim.Vec3f(0., 0., 1.)
    world_attrib.time_step = m.opt.timestep
    world_attrib.enable_rigid_self_collision = False
    world_attrib.enable_collision_force_map_rigidbody_piece = True
    world.set_attrib(world_attrib)

    print(f'time step {world_attrib.time_step}')
    print(f'gravity {m.opt.gravity[0]},{m.opt.gravity[1]},{m.opt.gravity[2]} ')
    return world


def load_data(xml_path):
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)  # so that d is populated by m

    return m, d

def add_cloth_to_sim(m, d, world, cloth_property_getter):
    sim_clothes = []
    cloth_names = []
    add_cloth = lambda x, t, name :_add_cloth_to_sim(x, t, name, world, sim_clothes, cloth_names, cloth_property_getter)
    _mj_data_helper.for_each_cloth(m, d, add_cloth)
    return sim_clothes,cloth_names


def add_rigid_body_to_sim(m, d, world):
    objects = []

    _mj_data_helper.for_each_rigid_meshes(m, d, lambda i,x,t,xmat,xpos:_add_rigid_body_to_sim(i,x,t,xmat,xpos, world, objects))

    return  objects

def set_cloth_pos_to_mujoco(m, d, sim_clothes, cloth_names):


    for cloth, cloth_name in zip(sim_clothes, cloth_names):
        x = cloth.get_positions()
        _mj_data_helper.set_cloth_positions(m, d, cloth_name, x)

def set_rigid_body_pos_to_sim(m, d,  rigid_bodies):
    last_rigid_body_transform = [rb.get_transform() for rb in rigid_bodies]
    _mj_data_helper.for_each_rigid_meshes(m, d, lambda i, x, t, xmat,xpos : _set_rigid_body_to_sim(i, x, t, xmat,xpos, rigid_bodies, last_rigid_body_transform))

