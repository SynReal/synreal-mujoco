import os.path
from copy import deepcopy

import style3dsim as sim
import mujoco
import numpy as np
import json

import synreal_mujoco._mj_data_helper as _mj_data_helper
from synreal_mujoco import cloth_property

def _add_cloth_to_sim(x, t, collision_mask, collision_group, name, world, sim_clothes, cloth_names, fabric_setter):

    cloth = sim.Cloth(t, x, np.array([], dtype = float), False)

    cloth_attrib = sim.ClothAttrib()

    fabric_setter(name, cloth_attrib)

    cloth.set_attrib(cloth_attrib)

    cloth.attach(world)

    n = len(x)

    groups = np.full(n, collision_group)
    masks = np.full(n, collision_mask)

    ver_ind = np.arange(n)

    cloth.set_collision_group_masks( groups, masks, ver_ind)

    sim_clothes.append(cloth)
    cloth_names.append(name)

def _set_rigid_body_to_sim(i,  xmat, xpos, rigid_bodies, last_rigid_body_transform):
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

    name = ''

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
    if hasattr(world_attrib, "enable_plastic_bending"):
        world_attrib.enable_plastic_bending = True
    if hasattr(world_attrib, "enable_volume_conserve"):
        world_attrib.enable_volume_conserve = True

    world.set_attrib(world_attrib)

    print(f'time step {world_attrib.time_step}')
    print(f'gravity {m.opt.gravity[0]},{m.opt.gravity[1]},{m.opt.gravity[2]} ')
    return world


def load_data(xml_path):
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)  # so that d is populated by m

    return m, d


def add_cloth_to_sim(m, d, world,name_start_with_will_considered_cloth, cloth_property_setter):
    sim_clothes = []
    cloth_names = []
    add_cloth = lambda x, t, collision_mask, collision_group, name :_add_cloth_to_sim(x, t, collision_mask, collision_group, name, world, sim_clothes, cloth_names, cloth_property_setter)
    _mj_data_helper.for_each_cloth(m, d,name_start_with_will_considered_cloth, add_cloth  )
    return sim_clothes,cloth_names



def extract_convex_hull(model, mesh_id):
    """
    Extract the convex hull collision mesh from the model's mesh graph.
    """
    # Access the mesh graph data from the model
    adr = model.mesh_graphadr[mesh_id]

    if adr < 0:
        raise ValueError("No convex hull data found for the mesh.")

    graph = model.mesh_graph[adr:]

    numvert = graph[0]  # Number of vertices
    numface = graph[1]  # Number of faces

    # Extract the global vertex indices and faces
    vert_begin = 2 + numvert                    # see mujoco doc
    face_begin = 2 + 3 * numvert + 3 * numface  # see mujoco doc
    vert_globalid = graph[ vert_begin: vert_begin + numvert]
    face_localid = deepcopy(graph[face_begin: face_begin + 3 * numface])
    v_map={}
    for  vi, v in enumerate(vert_globalid):
        v_map[v] = vi

    # Extract the original mesh vertices from the model
    vadr = model. mesh_vertadr[mesh_id]
    verts_all = model. mesh_vert.reshape(-1, 3)[  vadr:]

    # Get the convex hull vertices using the global indices
    verts = verts_all[vert_globalid]

    for fv in range(len(face_localid)):
        v = face_localid[fv]
        vi = v_map[v]
        face_localid[fv] = vi

    # Reshape faces and map to the correct vertex indices
    faces = np.array(face_localid).reshape(-1, 3)

    return verts, faces


def add_rigid_body_to_sim(m, d, world, property_fn=None, rigidbody_with_convex_hull=False):

    if property_fn is None:
        def property_fn(name, attrib):
            cloth_property.set_rigid_body_property_default(attrib)

    objects = []

    xmat = _mj_data_helper. _mj_get_attr(d, "geom_xmat")
    xpos = _mj_data_helper. _mj_get_attr(d, "geom_xpos")

    contype = _mj_data_helper._mj_get_attr(m, "geom_contype")
    conaffinity =  _mj_data_helper._mj_get_attr(m,"geom_conaffinity")

    geo_size = _mj_data_helper._mj_get_attr(m,"geom_size")

    def __add_rigid_body(slot_i, geom_id, mesh_id, rb_id , geom_type):

        t = _mj_data_helper. _get_mesh_tri(mesh_id, m)
        x = _mj_data_helper. _get_mesh_pos(mesh_id, m)

        if rigidbody_with_convex_hull:
            if m. mesh_graphadr[mesh_id] >= 0:
                x, t = extract_convex_hull(m, mesh_id)

        geo_pos = xpos[geom_id]
        geo_mat = xmat[geom_id]

        transform = _mj_data_helper. to_sim_transfrom(geo_mat, geo_pos)

        mesh = sim. Mesh(t, x)

        if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            rigid_body = sim.RigidBody(mesh, transform)
        elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            sphereSize = sim.SphereSize()
            rigid_body = sim.RigidBody(sphereSize, transform)
        elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            s = geo_size[geom_id]
            boxSize =  sim.BoxSize()
            boxSize.length_x = 2 * s[0]
            boxSize.length_y = 2 * s[1]
            boxSize.length_z = 2 * s[2]
            rigid_body = sim.RigidBody(boxSize, transform)
        elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            cylinderSize =  sim.CylinderSize()
            rigid_body = sim.RigidBody(cylinderSize, transform)
        else:
            print('unknown geometry type!')
            return

        geom_name = mujoco. mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, geom_id)

        rigid_body_attrib = sim. RigidBodyAttrib()
        property_fn(geom_name, rigid_body_attrib)
        rigid_body. set_attrib(rigid_body_attrib)

        rigid_body. set_pin(True)
        rigid_body. set_collision_group( contype[geom_id] )
        rigid_body. set_collision_mask( conaffinity[geom_id] )

        rigid_body. attach( world )

        objects. append( rigid_body )

    _mj_data_helper.for_each_geom_mesh(m, d, __add_rigid_body )

    return  objects

def set_cloth_pos_to_mujoco(m, d, sim_clothes, cloth_names):

    for cloth, cloth_name in zip(sim_clothes, cloth_names):
        x = cloth.get_positions()
        _mj_data_helper.set_cloth_positions(m, d, cloth_name, x)

def set_rigid_body_pos_to_sim(m, d,  rigid_bodies):
    last_rigid_body_transform = [rb.get_transform() for rb in rigid_bodies]
    def rigid_fn(rigid_i, x, t, geo_mat, geo_pos , collision_mask, collision_group):
        _set_rigid_body_to_sim(rigid_i,  geo_mat, geo_pos, rigid_bodies, last_rigid_body_transform)

    _mj_data_helper.for_each_rigid_meshes(m, d, rigid_fn)


def set_rigid_body_pos_with_velocity(rigid_bodies, last_transform, curr_transform ):
    for ri in range(len(rigid_bodies)):
        rb = rigid_bodies[ri]
        last_rot,last_translate = last_transform if last_transform is not None else curr_transform
        curr_rot,curr_translate = curr_transform
        rb.move(_mj_data_helper.to_sim_transfrom(last_rot[ri],last_translate[ri]), _mj_data_helper.to_sim_transfrom(curr_rot[ri],curr_translate[ri]))


def get_cloth_pos(sim_clothes, cloth_names):
    ret = []
    for cloth, cloth_name in zip(sim_clothes, cloth_names):
        x = cloth.get_positions()
        ret.append(x)
    return ret


def get_rigid_body_mesh( m: mujoco.MjModel, d: mujoco.MjData):
    ret_x=[]
    ret_t=[]

    def append_data(rigid_i, x, t, geo_mat, geo_pos , collision_mask, collision_group):
        ret_x.append(x)
        ret_t.append(t)

    _mj_data_helper.for_each_rigid_meshes(m, d, append_data)

    return ret_x,ret_t

def get_rigid_body_transform( m: mujoco.MjModel, d: mujoco.MjData):
    ret_geo_mat=[]
    ret_geo_pos=[]

    def append_data(rigid_i, x, t, geo_mat, geo_pos , collision_mask, collision_group):
        ret_geo_mat.append(geo_mat)
        ret_geo_pos.append(geo_pos)

    _mj_data_helper.for_each_rigid_meshes(m, d, append_data)

    return ret_geo_mat,ret_geo_pos


def get_collision_force_from_piece(rigidbody):
    return rigidbody.get_collision_force_piece()


