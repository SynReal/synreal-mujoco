import style3dsim as sim
import mujoco
import numpy as np


class Flexible:
    # This is a dummy class to hold constants from gx_utils.dtype.Flexible
    EXTRA_VERTEX_BY_STYLE3D_AT_END = 1

def _mj_get_attr(obj, *names):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    raise AttributeError(f"None of attributes present: {names}")

##############
def _get_flex_pos_buffer(d: mujoco.MjData):
    #return _mj_get_attr(d,  "flex_xpos")
    return _mj_get_attr(d, "flexvert_xpos" )

def _get_flex_vert_num_buffer(m: mujoco.MjModel):
    return _mj_get_attr(m, "flex_vertnum" )

def _get_flex_vert_offset_buffer(m: mujoco.MjModel):
    return _mj_get_attr(m, "flex_vertadr" )

def _get_flex_pos(id, m: mujoco.MjModel, d: mujoco.MjData):
    x = _get_flex_pos_buffer(d);
    offset = _get_flex_vert_offset_buffer(m)
    num = _get_flex_vert_num_buffer(m)
    return x[ offset[id] : offset[id]+num[id] - Flexible.EXTRA_VERTEX_BY_STYLE3D_AT_END] [:]

def _set_flex_pos(id, m: mujoco.MjModel, d: mujoco.MjData, verts):
    x = _get_flex_pos_buffer(d);
    offset = _get_flex_vert_offset_buffer(m)
    num = _get_flex_vert_num_buffer(m)
    x[ offset[id] : offset[id]+num[id] - Flexible.EXTRA_VERTEX_BY_STYLE3D_AT_END] [:] = verts


def _get_flex_tri_buffer(m: mujoco.MjModel):
    return _mj_get_attr(m, "flex_elem")

def _get_flex_tri_offset_buffer(m: mujoco.MjModel):
    return _mj_get_attr(m, "flex_elemadr")

def _get_flex_tri_num_buffer(m: mujoco.MjModel):
    return _mj_get_attr(m, "flex_elemnum")

def _get_flex_tri(id, m: mujoco.MjModel):
    offset=_get_flex_tri_offset_buffer(m)[id]
    num=_get_flex_tri_num_buffer(m)[id]
    return _get_flex_tri_buffer(m)[offset * 3:(offset + num) * 3].reshape(-1, 3)

def _get_flex_contype(index, m: mujoco.MjModel):
    return _mj_get_attr(m, "flex_contype")[index]

def _get_flex_conaffinity(index, m: mujoco.MjModel):
    return _mj_get_attr(m, "flex_conaffinity")[index]

def _set_flex_vertices(m: mujoco.MjModel, d: mujoco.MjData, flex_name: str, verts: np.ndarray) -> None:
    id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_FLEX, flex_name)
    _set_flex_pos(id, m, d, verts)

##############
def _get_mesh_pos(id,m: mujoco.MjModel):
    mesh_vert = _mj_get_attr(m, "mesh_vert")
    vert_begin = _mj_get_attr(m, "mesh_vertadr")
    vert_num = _mj_get_attr(m, "mesh_vertnum")

    v_begin = vert_begin[id]
    v_end = vert_begin[id] + vert_num[id]
    x = mesh_vert[v_begin:v_end, :]
    return x

def _get_mesh_tri(id, m: mujoco.MjModel):
    mesh_face = _mj_get_attr(m, "mesh_face")
    face_begin = _mj_get_attr(m, "mesh_faceadr")
    face_num = _mj_get_attr(m, "mesh_facenum")

    t_begin = face_begin[id]
    t_end = face_begin[id] + face_num[id]
    t = mesh_face[t_begin:t_end, :]
    return t


def _get_geo_num(m: mujoco.MjModel):
    return _mj_get_attr(m, "ngeom")

def to_sim_transfrom(xmat,xpos):
    transform = sim.Transform()
    transform.translation.x = xpos[0]
    transform.translation.y = xpos[1]
    transform.translation.z = xpos[2]

    transform.scale = sim.Vec3f(1, 1, 1)

    mat = sim.Matrix3f(
        sim.Vec3f(xmat[0], xmat[3], xmat[6]),
        sim.Vec3f(xmat[1], xmat[4], xmat[7]),
        sim.Vec3f(xmat[2], xmat[5], xmat[8])
    )

    transform.rotation = sim.Quat(mat)
    return  transform

def for_each_cloth(m: mujoco.MjModel, d: mujoco.MjData, fn ):
    vert_num = _get_flex_vert_num_buffer(m)

    num_flex = getattr(m, "nflex", 0)

    if num_flex != len(vert_num):
        return

    cloth_num = len(vert_num)
    for i in range(cloth_num):

        x = _get_flex_pos(i, m, d)
        t = _get_flex_tri(i, m)

        contype = _get_flex_contype(i,m)
        conaffinity = _get_flex_conaffinity(i,m)

        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_FLEX, i)

        fn( x=x, t=t, name = name, collision_mask = conaffinity, collision_group = contype )


def for_each_geom_mesh(m: mujoco.MjModel,d: mujoco.MjData, fn):
    mesh_ids = _mj_get_attr(m, "geom_dataid")
    geom_type = _mj_get_attr(m, "geom_type")
    mesh_graph_begin = _mj_get_attr(m, "mesh_graphadr")
    rigidbody_id = _mj_get_attr(m, "geom_bodyid")

    slot_i = 0
    geom_num = _get_geo_num(m)
    for i in range(geom_num):

        mesh_id = mesh_ids[i]

        if mesh_id < 0: # refer to a exsited mesh
            continue

        if geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH:   # geom type is mesh type
            continue

        if  mesh_graph_begin[mesh_id] < 0: # is a collision mesh
            continue

        rb_id = rigidbody_id[i]
        geom_id = i
        fn(slot_i,geom_id, mesh_id, rb_id)
        slot_i+=1



def for_each_rigid_meshes(m: mujoco.MjModel,d: mujoco.MjData, fn):

    contype = _mj_get_attr(m, "geom_contype")
    conaffinity =  _mj_get_attr(m,"geom_conaffinity")

    def rigid_mesh_fn(slot_i, geom_id, mesh_id, rb_id):
        t = _get_mesh_tri(mesh_id, m)
        x = _get_mesh_pos(mesh_id, m)

        xmat =_mj_get_attr(d, "geom_xmat" )
        xpos =_mj_get_attr(d, "geom_xpos" )

        geo_pos = xpos[geom_id]
        geo_mat = xmat[geom_id]

        fn( rigid_i = slot_i, x=x, t=t, geo_mat=geo_mat, geo_pos=geo_pos, collision_mask = conaffinity[geom_id], collision_group = contype[geom_id])

    for_each_geom_mesh(m,d,rigid_mesh_fn)


def set_cloth_positions(m: mujoco.MjModel, d: mujoco.MjData, mesh_name, x):
    _set_flex_vertices(m,d,mesh_name,x)

