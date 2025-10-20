import style3dsim
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

def _get_pos_buffer(d: mujoco.MjData):
    #return _mj_get_attr(d,  "flex_xpos")
    return _mj_get_attr(d, "flexvert_xpos" )

def _get_vert_num_buffer(m: mujoco.MjModel):
    return _mj_get_attr(m, "flex_vertnum" )

def _get_vert_offset_buffer(m: mujoco.MjModel):
    return _mj_get_attr(m, "flex_vertadr" )

def _get_pos(id,m: mujoco.MjModel, d: mujoco.MjData):
    x = _get_pos_buffer(d);
    offset = _get_vert_offset_buffer(m)
    num = _get_vert_num_buffer(m)
    return x[ offset[id] : offset[id]+num[id] - Flexible.EXTRA_VERTEX_BY_STYLE3D_AT_END] [:]

def _set_pos(id,m: mujoco.MjModel, d: mujoco.MjData,verts):
    x = _get_pos_buffer(d);
    offset = _get_vert_offset_buffer(m)
    num = _get_vert_num_buffer(m)
    x[ offset[id] : offset[id]+num[id] - Flexible.EXTRA_VERTEX_BY_STYLE3D_AT_END] [:] = verts


def _get_tri_buffer(m: mujoco.MjModel):
    return _mj_get_attr(m, "flex_elem")

def _get_tri_offset(m: mujoco.MjModel):
    return _mj_get_attr(m, "flex_elemadr")

def _get_tri_num_buffer(m: mujoco.MjModel):
    return _mj_get_attr(m, "flex_elemnum")

def _get_tri(id,m: mujoco.MjModel):
    offset=_get_tri_offset(m)[id]
    num=_get_tri_num_buffer(m)[id]
    return _get_tri_buffer(m )[offset*3:(offset+num)*3].reshape(-1, 3)


def _set_flex_vertices(m: mujoco.MjModel, d: mujoco.MjData, flex_name: str, verts: np.ndarray) -> None:
    id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_FLEX, flex_name)
    _set_pos(id,m,d, verts)


def for_each_piece(m: mujoco.MjModel,d: mujoco.MjData,fn):
    vert_num = _get_vert_num_buffer(m)

    num_flex = getattr(m, "nflex", 0)

    if num_flex!=len(vert_num):
        return

    for i in range(len(vert_num)):

        x = _get_pos(i,m,d)
        t = _get_tri(i,m)
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_FLEX, i)
        fn(x,t,name)


def set_positions(m: mujoco.MjModel, d: mujoco.MjData,mesh_name,x):
    _set_flex_vertices(m,d,mesh_name,x)

