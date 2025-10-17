import os
from pathlib import Path
os.chdir(Path(__file__).parent)

import mujoco
import mujoco.viewer
import numpy as np
from typing import Union

# --- Extracted functions from MjSim ---

FLEX_CUSTOM_BITS = {
    "Default": 1 << 0,
    "SkipSim": 1 << 1,
    "ResetPos": 1 << 2,
    "ClearPin": 1 << 3,
    "RestorePin": 1 << 4,
}

class Flexible:
    # This is a dummy class to hold constants from gx_utils.dtype.Flexible
    EXTRA_VERTEX_BY_STYLE3D_AT_END = 1

def _mj_get_attr(obj, *names):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    raise AttributeError(f"None of attributes present: {names}")

def _mj_name2flex_id(m: mujoco.MjModel, target: str | None) -> int | None:
    num_flex = getattr(m, "nflex", 0)
    if num_flex == 0:
        return None
    if target is None:
        return 0
    try:
        for i in range(num_flex):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_FLEX, i)
            if target == name:
                return i
    except Exception:
        pass
    return None

def get_flex_topology(m: mujoco.MjModel, flex_name: str):
    """Return faces and vertex window for a given flexible object.

    Returns dict with keys: faces (Nx3 int32), vert_adr (int), vert_num (int).
    Accounts for style3d's extra vertex at end by subtracting 1 from vert_num.
    """
    flex_id = _mj_name2flex_id(m, flex_name)
    assert flex_id is not None, f"Flex component for {flex_name} not found"

    flex_vertadr = _mj_get_attr(m, "flex_vertadr", "flexvertadr")
    flex_vertnum = _mj_get_attr(m, "flex_vertnum", "flexvertnum")
    elem_adr = _mj_get_attr(m, "flex_elemadr")
    elem_num = _mj_get_attr(m, "flex_elemnum")
    flex_elem = _mj_get_attr(m, "flex_elem")

    vadr = int(flex_vertadr[flex_id])
    vnum = int(flex_vertnum[flex_id]) - Flexible.EXTRA_VERTEX_BY_STYLE3D_AT_END
    eadr = int(elem_adr[flex_id])
    enum = int(elem_num[flex_id])

    faces = []
    if enum > 0:
        elem_data = np.array(flex_elem[eadr:eadr + enum * 3], dtype=np.int64)
        tris = elem_data.reshape(-1, 3)
        faces = tris.tolist()
    else:
        for i in range(0, max(0, vnum - 2), 3):
            if i + 2 < vnum:
                faces.append([i, i + 1, i + 2])
    faces = np.asarray(faces, dtype=np.int64)
    return {
        "faces": faces,
        "vert_adr": vadr,
        "vert_num": vnum,
    }

def get_flex_vertices(m: mujoco.MjModel, d: mujoco.MjData, flex_name: str) -> np.ndarray:
    """Return current world-space vertex positions (N,3) for a flexible object."""
    flex_id = _mj_name2flex_id(m, flex_name)
    assert flex_id is not None, f"Flex component for {flex_name} not found"
    
    flex_vertadr = _mj_get_attr(m, "flex_vertadr", "flexvertadr")
    flex_vertnum = _mj_get_attr(m, "flex_vertnum", "flexvertnum")
    vadr = int(flex_vertadr[flex_id])
    vnum = int(flex_vertnum[flex_id]) - 1  # style3d extra vertex at end
    xpos = _mj_get_attr(d, "flexvert_xpos", "flex_xpos")
    verts = np.asarray(xpos, dtype=np.float32)[vadr:vadr + vnum].reshape(-1, 3)
    return verts[:]

def set_flex_flag(m: mujoco.MjModel, flag: Union[int, str], enable: bool = True) -> bool | None:
    """Enable or disable a custom flex flag; returns previous state (True/False) or None if unsupported."""
    if isinstance(flag, str):
        if flag not in FLEX_CUSTOM_BITS:
            raise ValueError(f"Unknown flex flag '{flag}'")
        bit = FLEX_CUSTOM_BITS[flag]
    else:
        bit = int(flag)
    if bit <= 0:
        raise ValueError(f"Flex flag must be positive non-zero bit, got {bit}")

    prev_mask = int(m.flex_custom)
    was_enabled = bool(prev_mask & bit)

    if enable:
        m.flex_custom = prev_mask | bit
    else:
        m.flex_custom = prev_mask & ~bit

    return was_enabled

def set_flex_vertices(m: mujoco.MjModel, d: mujoco.MjData, flex_name: str, verts: np.ndarray) -> None:
    """Overwrite world-space vertices for the specified flexible object."""
    flex_id = _mj_name2flex_id(m, flex_name)
    assert flex_id is not None, f"Flex component for {flex_name} not found in model"

    verts = np.asarray(verts, dtype=np.float32)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"Expected verts with shape (N,3), got {verts.shape}")

    flex_vertadr = _mj_get_attr(m, "flex_vertadr", "flexvertadr")
    flex_vertnum = _mj_get_attr(m, "flex_vertnum", "flexvertnum")
    vadr = int(flex_vertadr[flex_id])
    total_vnum = int(flex_vertnum[flex_id])
    usable_vnum = total_vnum - Flexible.EXTRA_VERTEX_BY_STYLE3D_AT_END
    if usable_vnum <= 0:
        raise ValueError(f"Flexible component {flex_name} has no usable vertices")

    if verts.shape[0] == usable_vnum:
        sim_verts = verts
    elif verts.shape[0] == total_vnum:
        sim_verts = verts[:usable_vnum]
    else:
        raise ValueError(
            f"verts count {verts.shape[0]} does not match flex vertnum {total_vnum} "
            f"or usable count {usable_vnum}"
        )

    d_flex_xpos = _mj_get_attr(d, "flexvert_xpos", "flex_xpos")
    m_flex_xpos = _mj_get_attr(m, "flex_vert")
    flat_start = vadr * 3
    flat_count = usable_vnum * 3
    row_start = vadr
    row_count = usable_vnum

    d_flex_xpos[vadr:usable_vnum][:] = sim_verts
    #d_flex_xpos[flat_start:flat_start + flat_count][:-Flexible.EXTRA_VERTEX_BY_STYLE3D_AT_END] = sim_verts
    # m_flex_xpos[row_start:row_start + row_count, :] = sim_verts

# --- Original test.py content ---

#m = mujoco.MjModel.from_xml_path("ufactory_xarm7_cloth/scene.xml")
#m = mujoco.MjModel.from_xml_path('F:/mujoco_proj/based_on_ufactory_xarm7.xml')
m = mujoco.MjModel.from_xml_path('F:/mujoco_proj/4_grid/scene.xml')
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)

viewer = mujoco.viewer.launch_passive(m, d)

flex = _mj_name2flex_id(m, "cloth")
print("flex id:", flex)
topo = get_flex_topology(m, "cloth")
print("topo:", topo)
v_init = get_flex_vertices(m, d, "cloth")
print("verts:", v_init)


for i in range(500):
    mujoco.mj_step(m, d)
    x=get_flex_vertices(m,d,'cloth')
    x[:][:]+=1e-2*np.array([np.sin(1e-1*i),0,0])
    set_flex_vertices(m,d,'cloth',x)
    viewer.sync()

try:
    while True:
        set_flex_vertices(m, d, "cloth", v_init)
        #set_flex_flag(m, "ResetPos", True)
        mujoco.mj_step(m, d)
        # viewer.sync()
        # mujoco.mj_forward(m, d)
        #set_flex_flag(m, "ResetPos", False)
        print("reset verts")

        for _ in range(10):
            mujoco.mj_step(m, d)
            viewer.sync()
finally:
    viewer.close()

