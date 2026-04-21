
import mujoco
import synreal_mujoco.s3d_mj as s3d_mj
import synreal_mujoco._mj_data_helper as _mj_data_helper
import synreal_mujoco.cloth_property as cloth_property
import numpy as np
from pathlib import Path

import mujoco.viewer

class s3d_mj_mapper:

    def __init__(self,world,sim_cloth,cloth_names, rigid_bodies, rigid_body_id,collision_force):
        self.world = world
        self.sim_cloth = sim_cloth
        self.cloth_names = cloth_names
        self.rigid_bodies = rigid_bodies
        self.rigid_body_id = rigid_body_id
        self.collision_force = collision_force



def _get_geom_parent( m: mujoco.MjModel, d: mujoco.MjData ):
    rigid_body_id=[]

    def collect_rg_id(slot_i,geom_id, mesh_id, rb_id,geom_type):
        rigid_body_id.append(rb_id)

    _mj_data_helper.for_each_geom_mesh(m,d,collect_rg_id)

    return  rigid_body_id


#### load
# kwargs: rb_property_fn cloth_property_fn
#   rb_property_fn(rigidbody_name, rigidbody_attrib)
#   cloth_property_fn(cloth_name, cloth_attrib)
def smj_load_data(xml_path, **kwargs):

    if 'rb_property_fn' in kwargs:
        rb_property_fn = kwargs['rb_property_fn']
    else:
        rb_property_fn = lambda name, attrib : cloth_property.set_rigid_body_property_default(attrib)

    if 'cloth_property_fn' in kwargs:
        cloth_property_fn = kwargs['cloth_property_fn']
    else:
        cloth_property_fn = lambda name, attrib : cloth_property.set_cloth_property_default(attrib)


    if 'rigidbody_with_convex_hull' in kwargs:
        rigidbody_with_convex_hull = kwargs['rigidbody_with_convex_hull']
    else:
        rigidbody_with_convex_hull = False

    script_dir = Path(__file__).parent.resolve()

    s3d_mj. log_in_simulation( login_file=f'{script_dir}/../simulation_login.json')  # this line is optional, but a login prompt will pop up latter

    m, d = s3d_mj. load_data(xml_path)

    world = s3d_mj. get_a_sim_world(m)

    sim_cloth, cloth_names = s3d_mj. add_cloth_to_sim(m, d, world, cloth_property_fn)
    rigid_bodies = s3d_mj. add_rigid_body_to_sim(m, d, world , rb_property_fn, rigidbody_with_convex_hull)

    rigid_body_id = _get_geom_parent(m, d)

    collision_force = []

    mp = s3d_mj_mapper (
        world,
        sim_cloth,
        cloth_names,
        rigid_bodies,
        rigid_body_id,
        collision_force
    )

    return m, d, mp



# interaction
def update_rigidbody_to_cloth(m,d,mp):
    s3d_mj.set_rigid_body_pos_to_sim(m, d, mp.rigid_bodies)


def update_rigidbody_cloth_collision_force(m, d, mp):
    mp.collision_force = []
    for rb in mp.rigid_bodies:
        f_rb = s3d_mj.get_collision_force_from_piece(rb)
        mp.collision_force.append(f_rb)


def update_cloth_to_rigid_body(m,d,mp):

    wait_second = 0

    mp.world.fetch_sim(wait_second)

    s3d_mj.set_cloth_pos_to_mujoco(m, d, mp.sim_cloth, mp.cloth_names)


def apply_collision_force_to_rigidbody(m,d,mp):

    for i in range(len(mp.rigid_bodies)):
        l_rb_id = mp.rigid_body_id[i]

        if len(mp.collision_force) <= 0:
            continue

        rb_force = mp.collision_force[i]

        orientation = d.xmat[l_rb_id]
        orientation = orientation.reshape(3, 3)

        wrench = np.zeros(6)
        for f, bary in zip(*rb_force):  # force and bary

            r = orientation @ bary
            torque = np.cross(r, f)

            # append force and torque to rigid body
            wrench += [f[0], f[1], f[2], torque[0], torque[1], torque[2]]

        ### maybe clamp wrench
        #wrench_norm = np.linalg.norm(wrench)
        #wrench_threshold = 1e3
        #if wrench_norm > wrench_threshold:
        #    wrench *= wrench_threshold/wrench_norm

        d.xfrc_applied[l_rb_id] += wrench

## step
def smj_cloth_step(mp):
    mp.world.step_sim()

def smj_rigid_body_step(m,d):
    mujoco.mj_step(m, d)

## control
def  set_mocap_pos( m: mujoco.MjModel ,d: mujoco.MjData, body_name, pos):
    body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
    mocap_id = m. body_mocapid[body_id]
    d. mocap_pos[mocap_id] = pos

def  set_actuator_target_pos( m: mujoco.MjModel ,d: mujoco.MjData, actuator_name, target_pos):
    id = m. actuator(actuator_name).id
    d. ctrl[id] = target_pos

