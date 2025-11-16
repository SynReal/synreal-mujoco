from copy import deepcopy

import numpy as np

import mujoco_style3d.s3d_mj as s3d_mj
import mujoco_style3d._mj_data_helper as s3d_mj_helper
import mujoco
from mujoco import mjx
import xml.etree.ElementTree as ET
import os
import jax
import style3dsim as sim
import jax.numpy as jnp

class mjx_data_manager:

    def __init__(self,xml_path,batch_size):

        self.mj_data = self._load_mj(xml_path, batch_size)

        tree = self._remove_flex(xml_path)
        non_flex_xml = xml_path + '.temp'
        self.mjx_data = self._load_mjx(tree, non_flex_xml, batch_size)

        self.jit_step = jax.jit(jax.vmap(mjx.step, in_axes = ( None, 0 ) ))

        self.sim_world = self._setup_simulation(batch_size)


    def step(self):

        mjx_model, mjx_data = self._get_mjx_data()
        new_mjx_data = self.jit_step(mjx_model, mjx_data)
        self.mjx_data.set_mjx_data(new_mjx_data)

        self._sim_step(self.sim_world)

    def get_mj_data(self,batch_i):
        return self.mj_data.get_model(),self.mj_data.get_data(batch_i)

    def set_rigidbody_pos_to_mujoco(self, batch_i):
        new_mj_data = self.mjx_data.get_data(batch_i)
        delta_pos = _get_transform_pos(batch_i)
        _transform_mj_rigidbody_pos(self.mjx_data.get_model(), new_mj_data, delta_pos)

        mj_data = self.sim_world.mj_datas[batch_i]
        mj_data.geom_xmat = new_mj_data.geom_xmat
        mj_data.geom_xpos = new_mj_data.geom_xpos


    def set_piece_pos_to_mujoco(self, batch_i):
        mj_data = self.sim_world.mj_datas[batch_i]
        sim_pieces = self.sim_world.sim_pieces[batch_i]
        piece_names = self.sim_world.piece_names[batch_i]
        s3d_mj.set_piece_pos_to_mujoco(self.sim_world.mj_model, mj_data, sim_pieces, piece_names)


    def _get_mjx_data(self):
        return self.mjx_data.get_mjx_data()

    def _remove_flex(self, xml_path):
        matching_nodes = []
        tree = ET.parse(xml_path)
        root = tree.getroot()
        _find_xml_nodes_by_name(matching_nodes, root, 'flexcomp')
        for c,p in matching_nodes:
            p.remove(c)
        return tree

    def _load_mj(self, xml_path, batch_size):
        return _mj_data(xml_path, batch_size)

    def _load_mjx(self,tree,new_xml_path,batch_size):
        tree.write(new_xml_path)
        ret = _mjx_data(new_xml_path,batch_size)
        os.remove(new_xml_path)
        return ret

    def _sim_step(self, sim_world):
        # transform_mj_data
        for i in range(len(sim_world.rigid_bodies)):
            rb = sim_world.rigid_bodies[i]
            mj_data = sim_world.mjx_datas[i]
            s3d_mj.set_rigid_body_pos_to_sim(sim_world.mj_model, mj_data,  rb)

        sim_world.world.step_sim()

        sim_world.world.fetch_sim(0)

    def _setup_simulation(self, batch_size):

        sim.login('simsdk001', 'xSXiaCMd', True, None)

        mj_model  = self.mj_data.get_model()

        world = s3d_mj.get_a_sim_world()

        ret = _sim_data(world, [], [], [], mj_model, [], [])

        for i in range(batch_size):
            delta_pos = _get_transform_pos(i)
            #mjx_data = self._get_batch_mjx_data()[i]
            mjx_data = self.mjx_data.get_data(i)
            mj_data = self.mj_data.get_data(i)

            _transform_mj_rigidbody_pos(mj_model, mjx_data, delta_pos)

            sim_pieces, piece_names = s3d_mj.add_piece_to_sim(mj_model, mj_data, world)
            rigid_bodies = s3d_mj.add_rigid_body_to_sim(mj_model, mjx_data, world)

            ret.mj_datas.append(mj_data)
            ret.mjx_datas.append(mjx_data)
            ret.sim_pieces.append(sim_pieces)
            ret.piece_names.append(piece_names)
            ret.rigid_bodies.append(rigid_bodies)

        return ret




def _find_xml_nodes_by_name(matching_nodes,element, target_name ):
    # Recursively search through all child elements
    for child in element:
        if child.tag == target_name:
            matching_nodes.append((child,element))
        _find_xml_nodes_by_name(matching_nodes, child, target_name)



class _sim_data:
    def __init__(self,world,sim_pieces,piece_names,rigid_bodies,mj_model,mj_datas,mjx_datas):
        self.world = world
        self.sim_pieces = sim_pieces
        self.piece_names = piece_names
        self.rigid_bodies = rigid_bodies
        self.mj_model = mj_model
        self.mj_datas = mj_datas
        self.mjx_datas = mjx_datas


class _mjx_data:
    def __init__(self,xml_path,batch_size):

        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_forward(self.mj_model, self.mj_data)  # so that d is populated by m

        self.mjx_model = mjx.put_model(self.mj_model)
        self.mjx_data = mjx.put_data(self.mj_model, self.mj_data)

        rng = jax.random.PRNGKey(0)
        rng = jax.random.split(rng, batch_size)
        
        # self.mjx_data = jax.vmap(lambda rng: self.mjx_data.replace(qpos=jax.random.uniform(rng, (1,))))(rng)
        self.mjx_data = jax.vmap(lambda _: self.mjx_data)(jnp.arange(batch_size))
        #self.mjx_data = jax.vmap(lambda rng: self.mjx_data.replace(qpos=jax.random.uniform(rng, (1,),minval=1,maxval=1 )))(rng)

        #for i in range(batch_size):
        #    delta_pos = _get_transform_pos(i)
        #    mj_data = self.get_data(i)
        #    _transform_mj_rigidbody_pos(self.mj_model, mj_data, delta_pos)

            ## somehow just won't work!
            #= mjx.put_data(self.mj_model, mj_data)
            #self.set_data(i,mj_data)
            #self.mjx_data[i].replace(xmat = mj_data.xmat)
            #self.mjx_data[i].replace(xpos = mj_data.xpos)



    def get_model(self):
        return self.mj_model

    def get_data(self,batch_i):
        return mjx.get_data(self.mj_model, self.mjx_data[batch_i])

    def get_mjx_data(self):
        return self.mjx_model, self.mjx_data

    def set_mjx_data(self , data):
        self.mjx_data = data



class _mj_data:
    def __init__(self, xml_path, batch_size):

        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_forward(self.mj_model, mj_data)  # so that d is populated by m

        self.mj_datas=[]

        for i in range(batch_size):
            d = deepcopy(mj_data)

            delta_pos = _get_transform_pos(i)
            _transform_piece_pos(self.mj_model, d, delta_pos)
            self.mj_datas.append(d)

    def get_model(self ):
        return self.mj_model

    def get_data(self,batch_i ):
        return self.mj_datas[batch_i]

def _get_transform_pos( i):
    offset_length = 1
    return np.array([i * offset_length, 0, 0])

def _do_transform_mj_rigidbody_pos( xpos, delta_pos):
    xpos += delta_pos

def _transform_mj_rigidbody_pos( mj_model, mj_data, delta_pos):
    s3d_mj_helper.for_each_rigid_meshes(mj_model, mj_data, lambda i, x, t, xmat, xpos: _do_transform_mj_rigidbody_pos(xpos,delta_pos) )

def _do_transform_piece_pos(x,delta_pos):
    x[:] += delta_pos

def _transform_piece_pos( m,d,delta_pos):
    s3d_mj_helper.for_each_piece(m,d,lambda x,t,name:_do_transform_piece_pos(x,delta_pos))

