import time

import style3dsim as sim
import numpy as np

import mujoco
import mujoco.viewer

import threading

import mujoco_style3d.s3d_mj as s3d_mj

def log_callback(file_name: str, func_name: str, line: int, level: sim.LogLevel, message: str):
	if level == sim.LogLevel.INFO:
		print("[info]: ", message)
	elif level == sim.LogLevel.ERROR:
		print("[error]: ", message)
	elif level == sim.LogLevel.WARNING:
		print("[warning]: ", message)
	elif level == sim.LogLevel.DEBUG:
		print("[debug]: ", message)

#m = mujoco.MjModel.from_xml_path('F:/mujoco_proj/piper_description/mujoco_model/piper_bimanual_description_act_tmp.xml')
m = mujoco.MjModel.from_xml_path('F:/mujoco_proj/4_grid/scene.xml')
d = mujoco.MjData(m)
mujoco.mj_forward(m, d) # so that d is populated by m

x , t = s3d_mj.get_mesh(m,d,'cloth')

sim.set_log_callback(log_callback)

user = 'simsdk001'
password = 'xSXiaCMd'
sim.login(user, password, True, None)

cloth = sim.Cloth(t,x,np.array([],dtype=float),False)

#cloth_attrib = sim.ClothAttrib()
#cloth_attrib.stretch_stiff = sim.Vec3f(120, 100, 80)
#cloth_attrib.bend_stiff = sim.Vec3f(1e-6, 1e-6, 1e-6)
#cloth_attrib.density = 0.2
#cloth_attrib.static_friction = 0.03
#cloth_attrib.dynamic_friction = 0.03
#
#cloth.set_attrib(cloth_attrib)

world = sim.World()
world_attrib = sim.WorldAttrib()
world_attrib.enable_gpu = True
world.set_attrib(world_attrib)

cloth.attach(world)


flush_result_to_ui = False
lock=threading.Lock()

def run_sim():
    duration=0
    while True:
        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        begin0_t = time.time()
        mujoco.mj_step(m, d)
        begin1_t = time.time()
        world.step_sim()
        end1_t = time.time()
        world.fetch_sim(0)
        x = cloth.get_positions()
        end0_t = time.time()
        duration0 = end0_t - begin0_t
        duration1 = end1_t - begin1_t
        duration+=duration0
        if duration>1.0:
            duration-=1.0
            with lock:
                s3d_mj.set_piece_positions(m, d, 'cloth', x)
                global flush_result_to_ui
                flush_result_to_ui = True
        end0_t = time.time()
        duration0 = end0_t - begin0_t
        print("fps = ", 1./duration0, 1./duration1,'duration:',duration)


def update_ui():
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while True:
            with lock:
                global flush_result_to_ui
                if flush_result_to_ui:
                    viewer.sync()
                    flush_result_to_ui = False


t = threading.Thread(target=run_sim)
t.start()

t1=threading.Thread(target=update_ui)
t1.start()

t1.join()
t.join()
