
from mujoco import mjx
import mujoco.viewer
import jax


xml = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <body name="box_and_sphere" euler="0 0 -30">
      <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
      <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
      <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""
mj_model = mujoco.MjModel.from_xml_string(xml)
mj_data = mujoco.MjData(mj_model)


mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)

print(mj_data.qpos, type(mj_data.qpos))
print(mjx_data.qpos, type(mjx_data.qpos), mjx_data.qpos.devices())

# enable joint visualization option:
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True



batch_size=4
rng = jax.random.PRNGKey(0)
rng = jax.random.split(rng, batch_size)
mjx_data = jax.vmap(lambda rng: mjx_data.replace(qpos=jax.random.uniform(rng, (1,))))(rng)
jit_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))

print(mjx_data.qpos)

for di in range(batch_size):
    print(f" batch {di}")
    mj_data = mjx.get_data(mj_model, mjx_data)[di]
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        fi = 0
        while viewer.is_running() and fi<500 :

            mjx_data = jit_step(mjx_model, mjx_data)
            new_mj_data = mjx.get_data( mj_model, mjx_data)[di]

            mj_data.geom_xmat = new_mj_data.geom_xmat
            mj_data.geom_xpos = new_mj_data.geom_xpos

            print(f"frame {fi} ")
            viewer.sync()
            fi+=1
