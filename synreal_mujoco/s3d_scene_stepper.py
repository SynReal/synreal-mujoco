
import synreal_mujoco.s3d_mj as s3d_mj
import synreal_mujoco.s3d_scene as s3d_scene

class s3d_scene_stepper:
    def __init__(self,mujoco_model,mujoco_data,scene):
        self.mujoco_model = mujoco_model
        self.mujoco_data = mujoco_data
        self.scene : s3d_scene.s3d_scene = scene

    def set_rigid_body_pos_to_scene(self):
        s3d_mj. set_rigid_body_pos_to_sim(self.mujoco_model, self.mujoco_data, self.scene.rigid_bodies)

    def step_sim(self):
        self.scene. world. step_sim()


    def set_render_pos_to_mujoco(self):
        self.scene.world. fetch_sim(0)

        s3d_mj. set_cloth_pos_to_mujoco(self.mujoco_model, self.mujoco_data, self.scene.sim_cloth, self.scene.cloth_names)

