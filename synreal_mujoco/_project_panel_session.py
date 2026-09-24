"""Private panel/viewer coordination; simulation callers use project_panel.launch."""

from queue import Empty
import time

import mujoco.viewer

from synreal_mujoco import project_data_classes as pdc
from synreal_mujoco.s3d_scene_stepper import s3d_scene_stepper


def _build_candidate(data, project_path, build_scene):
    if not isinstance(data, pdc.project_data) or not data.entities:
        raise ValueError('Project must contain entities.')
    if sum(isinstance(entity, pdc.mjcf_scene) for entity in data.entities) != 1:
        raise ValueError('Project must contain exactly one MJCF entity.')
    for entity in data.entities:
        if not isinstance(entity, (pdc.mjcf_scene, pdc.cloth, pdc.deformable_body, pdc.rigid_mesh)):
            raise ValueError('Unsupported project entity.')
        if entity.path is None or not (project_path / entity.path).is_file():
            raise ValueError(f'File not found: {entity.path}')
    return build_scene(data, project_path).build()


def run_session(project_data, project_path, bundle, build_scene, panel):
    m, d, s = bundle
    paused = True
    next_poll = next_status = 0.0
    applied = False
    while True:
        stepper = s3d_scene_stepper(m, d, s)
        replacement = None
        with mujoco.viewer.launch_passive(m, d) as viewer:
            if applied:
                panel.notify('applied', project_data)
            while viewer.is_running():
                now = time.monotonic()
                if now >= next_poll:
                    next_poll = now + 0.05
                    if not panel.process.is_alive():
                        paused = False
                    for _ in range(16):
                        try:
                            command, value = panel.commands.get_nowait()
                        except Empty:
                            break
                        if command == 'pause':
                            paused = bool(value)
                        elif command == 'panel_error':
                            print(f'Project panel unavailable: {value}')
                        elif command == 'apply':
                            try:
                                replacement = _build_candidate(value, project_path, build_scene)
                            except Exception as error:
                                panel.notify('error', str(error))
                            else:
                                project_data = value
                                applied = True
                            break
                if replacement is not None:
                    break
                if now >= next_status:
                    panel.publish(float(d.time), paused)
                    next_status = now + 0.1
                if paused:
                    viewer.sync()
                    time.sleep(0.01)
                    continue
                mujoco.mj_step(m, d)
                stepper.set_rigidbody_pos_mj_2_s3d()
                stepper.step_s3d()
                stepper.set_cloth_pos_s3d_2_mj()
                viewer.sync()
        if replacement is None:
            return project_data
        # Release the old viewer before switching models.
        m, d, s = replacement
