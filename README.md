# synreal-mujoco
synreal-mujoco coupling solver

# install guide
the main idea of installation is to install style3dsim/forked mujoco/mujoco_sytle3d into one python venv. here is a recommanded steps:
1. clone style3d forked mujoco repo(https://github.com/SynReal/mujoco) and switch to branch style3d
2. create python virtual env and activate it 
3. pip install style3dsim*.whl
4. run install_py_package.py in mujoco/python folder to generate mujoco python wheel (will output to folder dist)
4.1 install the forked mujoco: cd dist && pip install mujoco*.whl
5. clone mujoco_style3d, and cd mujoco_style3d ,and pip install -e . (within the same venv created in forked mujoco folder)


# examples
`synreal_mujoco/project_runner.py` opens a separate project panel generated
directly from the runner's `self.project_data` dataclasses. It uses Tkinter in
its own process, with no additional UI dependency. Entity tabs expose numeric
fields, vector components, boolean checkboxes, and asset file pickers.

Use **File > Open** to rebuild the scene and update `self.project_data`.
This resets simulation time; failed builds keep the current model and scene.
The **File** menu provides **Save** (Ctrl+S), **Save As** (Ctrl+Shift+S), and
**Open** (Ctrl+O). Save writes the current JSON file, initially `main.json`.
Save As chooses a new file and makes it the subsequent Save target; asset
references are adjusted to keep pointing to the same files, without copying
assets. Open automatically loads and restarts the simulation with the selected
JSON project. If building it fails, the current project and simulation stay
active. Opening or closing prompts to save unsaved edits, with a Cancel option.
Saving does not restart the simulation; reopen the saved file to run the edits.
`auto_save.json` remains an output written on launch and successful Apply.

The Pause checkbox pauses both solvers. Closing the panel resumes simulation;
closing the viewer closes the panel. Commands are polled at 20 Hz and small
status updates at 10 Hz. To disable the panel use
`project_runner().run(project_path, show_panel=False)`.
Viewer selection, rigid transforms, and deformable rotation are read-only
because the current runner does not implement these settings.

The panel has one public entry point:

```python
from synreal_mujoco.project_panel import launch

if __name__ == '__main__':
    project_data = launch(project_data, project_path, build_scene=build_scene)
```

`build_scene(data, project_path)` returns a scene builder whose `build()` returns
the MuJoCo model, MuJoCo data, and SynReal scene. `launch` owns the panel process,
viewer, command handling, pause/restart, and cleanup. It blocks until the viewer
closes and returns the last successfully applied dataclass. The caller needs
no queues, timers, or panel shutdown code. `project_runner.py` uses this same
entry point; its loop for running without a panel contains only simulation work.

There are several examples in mujoco_style3d/examples folder. 
See mj_py_cloth.py first.
mujoco_style3d is just a wrapper of style3dsim for coupling with mujoco, so users can use style3dsim py api directly to set physical properties instead of setting in mujoco xml.
The c style style3dsim plugin in mujoco is deprecated.

# F&Q
1. run install_py_package.py on win with error
a: Turn on uft-8 support on win first. If use python<=3.11, fix VIRTUAL_ENV in .venv/Scripts/activate manually, for example, change VIRTUAL_ENV from "F:\mujoco\python\.venv" to "/f/mujoco/python/.venv"
