# mujoco_style3d
mujoco-style3d coupling solver

# install guide
1. switch to your python virtual env
2. pip install style3dsim*.whl
3. clone style3d forked mujoco repo(https://github.com/Style3D/mujoco) and switch to branch style3d
4. run install_py_package.py in mujoco/python folder to generate mujoco python wheel
5. cd dist && pip install mujoco*.whl
6. cd mujoco_style3d && pip install -e .

# examples
There are several examples in mujoco_style3d/examples folder. 
See mj_py_cloth.py first.
mujoco_style3d is just a wrapper of style3dsim for coupling with mujoco, so users can use style3dsim py api directly to set physical properties instead of setting in mujoco xml.
The c style style3dsim plugin in mujoco is deprecated.

# F&Q
1. run install_py_package.py on win with error
a: Turn on uft-8 support on win first. If use python<=3.11, fix VIRTUAL_ENV in .venv/Scripts/activate manually, for example, change VIRTUAL_ENV from "F:\mujoco\python\.venv" to "/f/mujoco/python/.venv"