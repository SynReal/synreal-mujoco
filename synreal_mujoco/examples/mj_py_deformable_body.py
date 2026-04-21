import mujoco.viewer

import synreal_mujoco.s3d_mj as s3d_mj
from synreal_mujoco import cloth_property

s3d_mj.log_in_simulation(login_file='../../simulation_login.json') # this line is optional, but a login prompt will pop up latter

m , d = s3d_mj.load_data('xml_projects/piper_secription/piper_description.xml')

tet_file = 'xml_projects/piper_secription/tets.vtk'
pos, tets = s3d_mj.load_tetrahedrons(tet_file)

world = s3d_mj.get_a_sim_world(m)

rigid_bodies = s3d_mj.add_rigid_body_to_sim(m, d, world, lambda name,attrib: cloth_property.set_rigid_body_property_default(attrib),False)

