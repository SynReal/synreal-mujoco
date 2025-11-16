
import sys

from numba.cuda.libdevice import sqrtf
from taichi.examples.simulation.physarum import position
from ursina.shaders.screenspace_shaders.fxaa import fxaa_shader

#sys.path.append("../build/lib/Debug")
sys.path.append("../build/lib/RelWithDebInfo")

import time
import numpy as np
import ursina as ua
import style3dsim as sim

from pxr import Usd, UsdGeom
from datetime import datetime
from ursina.shaders import unlit_shader
from shaders.body_shader import body_shader
from shaders.floor_shader import FloorEntity
from shaders.cloth_shader import cloth_shader
from shaders.background_shader import BackgroundEntity
from panda3d.core import loadPrcFileData, AntialiasAttrib


def log_callback(file_name: str, func_name: str, line: int, level: sim.LogLevel, message: str):
	formatted_time = datetime.now().strftime("%H:%M:%S")
	if level == sim.LogLevel.INFO:
		print(f"[{formatted_time}][info]: {message}")
	elif level == sim.LogLevel.ERROR:
		print(f"[{formatted_time}][error]: {message}")
	elif level == sim.LogLevel.WARNING:
		print(f"[{formatted_time}][warning]: {message}")
	elif level == sim.LogLevel.DEBUG:
		print(f"[{formatted_time}][debug]: {message}")

def generate_square_cloth_np(nx: int, ny: int, width: float, height: float):

	x = np.linspace(-width / 2, width / 2, nx)
	y = np.linspace(height, height - width, ny)
	xx, yy = np.meshgrid(x, y, indexing = 'xy')
	pos = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny)])

	# verts = sim.Arr3f()
	# uvcoords = sim.Arr2f()
	# verts.reserve(nx * ny)
	# uvcoords.reserve(nx * ny)

	# for i in range(nx * ny):
	# 	verts.push_back(sim.Vec3f(pos[i][0], pos[i][1], pos[i][2]))
	# 	uvcoords.push_back(sim.Vec2f(pos[i][0], pos[i][1]))

	# faces = sim.Arr3i()
	# faces.reserve(2 * (nx - 1) * (ny - 1))
	faces_list = []
	for i in range(nx - 1):
		for j in range(ny - 1):
			v0 = i * ny + j
			v1 = v0 + ny
			v2 = v0 + 1
			v3 = v2 + ny
			# faces.push_back(sim.Vec3i(v0, v1, v2))
			# faces.push_back(sim.Vec3i(v1, v3, v2))
			faces_list.append([v0, v1, v2])
			faces_list.append([v1, v3, v2])

	faces_np = np.array(faces_list)
	return pos, faces_np, pos[:, 0:2]


########################################################################################################################
#																													   #
#			Ursina Entity Coordinate						Style3D Simulator Coordinate							   #
#																													   #
#					y (up)											y (up)											   #
#					|												|												   #
#					|												|												   #
#	   (forward) z  |												|												   #
#				  \ |												|												   #
#			 	   \|												|												   #
#					*---------- x (right)							*---------- x (right)							   #
#																   /												   #
#																  /													   #
#																 /													   #
#																z (forward)											   #
#																													   #
########################################################################################################################
def CoordAxisEntity(resolution: int = 16, scale: float = 0.2):
	x_axis = ua.Entity(model = ua.Cylinder(resolution = resolution, radius = 0.002, direction = ua.Vec3(scale, 0,  0)), color = ua.color.red, shader = unlit_shader)
	y_axis = ua.Entity(model = ua.Cylinder(resolution = resolution, radius = 0.002, direction = ua.Vec3(0, scale,  0)), color = ua.color.green, shader = unlit_shader)
	z_axis = ua.Entity(model = ua.Cylinder(resolution = resolution, radius = 0.002, direction = ua.Vec3(0, 0, -scale)), color = ua.color.blue, shader = unlit_shader)
	x_arrow = ua.Entity(model = ua.Cone(resolution = resolution, radius = 0.01, height = 0.05), color = ua.color.red, position = (scale, 0, 0), rotation = (0, 0, 90), shader = unlit_shader)
	y_arrow = ua.Entity(model = ua.Cone(resolution = resolution, radius = 0.01, height = 0.05), color = ua.color.green, position = (0, scale, 0), rotation = (0, 0, 0), shader = unlit_shader)
	z_arrow = ua.Entity(model = ua.Cone(resolution = resolution, radius = 0.01, height = 0.05), color = ua.color.blue, position = (0, 0, -scale), rotation = (-90, 0, 0), shader = unlit_shader)


def SetupCamera():
	ua.EditorCamera(rotation_speed = 200, zoom_speed = -2)
	ua.camera.position = ua.Vec3(0, 1, 0)
	ua.camera.clip_plane_near = 2e-2
	ua.camera.clip_plane_far = 1e5
	ua.camera.orthographic = False
	ua.camera.collider = None
	ua.camera.fov = 25


def InitViewerApp():
	loadPrcFileData('', 'framebuffer-multisample 8')
	ua.scene.setAntialias(AntialiasAttrib.MMultisample)
	app = ua.Ursina(title = 'Style3D Simulator', borderless = False)
	FloorEntity(scale = 4, texture_scale = (40, 40))
	CoordAxisEntity(resolution = 16, scale = 0.2)
	BackgroundEntity()
	SetupCamera()
	return app


def LoadUsd(file_path, root_path):
	usd_stage = Usd.Stage.Open(file_path)
	usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath(root_path))
	indices = usd_geom.GetFaceVertexIndicesAttr().Get()
	normals = usd_geom.GetNormalsAttr().Get()
	points = usd_geom.GetPointsAttr().Get()
	return [indices, points, normals]


def CoordinateU2S(coord: ua.Vec3):
	return sim.Vec3f(coord.x, coord.y, -coord.z)


def CoordinateS2U(coord: sim.Vec3f):
	return ua.Vec3(coord.x, coord.y, -coord.z)


if __name__ == "__main__":

	app = InitViewerApp()

	# Set log callback
	sim.set_log_callback(log_callback)

	# Login
	user = input("User Name:")
	password = input("Password:")
	sim.login(user, password, True, None)

	# Create world
	world = sim.World()
	world_attrib = sim.WorldAttrib()
	world_attrib.enable_gpu = True
	world_attrib.gravity = sim.Vec3f(0, -10, 0)
	world.set_attrib(world_attrib)

	# create mesh collider
	[mesh_indices, mesh_points, mesh_normals] = LoadUsd("../Assets/bunny.usd", "/root/bunny")

	mesh_collider_verts = np.array(mesh_points).reshape(-1, 3)
	mesh_collider_verts[:, 2] = -mesh_collider_verts[:, 2]  # convert coordinate

	mesh_collider_tris = np.array(mesh_indices).reshape(-1, 3)
	mesh_collider_tris[:, [1, 2]] = mesh_collider_tris[:, [2, 1]] # the bunny mesh has wrong tri order

	collider_sim_mesh = sim.Mesh(mesh_collider_tris, mesh_collider_verts)

	# mesh_collider = sim.MeshCollider(ff, xx)
	# mesh_collider = sim.MeshCollider(mesh_collider_tris, mesh_collider_verts)
	mesh_collider = sim.MeshCollider(collider_sim_mesh.get_triangles(), collider_sim_mesh.get_positions())
	collider_attrib = sim.ColliderAttrib()
	collider_attrib.collision_gap = 0.005 # unit m
	collider_attrib.dynamic_friction = 0.3
	collider_attrib.static_friction = 0.6
	mesh_collider.set_attrib(collider_attrib)

	mesh_collider.attach(world) # attach to sim world

	collider_render_mesh = ua.Mesh(vertices = mesh_points, triangles = mesh_collider_tris.reshape(-1).tolist())
	collider_render_mesh.generate_normals()
	collider_render_ent = ua.Entity(model = collider_render_mesh, shader = body_shader)
	collider_render_ent.double_sided_setter(True)

	# Create solidified cloth bunny using bunny mesh
	solidified_cloth_bunny_verts = collider_sim_mesh.get_positions()
	solidified_cloth_bunny_verts[:, 1] += 2.0 # move to higher place
	solidified_cloth_bunny_verts[:, 0] += 1.0
	solidified_cloth_bunny_tris = collider_sim_mesh.get_triangles()
	solidified_cloth_bunny = sim.Cloth(tris=solidified_cloth_bunny_tris, verts=solidified_cloth_bunny_verts, keep_wrinkles=True)
	cloth_bunny_attrib = sim.ClothAttrib()
	cloth_bunny_attrib.stretch_stiff = sim.Vec3f(150, 150, 150)
	cloth_bunny_attrib.bend_stiff = sim.Vec3f(1e-5, 1e-5, 1e-5)
	cloth_bunny_attrib.density = 0.2
	cloth_bunny_attrib.thickness = 0.005
	cloth_bunny_attrib.static_friction = 0.03
	cloth_bunny_attrib.dynamic_friction = 0.06
	cloth_bunny_attrib.pressure = 3.0
	solidified_cloth_bunny.set_attrib(cloth_bunny_attrib)
	solidified_cloth_bunny.attach(world)  # attach to sim world
	# solidify all verts
	solidify_stiffs = np.full(solidified_cloth_bunny.get_vert_num(), 0.1, dtype=float)
	solidify_vert_ints = np.arange(0, solidified_cloth_bunny.get_vert_num())
	solidified_cloth_bunny.solidify(world, solidify_stiffs, solidify_vert_ints)

	solidified_cloth_bunny_verts[:, 2] = -solidified_cloth_bunny_verts[:, 2] # convert to ursina cooridinate
	bunny_render_mesh = ua.Mesh(vertices = solidified_cloth_bunny_verts, triangles = solidified_cloth_bunny_tris.reshape(-1).tolist())
	bunny_render_mesh.generate_normals()
	bunny_render_ent = ua.Entity(model = bunny_render_mesh, shader = body_shader)
	bunny_render_ent.double_sided_setter(True)

	# Create square cloth
	[verts, faces, uvcoords] = generate_square_cloth_np(100, 100, 0.5, 1.0)
	# cloth = sim.Cloth(faces, verts, uvcoords)
	cloth = sim.Cloth(faces, verts)
	cloth_attrib = sim.ClothAttrib()
	cloth_attrib.stretch_stiff = sim.Vec3f(120, 100, 80)
	cloth_attrib.bend_stiff = sim.Vec3f(1e-6, 1e-6, 1e-6)
	cloth_attrib.density = 0.2
	cloth_attrib.static_friction = 0.03
	cloth_attrib.dynamic_friction = 0.03
	cloth.set_attrib(cloth_attrib)

	pin_flags = np.empty(2, dtype = np.bool_)
	pin_flags.fill(True)
	pin_vert_indices = np.empty(2, dtype = np.int_)
	pin_vert_indices[0] = 0
	pin_vert_indices[1] = 99
	cloth.set_pin(pin_flags, pin_vert_indices) # pin two verts

	cloth.attach(world) # attach to sim world

	cloths_verts = cloth.get_positions()
	cloths_normals = cloth.get_normals()
	cloths_triangles = cloth.get_triangles()

	pin_vert_p = cloths_verts[pin_vert_indices, :] # store init pin vert positions

	cloths_verts[:, 2] = -cloths_verts[:, 2] # convert coordinate to ursina
	cloths_normals[:, 2] = -cloths_normals[:, 2]
	cloth_render_mesh = ua.Mesh(vertices = cloths_verts, normals = cloths_normals, triangles = cloths_triangles.tolist())
	cloth_render_ent = ua.Entity(model = cloth_render_mesh, shader = cloth_shader, collider = 'mesh')
	cloth_render_ent.double_sided_setter(True)

	def input(key):
		if key == 'escape':
			ua.application.quit()
		elif key == 'space':
			formatted_time = datetime.now().strftime("%H:%M:%S")
			if not world.is_simulating():
				print(f"[{formatted_time}]: Run")
				world.begin_sim_loop()
			else:
				print(f"[{formatted_time}]: Pause")
				world.end_sim_loop()
		elif key == '1':
			# move pinned verts
			pin_vert_p[:, 1] += 0.1
			cloth.set_positions(pin_vert_p, pin_vert_indices)
		#	cloth_entity.collider_setter('mesh')
		#	if ua.mouse.world_point:
		#		ua.Entity(model = 'sphere', scale = 0.005, position = ua.mouse.world_point, color = ua.color.red, shader = unlit_shader)
		elif key == '2':
			# cancel pinned verts
			pin_flags.fill(False)
			cloth.set_pin(pin_flags, pin_vert_indices)
		elif key == '3':
			#move collider.
			mesh_collider_verts[:, 0] += 0.05
			mesh_collider.move_verts(end_positions=mesh_collider_verts)


	def update():
		if world.fetch_sim(-1):
			# update square cloth render
			t0 = time.perf_counter()
			cur_cloths_verts = cloth.get_positions()#world.get_cloth_verts()
			cur_cloths_normals = cloth.get_normals()#world.get_cloth_normals()
			cur_cloths_verts[:, 2] = -cur_cloths_verts[:, 2]
			cur_cloths_normals[:, 2] = -cur_cloths_normals[:, 2]
			cloth_render_mesh.vertices = cur_cloths_verts
			cloth_render_mesh.normals = cur_cloths_normals
			t1 = time.perf_counter()
			cloth_render_mesh.generate()
			t2 = time.perf_counter()
			# print(f"copy time = {1e3 * (t1 - t0):.3f}ms, mesh generation time = {1e3 * (t2 - t1):.3f}ms")

			# update bunny cloth render
			cur_bunny_verts = solidified_cloth_bunny.get_positions()
			cur_bunny_normals = solidified_cloth_bunny.get_normals()
			cur_bunny_verts[:, 2] = -cur_bunny_verts[:, 2]
			cur_bunny_normals[:, 2] = -cur_bunny_normals[:, 2]
			bunny_render_mesh.vertices = cur_bunny_verts
			bunny_render_mesh.normals = cur_bunny_normals
			bunny_render_mesh.generate()

			# update collider render
			cur_collider_verts = np.array(mesh_collider_verts)
			cur_collider_verts[:, 2] = -cur_collider_verts[:, 2]
			collider_render_mesh.vertices = cur_collider_verts
			collider_render_mesh.generate()

	app.run()