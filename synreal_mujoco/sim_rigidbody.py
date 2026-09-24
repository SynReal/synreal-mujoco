
import numpy as np
import style3dsim as sim

def _transform_positions(positions, transform):
    q = transform.rotation
    x2 = q.x + q.x
    y2 = q.y + q.y
    z2 = q.z + q.z
    xx = q.x * x2
    xy = q.x * y2
    xz = q.x * z2
    yy = q.y * y2
    yz = q.y * z2
    zz = q.z * z2
    wx = q.w * x2
    wy = q.w * y2
    wz = q.w * z2

    rotation = np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ]
    )
    t = transform.translation
    translation = np.array([t.x, t.y, t.z])
    return positions @ rotation.T + translation

class sim_frozen_cloth:

    def __init__(self, mesh, transform):
        self._positions = np.asarray(mesh.get_positions())
        t = mesh.get_triangles()
        x = _transform_positions(self._positions, transform)
        self.cloth = sim.Cloth( t, x, np.array([], dtype=float), False)
        nVerts = len(mesh.get_positions())
        flag = np.array([True]*nVerts)
        self._indices = np.arange(nVerts)
        self.cloth.set_pin(flag, self._indices) 
        self.cloth_attrib = sim.ClothAttrib()
        self.transform : sim.Transform = transform

    def set_attrib(self, attrib):
        self.cloth_attrib.density = attrib.mass  
        self.cloth_attrib.static_friction = attrib.static_friction  
        self.cloth_attrib.dynamic_friction = attrib.dynamic_friction  
        self.cloth_attrib.thickness = 1e-3    
        self.cloth_attrib.stretch_stiff = sim.Vec3f(1e3,1e3,1e3) # stretching stiffness 
        self.cloth.set_attrib(self.cloth_attrib)


    def attach(self, world):
        self.cloth.attach(world)


    def set_pin(self, is_pinned):
        if not is_pinned:
            print("Warning: sim_frozen_cloth is always pinned. Ignoring set_pin(False).")

    def get_collision_force_piece(self):
        #TODO:
        return []

    def move(self, begin_transform: sim.Transform, end_transform:sim.Transform):
        self.cloth.set_positions(_transform_positions(self._positions, end_transform), self._indices)
        self.transform = end_transform

    def set_collision_group(self,g):
        #TODO:
        #self.cloth.set_collision_group(g)
        pass

    def set_collision_mask(self,m):
        #TODO:
        #self.cloth.set_collision_mask(m)
        pass

    def get_transform(self):
        return self.transform



class sim_rigidbody:

    def __init__(self, mesh, transform, use_frozen_cloth = False):
        self._positions = np.array(mesh.get_positions(), copy=True)
        if use_frozen_cloth:
            self.sim_object = sim_frozen_cloth(mesh, transform)
        else:
            self.sim_object = sim.RigidBody(mesh, transform)


    def set_attrib(self, attrib):
        self.sim_object.set_attrib(attrib)


    def attach(self, world):
        self.sim_object.attach(world)

    def set_pin(self, is_pinned):
        self.sim_object.set_pin(is_pinned)

    def get_collision_force_piece(self):
        return self.sim_object.get_collision_force_piece()

    def move(self, begin_transform, end_transform):
        self.sim_object.move(begin_transform, end_transform)

    def set_collision_group(self,g):
        self.sim_object.set_collision_group(g)

    def set_collision_mask(self,m):
        self.sim_object.set_collision_mask(m)

    def get_transform(self):
        return self.sim_object.get_transform()

    def get_positions(self):
        return _transform_positions(self._positions, self.get_transform())
