
import style3dsim as sim


def get_cloth_property_default():
    cloth_attrib = sim.ClothAttrib()

    cloth_attrib.stretch_stiff = sim.Vec3f(100, 100, 100)
    cloth_attrib.bend_stiff = sim.Vec3f(2e-6, 2e-6, 2e-6)
    cloth_attrib.density = 0.03
    cloth_attrib.static_friction = 0.03
    cloth_attrib.dynamic_friction = 0.03
    cloth_attrib.thickness = 1e-3

    return cloth_attrib

def get_cloth_property_s3d_default():
    cloth_attrib = sim.ClothAttrib()

    cloth_attrib.stretch_stiff = sim.Vec3f(150, 150, 10)
    cloth_attrib.bend_stiff = sim.Vec3f(2e-6, 2e-6, 2e-6)
    cloth_attrib.density = 0.3
    cloth_attrib.static_friction = 0.03
    cloth_attrib.dynamic_friction = 0.03
    cloth_attrib.thickness = 5e-4

    return cloth_attrib

def get_cloth_property_s3d_wool():
    cloth_attrib = sim.ClothAttrib()

    cloth_attrib = sim.ClothAttrib()
    cloth_attrib.stretch_stiff = sim.Vec3f(1000, 1000, 380)
    cloth_attrib.bend_stiff = sim.Vec3f(2.2e-6, 8e-7, 1.4e-6)
    cloth_attrib.density = 0.28
    cloth_attrib.static_friction = 0.03
    cloth_attrib.dynamic_friction = 0.03
    cloth_attrib.thickness = 4.2e-4

    return cloth_attrib

def get_cloth_property_s3d_silk():
    cloth_attrib = sim.ClothAttrib()
    cloth_attrib.stretch_stiff = sim.Vec3f(1000, 1000, 40)
    cloth_attrib.bend_stiff = sim.Vec3f(1.6e-6, 1.5e-6, 1.5e-6)
    cloth_attrib.density = 0.024
    cloth_attrib.static_friction = 0.03
    cloth_attrib.dynamic_friction = 0.03
    cloth_attrib.thickness = 1.1e-4

    return cloth_attrib
