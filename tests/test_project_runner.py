from pathlib import Path
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

from synreal_mujoco import project_data_classes as pdc
from synreal_mujoco.project_io import project_io
from synreal_mujoco.project_runner import project_runner
from synreal_mujoco.s3d_scene_builder import s3d_scene_builder


class ProjectRunnerClothTests(unittest.TestCase):
    def test_piper_cloth_project_configuration(self):
        project_dir = Path(__file__).resolve().parents[1] / 'synreal_mujoco/projects/piper_cloth'
        loader = project_io()
        loader.read(project_dir)
        with patch.object(project_runner, '_login_sim'):
            scene = project_runner()._config_project(loader.get_data(), project_dir)

        self.assertTrue(scene.mjcf_file.is_file())
        self.assertEqual(scene.cloth_files, [project_dir / 'assets/50k_plane.obj'])
        self.assertTrue(scene.cloth_files[0].is_file())
        cloth = scene.cloth_builder_map['50k_plane']
        np.testing.assert_allclose(cloth.translate, [-0.8, -2.0, 0.15])
        np.testing.assert_allclose(cloth.quat, [1, 0, 0, 0])
        np.testing.assert_allclose(
            [cloth.attrib.bend_stiff.x, cloth.attrib.bend_stiff.y, cloth.attrib.bend_stiff.z],
            [1e-6, 1e-6, 1e-6],
        )
        self.assertAlmostEqual(cloth.attrib.density, 0.1)
        self.assertAlmostEqual(cloth.attrib.stretch_stiff.x, 150)

    def test_cloth_material_overrides_and_instance_isolation(self):
        scene = s3d_scene_builder()
        runner = project_runner()
        first = pdc.cloth(path=Path('first.obj'), attrib=pdc.cloth_attrib(
            stretch_stiffness=np.array([10., 20., 30.]),
            bend_stiffness=np.array([1e-5, 2e-5, 3e-5]),
            thickness=0.002, density=0.4, pressure=2., static_friction=0.2,
            dynamic_friction=0.1, yield_curvature=12., volume_conserve_strength=80., frozen=True,
        ))
        runner._dispatch_entity(first, scene)
        runner._dispatch_entity(pdc.cloth(path=Path('second.obj')), scene)
        attrib = scene.cloth_builder_map['first'].attrib
        np.testing.assert_allclose([attrib.stretch_stiff.x, attrib.stretch_stiff.y, attrib.stretch_stiff.z], [10, 20, 30])
        np.testing.assert_allclose([attrib.bend_stiff.x, attrib.bend_stiff.y, attrib.bend_stiff.z], [1e-5, 2e-5, 3e-5])
        for name, expected in {
            'thickness': 0.002, 'density': 0.4, 'pressure': 2.,
            'static_friction': 0.2, 'dynamic_friction': 0.1,
            'yield_curvature': 12., 'volume_conserve_strength': 80.,
        }.items():
            with self.subTest(property=name):
                self.assertAlmostEqual(getattr(attrib, name), expected)
        self.assertTrue(attrib.frozen)
        self.assertIsNot(attrib, scene.cloth_builder_map['second'].attrib)
        self.assertFalse(scene.cloth_builder_map['second'].attrib.frozen)
        self.assertAlmostEqual(scene.cloth_builder_map['second'].attrib.density, 0.1)

    def test_cloth_transform_reaches_mjcf(self):
        entity = pdc.cloth(path=Path('rotated.obj'), trans=pdc.transform(
            translation=np.array([1., 2., 3.]), euler_xyz=np.array([0., 0., np.pi / 2]),
        ))
        scene = s3d_scene_builder()
        project_runner()._dispatch_entity(entity, scene)
        tree = ET.ElementTree(ET.fromstring('<mujoco><worldbody/></mujoco>'))
        scene._add_flex_cloth(tree)
        flex = tree.find('worldbody/flexcomp')
        np.testing.assert_allclose(np.fromstring(flex.get('pos'), sep=' '), [1, 2, 3])
        np.testing.assert_allclose(np.fromstring(flex.get('quat'), sep=' '), [np.sqrt(0.5), 0, 0, np.sqrt(0.5)])
        self.assertEqual(Path(flex.get('file')), entity.path.resolve())


if __name__ == '__main__':
    unittest.main()
