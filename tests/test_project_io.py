import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
import numpy as np

if __name__ == '__main__' and not __package__:
    # Allow direct execution without installing the project package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synreal_mujoco.project_io import (
    dc,
    project_io,
)


class ProjectIOTests(unittest.TestCase):
    def test_read_sample_project(self):
        project_dir = (
            Path(__file__).resolve().parents[1]
            / 'synreal_mujoco/projects/deformable_body'
        )
        loader = project_io()
        self.assertIsInstance(loader.project_data, dc.project_data)
        loader.read(project_dir)
        result = loader.get_data()

        self.assertIs(result, loader.project_data)
        self.assertIsInstance(result, dc.project_data)
        self.assertEqual(len(result.entities), 2)
        rigid, deformable = result.entities
        self.assertIsInstance(rigid, dc.mjcf_rigidbody)
        self.assertIsInstance(deformable, dc.deformable_body)
        self.assertEqual(rigid.path, Path('../../examples/xml_projects/piper_secription/piper_description.xml'))
        self.assertEqual(deformable.path, Path('../../examples/xml_projects/piper_secription/tets1.vtk'))
        self.assertEqual(rigid.option, {})
        self.assertIsInstance(deformable.attrib, dc.deformable_body_attrib)
        self.assertEqual(deformable.attrib.density, 0.3)
        self.assertEqual(deformable.attrib.youngsModulus, 1e6)
        self.assertIsNone(result.viewer)

    def test_read_preserves_nested_values(self):
        options = {
            'enabled': False,
            'count': 0,
            'values': [1, 'text', None, [], {'nested': [True, 2.5]}],
        }
        source = {'project_data': {
            'viewer': 'passive',
            'entities': [{'mjcf_rigidbody': {'path': None, 'option': options}}],
        }}
        with TemporaryDirectory() as directory:
            project_dir = Path(directory)
            (project_dir / 'main.json').write_text(json.dumps(source), encoding='utf-8')
            loader = project_io()
            loader.read(project_dir)

        result = loader.get_data()
        self.assertEqual(result.viewer, 'passive')
        self.assertIsNone(result.entities[0].path)
        self.assertEqual(result.entities[0].option, options)

    def test_invalid_root_does_not_replace_project_data(self):
        loader = project_io()
        initial = loader.project_data
        with TemporaryDirectory() as directory:
            project_dir = Path(directory)
            (project_dir / 'main.json').write_text('{"mjcf_rigidbody": {}}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'project_data object'):
                loader.read(project_dir)
        self.assertIs(loader.project_data, initial)
        self.assertIsInstance(loader.project_data, dc.project_data)

    def test_empty_entities_and_default_fields(self):
        result = project_io._undump_json(None, {'project_data': {'entities': []}})
        self.assertIsInstance(result, dc.project_data)
        self.assertEqual(result.entities, [])
        self.assertIsNone(result.viewer)
        default = project_io._undump_json(None, {'mjcf_rigidbody': {}})
        self.assertIsInstance(default, dc.mjcf_rigidbody)
        self.assertIsNone(default.path)
        self.assertIsNone(default.trans)

    def test_save_and_reload_numpy_vectors_and_nested_attributes(self):
        loader = project_io()
        loader.project_data.entities = [dc.deformable_body()]
        body = loader.project_data.entities[0]
        body.trans.translation = np.array([1.5, -2.0, 3.0])
        body.attrib.density = 123.0
        with TemporaryDirectory() as directory:
            project_dir = Path(directory)
            loader.save(project_dir / 'main.json')
            restored = project_io()
            restored.read(project_dir)

        body = restored.get_data().entities[0]
        self.assertIsInstance(body.attrib, dc.deformable_body_attrib)
        self.assertEqual(body.attrib.density, 123.0)
        self.assertIsInstance(body.trans, dc.transform)
        self.assertIsInstance(body.trans.translation, np.ndarray)
        np.testing.assert_array_equal(body.trans.translation, [1.5, -2.0, 3.0])

    def test_cloth_attributes_json_round_trip(self):
        original = dc.cloth_attrib()
        encoded = json.loads(json.dumps(project_io._dump_json(original)))
        restored = project_io._undump_json(None, encoded)
        self.assertIsInstance(restored, dc.cloth_attrib)
        self.assertIsInstance(restored.stretchStiffness, np.ndarray)
        np.testing.assert_array_equal(restored.stretchStiffness, original.stretchStiffness)
        np.testing.assert_array_equal(restored.bendStiffness, original.bendStiffness)


if __name__ == '__main__':
    unittest.main()
