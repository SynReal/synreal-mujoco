from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import MagicMock, patch

from synreal_mujoco import project_data_classes as pdc
from synreal_mujoco.project_io import project_io
from synreal_mujoco.project_panel import _ProjectEditor, launch, save_project_data
from synreal_mujoco._project_panel_session import run_session
from synreal_mujoco.project_runner import project_runner


class ProjectPanelTests(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        mesh = self.directory / 'cloth.obj'
        mesh.touch()
        # No JSON exists. Controls must be generated from the object passed in.
        self.data = pdc.project_data(entities=[pdc.cloth(path=mesh)])
        self.commands, self.events, self.status = Queue(), Queue(), Queue()
        self.editor = _ProjectEditor(self.root, self.data, self.directory,
                                    self.commands, self.events, self.status, Event())

    def variable(self, *path, component=0):
        return self.editor.bindings[('entities', 0) + path][0][component]

    def test_typed_controls_collect_independent_dataclass(self):
        self.variable('attrib', 'density').set('0.25')
        self.variable('attrib', 'bend_stiffness', component=1).set('2e-7')
        self.variable('attrib', 'frozen').set(True)
        self.variable('trans', 'euler_xyz').set('0.125')
        result = self.editor.collect()
        self.assertIsInstance(result, pdc.project_data)
        self.assertIsInstance(result.entities[0].path, Path)
        self.assertIsInstance(result.entities[0].attrib, pdc.cloth_attrib)
        self.assertEqual(result.entities[0].attrib.density, 0.25)
        self.assertTrue(result.entities[0].attrib.frozen)
        self.assertEqual(result.entities[0].attrib.bend_stiffness[1], 2e-7)
        self.assertEqual(result.entities[0].trans.euler_xyz[0], 0.125)
        self.assertEqual(self.data.entities[0].attrib.density, 0.1)
        self.assertEqual(self.data.entities[0].trans.euler_xyz[0], 0)

    def test_apply_sends_dataclass_and_acknowledges(self):
        self.variable('attrib', 'density').set('0.2')
        self.editor.apply()
        command, candidate = self.commands.get_nowait()
        self.assertEqual(command, 'apply')
        self.assertIsInstance(candidate, pdc.project_data)
        self.assertTrue(self.editor.pending)
        self.events.put(('applied', candidate))
        self.editor.poll()
        self.assertFalse(self.editor.pending)
        self.assertEqual(self.editor.project_data.entities[0].attrib.density, 0.2)
        self.assertTrue((self.directory / 'auto_save.json').is_file())
        self.variable('attrib', 'density').set('0.9')
        self.editor.revert()
        self.assertEqual(float(self.variable('attrib', 'density').get()), 0.2)

    def test_invalid_field_blocks_apply_and_save(self):
        self.variable('attrib', 'density').set('nan')
        with patch('tkinter.messagebox.showerror') as error:
            self.editor.apply()
            self.assertFalse(self.editor.save())
        self.assertEqual(error.call_count, 2)
        self.assertTrue(self.commands.empty())
        self.assertFalse((self.directory / 'main.json').exists())

    def test_save_collects_unfocused_edits(self):
        self.variable('attrib', 'density').set('0.5')
        self.assertTrue(self.editor.save())
        loader = project_io()
        loader.read(self.directory)
        self.assertEqual(loader.get_data().entities[0].attrib.density, 0.5)
        self.assertEqual(self.editor.project_data.entities[0].attrib.density, 0.1)

    def test_pause_status_and_error(self):
        self.editor.paused.set(True)
        self.editor.pause()
        self.assertEqual(self.commands.get_nowait(), ('pause', True))
        self.status.put((1.25, True))
        self.editor.poll()
        self.assertEqual(self.editor.clock.get(), '1.250 s')
        self.editor.apply()
        self.events.put(('error', 'Bad mesh'))
        self.editor.poll()
        self.assertFalse(self.editor.pending)
        self.assertIn('Bad mesh', self.editor.message.get())

    def test_file_menu_and_pending_shortcuts(self):
        self.assertEqual([self.editor.file_menu.entrycget(i, 'label') for i in range(3)],
                         ['Save', 'Save As...', 'Open...'])
        self.editor.set_pending(True)
        callback = MagicMock()
        self.editor._file_shortcut(callback)
        callback.assert_not_called()
        for index in range(3):
            self.assertEqual(self.editor.file_menu.entrycget(index, 'state'), 'disabled')

    def test_save_as_preserves_asset_references_and_changes_save_target(self):
        self.editor.project_data.entities[0].path = Path('cloth.obj')
        self.editor.build_tabs()
        other = self.directory / 'other'
        other.mkdir()
        target = other / 'copy.json'
        self.variable('attrib', 'density').set('0.3')
        with patch('tkinter.filedialog.asksaveasfilename', return_value=str(target)):
            self.assertTrue(self.editor.save_as())
        saved = project_io._load_project_data(target)
        self.assertEqual((target.parent / saved.entities[0].path).resolve(), self.directory / 'cloth.obj')
        self.assertEqual(saved.entities[0].attrib.density, 0.3)
        self.assertEqual(self.editor.project_file, target)
        self.assertFalse((self.directory / 'main.json').exists())
        self.variable('attrib', 'density').set('0.7')
        self.assertTrue(self.editor.save())
        self.assertEqual(project_io._load_project_data(target).entities[0].attrib.density, 0.7)

    def test_open_automatically_applies_and_rebases_assets(self):
        other = self.directory / 'other'
        other.mkdir()
        (other / 'new.obj').touch()
        target = other / 'new.json'
        data = pdc.project_data(entities=[pdc.cloth(path=Path('new.obj'))])
        data.entities[0].attrib.density = 0.6
        save_project_data(target, data)
        with patch('tkinter.filedialog.askopenfilename', return_value=str(target)):
            self.assertTrue(self.editor.open_project())
        self.assertTrue(self.editor.pending)
        self.assertEqual(self.editor.project_file.name, 'main.json')
        command, candidate = self.commands.get_nowait()
        self.assertEqual(command, 'apply')
        self.assertEqual((self.directory / candidate.entities[0].path).resolve(), other / 'new.obj')
        self.events.put(('applied', candidate))
        self.editor.poll()
        self.assertFalse(self.editor.pending)
        self.assertEqual(self.editor.project_file, target)
        self.assertFalse(self.editor.dirty)
        self.assertTrue(self.commands.empty())
        collected = self.editor.collect()
        self.assertEqual((self.directory / collected.entities[0].path).resolve(), other / 'new.obj')
        self.assertEqual(float(self.variable('attrib', 'density').get()), 0.6)
        self.assertTrue(self.editor.save())
        self.assertEqual(project_io._load_project_data(target).entities[0].path, Path('new.obj'))

    def test_open_cancel_preserves_unsaved_edits(self):
        target = self.directory / 'new.json'
        save_project_data(target, self.data)
        self.variable('attrib', 'density').set('0.9')
        with patch('tkinter.filedialog.askopenfilename', return_value=str(target)), \
                patch('tkinter.messagebox.askyesnocancel', return_value=None):
            self.assertFalse(self.editor.open_project())
        self.assertEqual(self.variable('attrib', 'density').get(), '0.9')
        self.assertEqual(self.editor.project_file.name, 'main.json')
        self.assertTrue(self.editor.dirty)

    def test_open_saves_unsaved_edits_before_switching(self):
        target = self.directory / 'new.json'
        save_project_data(target, self.data)
        self.variable('attrib', 'density').set('0.9')
        with patch('tkinter.filedialog.askopenfilename', return_value=str(target)), \
                patch('tkinter.messagebox.askyesnocancel', return_value=True):
            self.assertTrue(self.editor.open_project())
        saved = project_io._load_project_data(self.directory / 'main.json')
        self.assertEqual(saved.entities[0].attrib.density, 0.9)
        command, candidate = self.commands.get_nowait()
        self.events.put(('applied', candidate))
        self.editor.poll()
        self.assertEqual(float(self.variable('attrib', 'density').get()), 0.1)

    def test_failed_open_build_preserves_current_project_and_edits(self):
        target = self.directory / 'new.json'
        save_project_data(target, self.data)
        original_data = self.editor.project_data
        original_file = self.editor.project_file
        self.variable('attrib', 'density').set('0.9')
        with patch('tkinter.filedialog.askopenfilename', return_value=str(target)), \
                patch('tkinter.messagebox.askyesnocancel', return_value=False):
            self.assertTrue(self.editor.open_project())
        self.events.put(('error', 'Could not build mesh'))
        self.editor.poll()
        self.assertIs(self.editor.project_data, original_data)
        self.assertEqual(self.editor.project_file, original_file)
        self.assertEqual(self.variable('attrib', 'density').get(), '0.9')
        self.assertTrue(self.editor.dirty)
        self.assertFalse(self.editor.pending)
        self.assertIsNone(self.editor._pending_open)
        self.assertIn('Open failed', self.editor.message.get())

    def test_open_queue_failure_does_not_switch_project(self):
        target = self.directory / 'new.json'
        save_project_data(target, self.data)
        original = self.editor.project_data
        with patch('tkinter.filedialog.askopenfilename', return_value=str(target)), \
                patch.object(self.editor, 'send', return_value=False):
            self.assertFalse(self.editor.open_project())
        self.assertIs(self.editor.project_data, original)
        self.assertEqual(self.editor.project_file.name, 'main.json')
        self.assertFalse(self.editor.pending)
        self.assertIsNone(self.editor._pending_open)

    def test_invalid_open_preserves_editor(self):
        target = self.directory / 'invalid.json'
        original = self.editor.project_data
        for content in ['{', '{"project_data": {"entities": [{}]}}',
                        '{"project_data": {"entities": [{"cloth": {"path": "cloth.obj", "attrib": "bad"}}]}}']:
            target.write_text(content)
            with patch('tkinter.filedialog.askopenfilename', return_value=str(target)), \
                    patch('tkinter.messagebox.showerror') as error:
                self.assertFalse(self.editor.open_project())
            error.assert_called_once()
            self.assertIs(self.editor.project_data, original)

    def test_save_as_cancel_or_failure_preserves_target_and_dirty_state(self):
        original = self.editor.project_file
        self.variable('attrib', 'density').set('0.8')
        with patch('tkinter.filedialog.asksaveasfilename', return_value=''):
            self.assertFalse(self.editor.save_as())
        target = self.directory / 'new.json'
        with patch('tkinter.filedialog.asksaveasfilename', return_value=str(target)), \
                patch('synreal_mujoco.project_panel.save_project_data', side_effect=OSError('disk full')), \
                patch('tkinter.messagebox.showerror'):
            self.assertFalse(self.editor.save_as())
        self.assertEqual(self.editor.project_file, original)
        self.assertTrue(self.editor.dirty)


class PanelLaunchTests(unittest.TestCase):
    def test_launch_owns_lifecycle_and_returns_applied_data(self):
        initial, applied = pdc.project_data(), pdc.project_data()
        build_scene = MagicMock()
        bundle = build_scene.return_value.build.return_value
        with patch('synreal_mujoco.project_panel._PanelProcess') as process, \
                patch('synreal_mujoco._project_panel_session.run_session', return_value=applied) as session:
            result = launch(initial, Path('.'), build_scene=build_scene)
        self.assertIs(result, applied)
        build_scene.assert_called_once_with(initial, Path('.').resolve())
        process.return_value.start.assert_called_once()
        process.return_value.close.assert_called_once()
        session.assert_called_once_with(initial, Path('.').resolve(), bundle, build_scene, process.return_value)

    def test_launch_cleans_up_after_session_failure(self):
        with patch('synreal_mujoco.project_panel._PanelProcess') as process, \
                patch('synreal_mujoco._project_panel_session.run_session', side_effect=RuntimeError('viewer failed')):
            with self.assertRaisesRegex(RuntimeError, 'viewer failed'):
                launch(pdc.project_data(), Path('.'), build_scene=MagicMock())
        process.return_value.close.assert_called_once()

    def test_failed_initial_build_does_not_start_panel(self):
        build_scene = MagicMock(side_effect=ValueError('bad project'))
        with patch('synreal_mujoco.project_panel._PanelProcess') as process:
            with self.assertRaisesRegex(ValueError, 'bad project'):
                launch(pdc.project_data(), Path('.'), build_scene=build_scene)
        process.assert_not_called()

    def test_failed_apply_preserves_model_and_steps(self):
        original = pdc.project_data()
        panel, viewer = MagicMock(), MagicMock()
        model, data, scene = MagicMock(), MagicMock(), MagicMock()
        panel.commands = Queue()
        panel.commands.put(('apply', pdc.project_data()))
        viewer.is_running.side_effect = [True, False]
        with patch('synreal_mujoco._project_panel_session.mujoco.viewer.launch_passive') as view, \
                patch('synreal_mujoco._project_panel_session.mujoco.mj_step') as step, \
                patch('synreal_mujoco._project_panel_session.s3d_scene_stepper'):
            view.return_value.__enter__.return_value = viewer
            result = run_session(original, Path('.'), (model, data, scene), MagicMock(), panel)
        self.assertIs(result, original)
        panel.notify.assert_any_call('error', 'Project must contain entities.')
        step.assert_called_once_with(model, data)

    def test_successful_apply_restarts_viewer_and_returns_new_data(self):
        original = pdc.project_data()
        candidate = pdc.project_data(entities=[pdc.cloth()])
        panel, viewer = MagicMock(), MagicMock()
        bundle = (MagicMock(), MagicMock(), MagicMock())
        panel.commands = Queue()
        panel.commands.put(('apply', candidate))
        viewer.is_running.side_effect = [True, False]
        with patch('synreal_mujoco._project_panel_session.mujoco.viewer.launch_passive') as view, \
                patch('synreal_mujoco._project_panel_session.s3d_scene_stepper'), \
                patch('synreal_mujoco._project_panel_session._build_candidate', return_value=bundle):
            view.return_value.__enter__.return_value = viewer
            result = run_session(original, Path('.'), bundle, MagicMock(), panel)
        self.assertIs(result, candidate)
        panel.notify.assert_any_call('applied', candidate)
        self.assertEqual(view.call_count, 2)

    def test_runner_delegates_to_launch(self):
        runner = project_runner()
        project_path = Path('synreal_mujoco/projects/piper_cloth').resolve()
        applied = pdc.project_data()
        with patch('synreal_mujoco.project_runner.project_io.project_io') as io, \
                patch('synreal_mujoco.project_runner.launch_panel', return_value=applied) as panel:
            runner.run(project_path)
        panel.assert_called_once_with(io.return_value.get_data.return_value, project_path,
                                      build_scene=runner._config_project)
        self.assertIs(runner.project_data, applied)


if __name__ == '__main__':
    unittest.main()
