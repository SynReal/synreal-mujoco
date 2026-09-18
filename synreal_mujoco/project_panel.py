"""Independent dataclass editor with explicit project file operations."""

from copy import deepcopy
from dataclasses import fields, is_dataclass
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty, Full
from typing import get_type_hints

import numpy as np

from synreal_mujoco.project_io import project_io
from synreal_mujoco import project_data_classes as pdc

__all__ = ['launch']


def launch(project_data, project_path, *, build_scene):
    """Run the panel and MuJoCo viewer, returning the last applied project data.

    ``build_scene(data, project_path)`` must return a scene builder whose
    ``build()`` returns ``(model, data, scene)``. It is invoked on the calling
    thread, initially and after Apply. A failed Apply preserves the active
    scene. No caller-owned queues, polling, or shutdown code are needed.

    Call from a guarded entry point (``if __name__ == '__main__'``) since the
    panel uses a spawned process. This function blocks until the viewer closes.
    """
    from synreal_mujoco._project_panel_session import run_session

    project_path = Path(project_path).resolve()
    # Build before launching the panel so startup failures leave no UI process.
    bundle = build_scene(project_data, project_path).build()
    panel = _PanelProcess(project_data, project_path)
    try:
        panel.start()
        return run_session(project_data, project_path, bundle, build_scene, panel)
    finally:
        panel.close()


def set_field(model, path, value):
    for key in path[:-1]:
        model = model[key] if isinstance(key, int) else getattr(model, key)
    setattr(model, path[-1], value)


def parse_number(text, name):
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f'{name}: enter a number.') from None
    if not math.isfinite(value):
        raise ValueError(f'{name}: enter a finite number.')
    if name in ('density', 'thickness', 'youngs_modulus') and value <= 0:
        raise ValueError(f'{name}: must be greater than zero.')
    if name in ('stretch_stiffness', 'bend_stiffness', 'static_friction',
                'dynamic_friction', 'surface_offsets', 'yield_curvature',
                'volume_conserve_strength') and value < 0:
        raise ValueError(f'{name}: cannot be negative.')
    if name == 'poisson_ratio' and not 0 <= value < 0.5:
        raise ValueError('poisson_ratio: must be in [0, 0.5).')
    return value


def save_project_data(path, data):
    """Serialize only on explicit save, using an atomic replacement."""
    path = Path(path)
    content = json.dumps(project_io._dump_json(data), indent=4, allow_nan=False)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(content + '\n', encoding='utf-8')
    temporary.replace(path)


def _rebase_paths(data, source_directory, target_directory):
    """Keep references pointing to the same assets when a JSON file moves."""
    def convert(value):
        if isinstance(value, Path):
            if value.is_absolute():
                return value
            absolute = (source_directory / value).resolve()
            try:
                return Path(os.path.relpath(absolute, target_directory))
            except ValueError:  # Different Windows drives require absolute paths.
                return absolute
        if is_dataclass(value):
            for field in fields(value):
                setattr(value, field.name, convert(getattr(value, field.name)))
        elif isinstance(value, list):
            return [convert(item) for item in value]
        return value

    return convert(deepcopy(data))


def _validate_editor_data(data):
    if not isinstance(data, pdc.project_data) or not isinstance(data.entities, list):
        raise ValueError('Project must contain an entities list.')
    for entity in data.entities:
        if not isinstance(entity, (pdc.mjcf_rigidbody, pdc.cloth, pdc.deformable_body)):
            raise ValueError('Unsupported project entity.')
        if not isinstance(entity.path, Path):
            raise ValueError('Each entity must have an asset path.')
        if entity.trans is not None:
            if not isinstance(entity.trans, pdc.transform):
                raise ValueError('Invalid entity transform.')
            for name in ('translation', 'euler_xyz'):
                vector = getattr(entity.trans, name)
                if not isinstance(vector, np.ndarray) or vector.shape != (3,) or not np.issubdtype(vector.dtype, np.number):
                    raise ValueError(f'{name}: expected three numbers.')
                for value in vector:
                    parse_number(str(value), name)
        if isinstance(entity, (pdc.cloth, pdc.deformable_body)):
            expected = pdc.cloth_attrib if isinstance(entity, pdc.cloth) else pdc.deformable_body_attrib
            if not isinstance(entity.attrib, expected):
                raise ValueError('Invalid material attributes.')
            for field in fields(entity.attrib):
                value = getattr(entity.attrib, field.name)
                if field.type is np.ndarray:
                    if not isinstance(value, np.ndarray) or value.shape != (3,) or not np.issubdtype(value.dtype, np.number):
                        raise ValueError(f'{field.name}: expected three numbers.')
                    for component in value:
                        parse_number(str(component), field.name)
                elif field.type is bool:
                    if not isinstance(value, bool):
                        raise ValueError(f'{field.name}: expected true or false.')
                else:
                    if not isinstance(value, (int, float)) or isinstance(value, bool):
                        raise ValueError(f'{field.name}: expected a number.')
                    parse_number(str(value), field.name)


class _PanelProcess:
    def __init__(self, project_data, project_path):
        context = mp.get_context('spawn')
        self.commands = context.Queue(maxsize=16)
        self.events = context.Queue(maxsize=16)
        self.status = context.Queue(maxsize=1)
        self.stop = context.Event()
        self.process = context.Process(
            target=_run_panel,
            args=(deepcopy(project_data), Path(project_path), self.commands,
                  self.events, self.status, self.stop),
            name='project-panel', daemon=True,
        )

    def start(self):
        self.process.start()

    def notify(self, kind, value):
        try:
            self.events.put_nowait((kind, value))
        except Full:
            pass

    def publish(self, simulation_time, paused):
        try:
            self.status.put_nowait((simulation_time, paused))
        except Full:
            pass

    def close(self):
        self.stop.set()
        if self.process.pid is not None:
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2)
        for channel in (self.commands, self.events, self.status):
            channel.cancel_join_thread()
            channel.close()


def _run_panel(data, path, commands, events, status, stop):
    try:
        import tkinter as tk
        root = tk.Tk()
        _ProjectEditor(root, data, path, commands, events, status, stop)
        root.mainloop()
    except Exception as error:
        try:
            commands.put_nowait(('panel_error', str(error)))
        except Full:
            pass


class _ProjectEditor:
    def __init__(self, root, project_data, project_path, commands, events, status, stop):
        import tkinter as tk
        from tkinter import ttk

        self.root, self.project_path = root, Path(project_path).resolve()
        self.project_file = self.project_path / 'main.json'
        self.project_data = deepcopy(project_data)
        self.commands, self.events, self.status, self.stop = commands, events, status, stop
        self.bindings, self.controls = {}, []
        self.dirty, self.pending = False, False
        self._pending_open = None
        self.paused = tk.BooleanVar(root, value=False)
        self.message = tk.StringVar(root, value='Use File to save or open a project.')
        self.clock = tk.StringVar(root, value='0.000 s')
        root.title(f'{self.project_path.name} | Project')
        root.geometry('640x780')
        root.minsize(540, 420)
        root.protocol('WM_DELETE_WINDOW', self.close)
        menu = tk.Menu(root)
        self.file_menu = tk.Menu(menu, tearoff=False)
        self.file_menu.add_command(label='Save', accelerator='Ctrl+S', command=self.save)
        self.file_menu.add_command(label='Save As...', accelerator='Ctrl+Shift+S', command=self.save_as)
        self.file_menu.add_command(label='Open...', accelerator='Ctrl+O', command=self.open_project)
        menu.add_cascade(label='File', menu=self.file_menu)
        root.configure(menu=menu)
        root.bind('<Control-s>', lambda event: self._file_shortcut(self.save))
        root.bind('<Control-Shift-S>', lambda event: self._file_shortcut(self.save_as))
        root.bind('<Control-o>', lambda event: self._file_shortcut(self.open_project))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        toolbar = ttk.Frame(root, padding=10)
        toolbar.grid(row=0, column=0, sticky='ew')
        self.project_label = ttk.Label(toolbar, text=self.project_path.name, font=('Segoe UI', 12, 'bold'))
        self.project_label.pack(side='left')
        ttk.Checkbutton(toolbar, text='Pause', variable=self.paused, command=self.pause).pack(side='right')
        ttk.Label(toolbar, textvariable=self.clock, width=12).pack(side='right', padx=10)
        self.notebook = ttk.Notebook(root)
        self.notebook.grid(row=1, column=0, sticky='nsew', padx=10)
        self.build_tabs()
        footer = ttk.Frame(root, padding=10)
        footer.grid(row=2, column=0, sticky='ew')
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.message, wraplength=500).grid(row=0, column=0, sticky='w')
        root.after(100, self.poll)

    def build_tabs(self):
        import tkinter as tk
        from tkinter import ttk

        for tab in self.notebook.tabs():
            self.notebook.nametowidget(tab).destroy()
        self.bindings.clear()
        self.controls.clear()
        for index, entity in enumerate(self.project_data.entities or []):
            outer = ttk.Frame(self.notebook)
            title = type(entity).__name__.replace('_', ' ').title()
            self.notebook.add(outer, text=f'{index + 1}. {title}')
            outer.columnconfigure(0, weight=1)
            outer.rowconfigure(0, weight=1)
            canvas = tk.Canvas(outer, highlightthickness=0, background='#f0f0f0')
            scrollbar = ttk.Scrollbar(outer, orient='vertical', command=canvas.yview)
            canvas.configure(yscrollcommand=scrollbar.set)
            canvas.grid(row=0, column=0, sticky='nsew')
            scrollbar.grid(row=0, column=1, sticky='ns')
            form = ttk.Frame(canvas, padding=12)
            form.columnconfigure(1, weight=1)
            window = canvas.create_window(0, 0, window=form, anchor='nw')
            form.bind('<Configure>', lambda event, c=canvas: c.configure(scrollregion=c.bbox('all')))
            canvas.bind('<Configure>', lambda event, c=canvas, w=window: c.itemconfigure(w, width=event.width))
            self.add_fields(form, entity, ('entities', index), type(entity).__name__)
        settings = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(settings, text='Project')
        ttk.Label(settings, text=f'Viewer: {self.project_data.viewer or "MuJoCo (default)"}').pack(anchor='w')

    def add_fields(self, parent, model, path, kind, row=0):
        from tkinter import ttk
        types = get_type_hints(type(model))
        for field in fields(model):
            value = getattr(model, field.name)
            current = path + (field.name,)
            if is_dataclass(value):
                ttk.Separator(parent).grid(row=row, column=0, columnspan=2, sticky='ew', pady=8)
                title = {'attrib': 'Material', 'trans': 'Transform'}.get(field.name, field.name)
                ttk.Label(parent, text=title, font=('Segoe UI', 10, 'bold')).grid(row=row + 1, column=0, columnspan=2, sticky='w')
                row = self.add_fields(parent, value, current, kind, row + 2)
                continue
            label = {'euler_xyz': 'Euler XYZ (rad)', 'translation': 'Translation (m)'}.get(field.name, field.name.replace('_', ' ').capitalize())
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', padx=(0, 10), pady=6)
            # These settings are not implemented by the current scene runner.
            readonly = value is None or (kind == 'mjcf_rigidbody' and 'trans' in path) or (kind == 'deformable_body' and field.name == 'euler_xyz')
            if readonly:
                ttk.Label(parent, text='Not set' if value is None else str(value), foreground='#666666').grid(row=row, column=1, sticky='w')
            else:
                self.add_control(parent, current, value, types[field.name], row)
            row += 1
        return row

    def add_control(self, parent, path, value, field_type, row):
        import tkinter as tk
        from tkinter import ttk
        container = ttk.Frame(parent)
        container.grid(row=row, column=1, sticky='ew')
        variables = []
        if isinstance(value, bool):
            variable = tk.BooleanVar(parent, value=value)
            ttk.Checkbutton(container, variable=variable).pack(anchor='w')
            variables.append(variable)
        elif isinstance(value, (int, float, np.ndarray)):
            vector = isinstance(value, np.ndarray)
            for index, number in enumerate(value if vector else [value]):
                container.columnconfigure(index, weight=1)
                variable = tk.StringVar(parent, value=str(number))
                increment = 10 ** (math.floor(math.log10(abs(number))) - 1) if number else 0.01
                if vector:
                    label = 'XYZ'[index] if path[-1] in ('translation', 'euler_xyz') else str(index + 1)
                    ttk.Label(container, text=label).grid(row=0, column=index, sticky='w')
                ttk.Spinbox(container, textvariable=variable, from_=-1e15, to=1e15, increment=increment, format='%.12g', width=10).grid(row=int(vector), column=index, sticky='ew', padx=(0, 4))
                variables.append(variable)
        else:
            variable = tk.StringVar(parent, value=str(value))
            container.columnconfigure(0, weight=1)
            ttk.Entry(container, textvariable=variable).grid(row=0, column=0, sticky='ew')
            if isinstance(value, Path):
                ttk.Button(container, text='Browse', command=lambda: self.browse(variable), width=7).grid(row=0, column=1, padx=(4, 0))
            variables.append(variable)
        self.bindings[path] = (variables, value, field_type)
        self.controls.extend(child for child in container.winfo_children() if not isinstance(child, ttk.Label))
        for variable in variables:
            variable.trace_add('write', self.mark_dirty)

    def mark_dirty(self, *_):
        self.dirty = True
        self.message.set('Modified')

    def browse(self, variable):
        from tkinter import filedialog
        name = filedialog.askopenfilename(parent=self.root, initialdir=self.project_file.parent)
        if name:
            try:
                name = os.path.relpath(name, self.project_path)
            except ValueError:
                pass
            variable.set(Path(name).as_posix())

    def collect(self):
        data = deepcopy(self.project_data)
        for path, (variables, original, field_type) in self.bindings.items():
            if isinstance(original, bool):
                value = bool(variables[0].get())
            elif isinstance(original, np.ndarray):
                value = np.array([parse_number(v.get(), path[-1]) for v in variables], dtype=float)
            elif field_type in (float, int):
                value = parse_number(variables[0].get(), path[-1])
                if field_type is int:
                    if not value.is_integer():
                        raise ValueError(f'{path[-1]}: enter an integer.')
                    value = int(value)
            elif isinstance(original, Path):
                value = Path(variables[0].get())
                if not (self.project_path / value).is_file():
                    raise ValueError(f'File not found: {value}')
            else:
                value = variables[0].get()
            set_field(data, path, value)
        return data

    def send(self, kind, value):
        try:
            self.commands.put_nowait((kind, value))
            return True
        except Full:
            self.message.set('Simulation is busy. Try again.')
            return False

    def pause(self):
        self.send('pause', self.paused.get())

    def apply(self):
        from tkinter import messagebox
        if self.pending:
            return
        try:
            if self.send('apply', self.collect()):
                self.set_pending(True)
                self.message.set('Restarting simulation')
        except ValueError as error:
            messagebox.showerror('Invalid project', str(error), parent=self.root)

    def set_pending(self, pending):
        self.pending = pending
        for widget in self.controls:
            widget.state(['disabled' if pending else '!disabled'])
        for index in range(3):
            self.file_menu.entryconfigure(index, state='disabled' if pending else 'normal')

    def _file_shortcut(self, action):
        if not self.pending:
            action()
        return 'break'

    def _update_file_title(self):
        self.root.title(f'{self.project_file} | Project')
        self.project_label.configure(text=f'{self.project_file.parent.name} / {self.project_file.name}')

    def save(self):
        return self._save_to(self.project_file)

    def _save_to(self, target):
        from tkinter import messagebox
        if self.pending:
            return False
        try:
            data = self.collect()
            target = Path(target).resolve()
            save_project_data(target, _rebase_paths(data, self.project_path, target.parent))
            self.project_file = target
            self._update_file_title()
            self.dirty = False
            self.message.set(f'Saved {target.name}; reopen it to restart the simulation.')
            return True
        except (ValueError, OSError) as error:
            messagebox.showerror('Cannot save project', str(error), parent=self.root)
            return False

    def save_as(self):
        from tkinter import filedialog
        if self.pending:
            return False
        name = filedialog.asksaveasfilename(
            parent=self.root, title='Save Project As',
            initialdir=self.project_file.parent, initialfile=self.project_file.name,
            defaultextension='.json', filetypes=[('Project JSON', '*.json')],
        )
        return self._save_to(name) if name else False

    def _confirm_unsaved(self):
        from tkinter import messagebox
        if not self.dirty:
            return True
        answer = messagebox.askyesnocancel(
            'Unsaved project', f'Save changes to {self.project_file.name}?', parent=self.root,
        )
        return answer is False or (answer is True and self.save())

    def open_project(self):
        from tkinter import filedialog, messagebox
        if self.pending:
            return False
        name = filedialog.askopenfilename(
            parent=self.root, title='Open Project', initialdir=self.project_file.parent,
            filetypes=[('Project JSON', '*.json')],
        )
        if not name:
            return False
        try:
            target = Path(name).resolve()
            data = project_io._load_project_data(target)
            _validate_editor_data(data)
            # Keep the session's asset base fixed; no caller or simulation changes.
            data = _rebase_paths(data, target.parent, self.project_path)
        except (ValueError, OSError, TypeError, AttributeError) as error:
            messagebox.showerror('Cannot open project', str(error), parent=self.root)
            return False
        if not self._confirm_unsaved():
            return False
        if not self.send('apply', data):
            return False
        self._pending_open = target
        self.set_pending(True)
        self.message.set(f'Opening {target.name} and restarting simulation')
        return True

    def revert(self):
        self.build_tabs()
        self.message.set('Restored last opened or applied settings')

    def poll(self):
        if self.stop.is_set():
            self.root.destroy()
            return
        while True:
            try:
                kind, value = self.events.get_nowait()
            except Empty:
                break
            if kind in ('applied', 'error'):
                self.set_pending(False)
            if kind == 'applied':
                self.project_data = deepcopy(value)
                if self._pending_open is not None:
                    self.project_file = self._pending_open
                    self._pending_open = None
                    self.build_tabs()
                    self._update_file_title()
                    self.dirty = False
                try:
                    save_project_data(self.project_path / 'auto_save.json', value)
                    self.message.set('Applied to simulation')
                except OSError as error:
                    self.message.set(f'Applied; auto-save failed: {error}')
            elif kind == 'error':
                action = 'Open' if self._pending_open is not None else 'Apply'
                self._pending_open = None
                self.message.set(f'{action} failed: {value}')
        try:
            simulation_time, paused = self.status.get_nowait()
            self.clock.set(f'{simulation_time:.3f} s')
            self.paused.set(paused)
        except Empty:
            pass
        self.root.after(100, self.poll)

    def close(self):
        if not self._confirm_unsaved():
            return
        self.send('pause', False)
        self.root.destroy()
