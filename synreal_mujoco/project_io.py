

from pathlib import Path
import json
import numpy as np

if __package__:
    from . import project_data_classes as dc
    from .project_data_classes import data_class_helper
else:
    import project_data_classes as dc
    from project_data_classes import data_class_helper


class project_io:
    def __init__(self):
        self.project_data: dc.project_data = dc.project_data()

    def read(self, project_path:Path):
        self.entry_file = self._compute_project_entry_file(project_path)
        self.project_data = self._load_project_data(self.entry_file)

    def save(self, project_path:Path):

        with open(project_path,'w',encoding='utf-8') as file:
            json_data = project_io._dump_json(self.project_data)
            json.dump(json_data, file, indent=4)

    def write(self):
        self.save(self.entry_file)

    def get_data(self) -> dc.project_data:
        return self.project_data

    def _compute_project_entry_file(self,project_path:Path):
        entry_file = project_path / 'main.json'
        if not Path.exists(entry_file):
            print(str(entry_file) + ' not exists!')
        else:
            print('opening project: ' + str(entry_file))
        return entry_file

    @staticmethod
    def _load_project_data(entry_file:Path) -> dc.project_data:

        with open(entry_file,'r',encoding='utf-8') as file:
            json_data = json.load(file)
            result = project_io._undump_json(None, json_data)
            if not isinstance(result, dc.project_data):
                raise ValueError('Project JSON must contain a project_data object at the root')
            return result


    @staticmethod
    def _dump_json(p_project_data:dc.project_data) -> dict:
        """Return the JSON-compatible representation of project data."""
        def encode(value):
            if isinstance(value, Path):
                return str(value.as_posix())
            if isinstance(value, np.ndarray):
                return encode(value.tolist())
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            if isinstance(value, dict):
                return {key: encode(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [encode(item) for item in value]

            class_name = data_class_helper.get_class_name(value)
            if class_name is not None:
                return {class_name: {key: encode(item) for key, item in vars(value).items()}}
            return value

        return encode(p_project_data)

    @staticmethod
    def _undump_json(j_key, j_value):
        """Return a decoded value, constructing classes from registered type names."""
        if isinstance(j_value, dict):
            # Dataclass values wrap their fields in a class name, including nested fields.
            if data_class_helper.get_class(j_key) is None and len(j_value) == 1:
                class_name, fields = next(iter(j_value.items()))
                if data_class_helper.get_class(class_name) is not None:
                    return project_io._undump_json(class_name, fields)

            cls = data_class_helper.get_class(j_key)
            if cls is not None:
                result = cls()
                for key, value in j_value.items():
                    decoded = project_io._undump_json(key, value)
                    if key == 'path' and decoded is not None:
                        decoded = Path(decoded)
                    elif isinstance(getattr(result, key, None), np.ndarray):
                        decoded = np.asarray(decoded)
                    setattr(result, key, decoded)
                return result

            return {
                key: project_io._undump_json(key, value)
                for key, value in j_value.items()
            }

        if isinstance(j_value, list):
            return [project_io._undump_json(None, value) for value in j_value]

        return j_value
