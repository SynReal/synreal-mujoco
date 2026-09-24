from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
import numpy as np

@dataclass
class transform:
    translation: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))
    euler_xyz: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))


@dataclass
class deformable_body_attrib:
    density: float = 950 
    dynamic_friction: float = 0.03  # Resistance to sliding contact.
    poisson_ratio: float = 0.45  # Lateral to axial strain ratio (0 to 0.5).
    static_friction: float = 0.03  # Resistance to the start of sliding.
    surface_offsets: float = 2e-3  # Collision surface expansion in meters.
    youngs_modulus: float = 5e4  # Material stiffness in pascals.


@dataclass
class cloth_attrib:
    stretch_stiffness: np.ndarray = field(default_factory=lambda: np.array([150.0, 150.0, 150.0]))
    bend_stiffness: np.ndarray = field(default_factory=lambda: np.array([2e-6, 2e-6, 2e-6]))
    thickness: float = 1.0e-3
    density: float = 0.1
    pressure: float = 0.0
    static_friction: float = 0.03
    dynamic_friction: float = 0.03

    yield_curvature: float = 1e1  # Curve radius equivalent to 0.1 m.
    volume_conserve_strength: float = 1e2
    frozen: bool = False


@dataclass
class cloth:
    path: Path | None = None
    attrib: cloth_attrib = field(default_factory=cloth_attrib)
    trans: transform = field(default_factory=transform)

@dataclass
class deformable_body:
    path: Path | None = None
    attrib: deformable_body_attrib = field(default_factory=deformable_body_attrib)
    trans: transform = field(default_factory=transform)

@dataclass
class rigid_mesh:
    path: Path | None = None
    trans: transform | None = field(default_factory=transform)

@dataclass
class rigid_shape:
    type: str = 'box'
    trans: transform | None = None

@dataclass
class mjcf_scene:
    path: Path | None = None
    trans: transform | None = None

@dataclass
class project_data:
    entities: list[cloth | deformable_body | mjcf_scene | rigid_mesh] | None = None
    viewer: str | None = None


class data_class_helper:
    _data_classes = {
        cls.__name__: cls
        for cls in tuple(globals().values())
        if isinstance(cls, type) and is_dataclass(cls) and cls.__module__ == __name__
    }

    @staticmethod
    def get_class(name):
        return data_class_helper._data_classes.get(name)

    @staticmethod
    def get_class_name(value):
        return next(
            (name for name, cls in data_class_helper._data_classes.items()
             if isinstance(value, cls)),
            None,
        )
