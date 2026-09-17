from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
import numpy as np

@dataclass
class transform:
    translation: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))
    eulerXYZ: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))

@dataclass
class mjcf_rigidbody:
    path: Path | None = None
    trans: transform | None = None

@dataclass
class deformable_body_attrib:
    density: float = 950 
    dynamicFriction: float = 0.03  # Resistance to sliding contact.
    poissonRatio: float = 0.45  # Lateral to axial strain ratio (0 to 0.5).
    staticFriction: float = 0.03  # Resistance to the start of sliding.
    surfaceOffsets: float = 2e-3  # Collision surface expansion in meters.
    youngsModulus: float = 5e4  # Material stiffness in pascals.


@dataclass
class cloth_attrib:
    stretchStiffness: np.ndarray = field(default_factory=lambda: np.array([150.0, 150.0, 150.0]))
    bendStiffness: np.ndarray = field(default_factory=lambda: np.array([2e-6, 2e-6, 2e-6]))
    thickness: float = 1.0e-3
    density: float = 0.1
    pressure: float = 0.0
    staticFriction: float = 0.03
    dynamicFriction: float = 0.03

    yieldCurvature: float = 1e1  # Curve radius equivalent to 0.1 m.
    volumeConserveStrength: float = 1e2
    frozen: bool = False

@dataclass
class deformable_body:
    path: Path | None = None
    attrib: deformable_body_attrib = field(default_factory=deformable_body_attrib)
    trans: transform = field(default_factory=transform)

@dataclass
class project_data:
    entities: list | None = None
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
