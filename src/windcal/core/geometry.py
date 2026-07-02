from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelGeometry:
    """Stores the model geometry for processing into load data into coefficients"""

    name: str = field(default="Unknown", init=True, repr=True, hash=True, compare=True)
    ref_length: float = field(default=0.0, init=True, repr=True, hash=True, compare=True)
    ref_area: float = field(default=0.0, init=True, repr=True, hash=True, compare=True)
    mean_aero_chord: float = field(default=0.0, init=True, repr=True, hash=True, compare=True)
    span: float = field(default=0.0, init=True, repr=True, hash=True, compare=True)
    moment_x_shift: float = field(default=0.0, init=True, repr=True, hash=True, compare=True)
    moment_z_shift: float = field(default=0.0, init=True, repr=True, hash=True, compare=True)
