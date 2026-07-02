from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)  # frozen=True makes the config immutable and hashable
class CalibrationMetadata:
    """Carries downstream configurations for the final BalanceCalibration artifact."""

    author: str = "System"
    # Holds type, manufacturer, serial_number, diameter, etc.
    balance_info: Optional[dict] = field(default_factory=dict)
    # Holds limit loads and units
    max_loads: Optional[dict] = field(default_factory=dict)
    # Holds the (X1-X4) locations for conversion
    distances: Optional[Tuple[float, float, float, float]] = None
