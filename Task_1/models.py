from dataclasses import dataclass
from enum import Enum, auto

from pe_typing import FloatArray


class ControlMode(Enum):
    FREE_MOTION = auto(),
    VELOCITY_CONTROL = auto(),
    TORQURE_CONTROL = auto()


@dataclass
class SceneInfo:
    planeId: int 
    boxId: int


@dataclass
class SimulationResult:
    positions: FloatArray
    velocities: FloatArray


TheoreticalResult = SimulationResult