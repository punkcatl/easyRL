from dataclasses import dataclass
from mjlab.managers import CommandTermCfg


@dataclass(kw_only=True)
class UniformVelocityCommandCfg(CommandTermCfg):
    """Uniform velocity command configuration."""
    entity_name: str = "robot"
    heading_command: bool = False
    heading_control_stiffness: float = 1.0
    rel_standing_envs: float = 0.0
    rel_heading_envs: float = 1.0
    resampling_time: tuple[float, float] = (3.0, 8.0)

    @dataclass
    class Ranges:
        lin_vel_x: tuple[float, float] = (-1.0, 1.0)
        lin_vel_y: tuple[float, float] = (-0.5, 0.5)
        ang_vel_z: tuple[float, float] = (-1.0, 1.0)

    ranges: Ranges = None

    def __post_init__(self):
        if self.ranges is None:
            self.ranges = self.Ranges()
