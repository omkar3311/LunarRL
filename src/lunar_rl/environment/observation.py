"""Observation parsing and explanation utilities for LunarLander-v3."""

from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np


@dataclass
class LunarObservation:
    """Structured representation of LunarLander-v3 observation vector."""

    x: float
    y: float
    vx: float
    vy: float
    angle: float
    angular_velocity: float
    left_leg_contact: bool
    right_leg_contact: bool

    @classmethod
    def from_array(cls, obs: np.ndarray) -> "LunarObservation":
        """Creates LunarObservation from raw 8D numpy array."""
        if len(obs) != 8:
            raise ValueError(f"Expected observation vector of length 8, got {len(obs)}")
        return cls(
            x=float(obs[0]),
            y=float(obs[1]),
            vx=float(obs[2]),
            vy=float(obs[3]),
            angle=float(obs[4]),
            angular_velocity=float(obs[5]),
            left_leg_contact=bool(obs[6] > 0.5),
            right_leg_contact=bool(obs[7] > 0.5),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Converts observation to a descriptive dictionary."""
        return {
            "x_position": self.x,
            "y_position": self.y,
            "x_velocity": self.vx,
            "y_velocity": self.vy,
            "angle_rad": self.angle,
            "angle_deg": np.degrees(self.angle),
            "angular_velocity": self.angular_velocity,
            "left_leg_contact": self.left_leg_contact,
            "right_leg_contact": self.right_leg_contact,
        }

    def explain(self) -> str:
        """Returns a formatted human-readable explanation of current lander state."""
        tilt_status = "Upright" if abs(self.angle) < 0.1 else f"Tilted ({np.degrees(self.angle):+.1f}°)"
        touchdown_status = []
        if self.left_leg_contact:
            touchdown_status.append("Left Leg Ground Contact")
        if self.right_leg_contact:
            touchdown_status.append("Right Leg Ground Contact")
        legs_str = ", ".join(touchdown_status) if touchdown_status else "Airborne (No Contact)"

        lines = [
            f"  • Horizontal Pos (x): {self.x:+.4f} (Target: 0.00)",
            f"  • Vertical Pos (y):   {self.y:+.4f} (Pad: ~0.00)",
            f"  • Horiz. Velocity:    {self.vx:+.4f}",
            f"  • Vert. Velocity:     {self.vy:+.4f} ({'Descending' if self.vy < 0 else 'Ascending'})",
            f"  • Tilt Angle:         {self.angle:+.4f} rad ({tilt_status})",
            f"  • Angular Velocity:   {self.angular_velocity:+.4f} rad/s",
            f"  • Landing Gear:       {legs_str}",
        ]
        return "\n".join(lines)


OBSERVATION_DESCRIPTIONS: List[Dict[str, str]] = [
    {
        "index": "0",
        "name": "Horizontal Position (x)",
        "unit": "Normalized units (-1.0 to 1.0)",
        "description": "Lander's x-coordinate. 0.0 represents the exact center between the landing flags.",
    },
    {
        "index": "1",
        "name": "Vertical Position (y)",
        "unit": "Normalized units (0.0 to ~1.4)",
        "description": "Lander's altitude. 0.0 is the landing pad surface height.",
    },
    {
        "index": "2",
        "name": "Horizontal Velocity (vx)",
        "unit": "Normalized velocity",
        "description": "Velocity along the x-axis. Positive = moving right, Negative = moving left.",
    },
    {
        "index": "3",
        "name": "Vertical Velocity (vy)",
        "unit": "Normalized velocity",
        "description": "Velocity along the y-axis. Positive = moving up, Negative = descending down.",
    },
    {
        "index": "4",
        "name": "Angle (theta)",
        "unit": "Radians (-pi to pi)",
        "description": "Lander tilt angle. 0.0 is perfectly vertical/upright.",
    },
    {
        "index": "5",
        "name": "Angular Velocity (omega)",
        "unit": "Normalized radians/s",
        "description": "Rate of rotation. Positive = counter-clockwise, Negative = clockwise.",
    },
    {
        "index": "6",
        "name": "Left Leg Contact",
        "unit": "Boolean (0.0 or 1.0)",
        "description": "1.0 if left landing leg touches ground or pad, 0.0 otherwise.",
    },
    {
        "index": "7",
        "name": "Right Leg Contact",
        "unit": "Boolean (0.0 or 1.0)",
        "description": "1.0 if right landing leg touches ground or pad, 0.0 otherwise.",
    },
]

ACTION_DESCRIPTIONS: Dict[int, str] = {
    0: "Do nothing (No engine fired)",
    1: "Fire left orientation engine (turns lander right)",
    2: "Fire main engine (thrust upward)",
    3: "Fire right orientation engine (turns lander left)",
}
