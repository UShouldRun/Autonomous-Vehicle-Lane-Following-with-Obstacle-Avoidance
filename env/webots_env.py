"""
webots_env.py — Low-level Webots hardware driver.

Knows nothing about Gym, rewards, or RL. Only speaks to devices.
Can be unit-tested by passing a mock Supervisor object.
"""

import numpy as np
import cv2
from controller import Supervisor

TIME_STEP       = 32     # ms — change basicTimeStep in your .wbt to match
MAX_SPEED       = 6.28   # rad/s — E-puck motor limit

# Yellow line HSV bounds — tune against your world's colour
YELLOW_LO = np.array([18,  80,  80], dtype=np.uint8)
YELLOW_HI = np.array([35, 255, 255], dtype=np.uint8)


class WebotsEnv:
    """
    Thin wrapper around Webots devices for the E-puck on the City track.

      - Enable and read sensors (camera, LiDAR)
      - Write motor commands
      - Reset simulation state
      - Expose raw physical quantities (speed, alignment angle)
    """

    def __init__(self):
        self.robot = Supervisor()

        # ── Camera ────────────────────────────────────────────────
        self.camera = self.robot.getDevice("camera")
        self.camera.enable(TIME_STEP)
        self.cam_w = self.camera.getWidth()
        self.cam_h = self.camera.getHeight()

        # ── LiDAR ─────────────────────────────────────────────────
        self.lidar = self.robot.getDevice("lidar")
        self.lidar.enable(TIME_STEP)
        self.lidar_size = self.lidar.getNumberOfPoints()

        # ── Motors ────────────────────────────────────────────────
        self.left_motor  = self.robot.getDevice("left wheel motor")
        self.right_motor = self.robot.getDevice("right wheel motor")
        for m in (self.left_motor, self.right_motor):
            m.setPosition(float("inf"))  # velocity-control mode
            m.setVelocity(0.0)

        # ── Supervisor: needed for teleport-reset ─────────────────
        self.robot_node = self.robot.getSelf()
        self._init_translation = list(
            self.robot_node.getField("translation").getSFVec3f()
        )
        self._init_rotation = list(
            self.robot_node.getField("rotation").getSFRotation()
        )

    # ── Simulation control ────────────────────────────────────────

    def step(self) -> bool:
        """Advance one TIME_STEP. Returns False when Webots wants to quit."""
        return self.robot.step(TIME_STEP) != -1

    def reset(self):
        """Teleport robot to spawn, clear physics, step once for fresh reads."""
        self.set_velocity(0.0, 0.0)
        self.robot_node.getField("translation").setSFVec3f(self._init_translation)
        self.robot_node.getField("rotation").setSFRotation(self._init_rotation)
        self.robot.simulationResetPhysics()
        self.robot.step(TIME_STEP)

    # ── Sensor reads ──────────────────────────────────────────────

    def get_camera_image(self) -> np.ndarray:
        """(H, W, 3) uint8 RGB array."""
        raw = self.camera.getImage()
        img = np.frombuffer(raw, dtype=np.uint8).reshape((self.cam_h, self.cam_w, 4))
        return img[:, :, :3].copy()   # drop alpha channel

    def get_lidar_scan(self) -> np.ndarray:
        """1-D float32 array of range values in metres, NaN/inf replaced with 10.0."""
        scan = np.array(self.lidar.getRangeImage(), dtype=np.float32)
        return np.clip(np.nan_to_num(scan, nan=10.0, posinf=10.0), 0.0, 10.0)

    def get_speed(self) -> float:
        """Scalar translational speed in m/s."""
        v = self.robot_node.getVelocity()   # [vx, vy, vz, wx, wy, wz]
        return float(np.linalg.norm(v[:3]))

    def get_alignment_angle(self) -> float:
        """
        Normalised lateral offset of the yellow centre line ∈ [-1, 1].
        Negative  → line is to the left of centre.
        Returns 0.0 when the line is not visible.
        """
        img  = self.get_camera_image()
        hsv  = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, YELLOW_LO, YELLOW_HI)
        cols = np.where(mask.any(axis=0))[0]
        if len(cols) == 0:
            return 0.0
        cx     = float(cols.mean())
        centre = self.cam_w / 2.0
        return (cx - centre) / centre

    def get_min_lidar_distance(self) -> float:
        return float(self.get_lidar_scan().min())

    # ── Actuator writes ───────────────────────────────────────────

    def set_velocity(self, left: float, right: float):
        """Set wheel velocities in rad/s, clamped to MAX_SPEED."""
        self.left_motor.setVelocity(np.clip(left,  -MAX_SPEED, MAX_SPEED))
        self.right_motor.setVelocity(np.clip(right, -MAX_SPEED, MAX_SPEED))

    def apply_continuous(self, steering: float, throttle: float):
        """
        steering ∈ [-1, 1]  (negative = turn left)
        throttle ∈ [-1, 1]  (positive = forward)
        """
        left  = MAX_SPEED * (throttle - steering)
        right = MAX_SPEED * (throttle + steering)
        self.set_velocity(left, right)

    def apply_discrete(self, action: int):
        """
        0 = left   | 1 = right
        2 = straight | 3 = brake
        """
        half = MAX_SPEED * 0.5
        cmds = {
            0: (-half,  half),
            1: ( half, -half),
            2: ( half,  half),
            3: (0.0,    0.0),
        }
        self.set_velocity(*cmds[int(action)])
