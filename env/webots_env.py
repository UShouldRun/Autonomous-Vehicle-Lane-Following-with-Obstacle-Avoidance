"""""
webots_env.py — Low-level Webots hardware driver for the City demo car.

Knows nothing about Gym, rewards, or RL. Only speaks to devices.
Uses Ackermann steering (left_steer / right_steer) + front-wheel drive.
"""

from typing import List, Optional

import numpy as np
import cv2
from controller import Supervisor

TIME_STEP       = 32      # ms — must match basicTimeStep in your .wbt
MAX_STEER       = 0.5     # radians — max steering angle
MAX_SPEED       = 50.0    # rad/s  — front wheel motor limit

# Yellow line HSV bounds — tune against your world's colour
YELLOW_LO = np.array([18,  80,  80], dtype=np.uint8)
YELLOW_HI = np.array([35, 255, 255], dtype=np.uint8)


class WebotsEnv:
    """
    Thin wrapper around the City demo car's Webots devices.

      - Enable and read sensors (camera, LiDAR)
      - Write steering and throttle commands (Ackermann drive)
      - Reset simulation state via Supervisor
      - Expose raw physical quantities (speed, alignment angle)
      - Track episode-level metrics (distance travelled, lap completion,
        near-misses) for use by the Gym wrapper.
    """

    def __init__(
        self,
        near_miss_threshold: float = 1.0,
        collision_threshold: float = 0.3,
        lap_departure_distance: float = 20.0,
        lap_return_distance: float = 5.0,
    ):
        """
        Parameters
        ----------
        near_miss_threshold : float
            LiDAR distance (m) below which a step counts as a near-miss
            (but not a collision). A near-miss is a min-lidar reading in
            (collision_threshold, near_miss_threshold].
        collision_threshold : float
            LiDAR distance (m) below which a step is treated as a
            collision; used here only to separate near-miss from crash.
        lap_departure_distance : float
            Metres the car must travel away from its spawn before the
            lap-completion detector becomes armed.
        lap_return_distance : float
            Metres within which, after having departed, a return to the
            spawn counts as completing one lap.
        """
        self.robot = Supervisor()

        # ── Camera ────────────────────────────────────────────────
        self.camera = self.robot.getDevice("camera")
        self.camera.enable(TIME_STEP)
        self.cam_w = self.camera.getWidth()
        self.cam_h = self.camera.getHeight()

        # ── LiDAR ─────────────────────────────────────────────────
        self.lidar = self.robot.getDevice("lidar")
        self.lidar.enable(TIME_STEP)
        # Use horizontal resolution for a flat 1-D range scan (no point cloud needed)
        self.lidar_size = self.lidar.getHorizontalResolution()

        # ── Steering motors ───────────────────────────────────────
        self.left_steer  = self.robot.getDevice("left_steer")
        self.right_steer = self.robot.getDevice("right_steer")
        for s in (self.left_steer, self.right_steer):
            s.setPosition(0.0)   # position-control mode, start straight

        # ── Drive wheels (front) ──────────────────────────────────
        self.left_wheel  = self.robot.getDevice("left_front_wheel")
        self.right_wheel = self.robot.getDevice("right_front_wheel")
        for w in (self.left_wheel, self.right_wheel):
            w.setPosition(float("inf"))  # velocity-control mode
            w.setVelocity(0.0)

        # ── Supervisor: needed for teleport-reset ─────────────────
        self.robot_node = self.robot.getSelf()
        print(self.robot_node)
        self._init_translation = list(
            self.robot_node.getField("translation").getSFVec3f()
        )
        self._init_rotation = list(
            self.robot_node.getField("rotation").getSFRotation()
        )

        # ── Episode tracking config ───────────────────────────────
        self._near_miss_threshold       = float(near_miss_threshold)
        self._collision_threshold       = float(collision_threshold)
        self._lap_departure_distance    = float(lap_departure_distance)
        self._lap_return_distance       = float(lap_return_distance)

        # ── Episode tracking state (populated by _reset_tracking) ─
        self._distance_travelled: float     = 0.0
        self._last_translation: List[float] = list(self._init_translation)
        self._has_departed: bool            = False
        self._lap_start_time: float         = 0.0
        self._laps_completed: int           = 0
        self._lap_times: List[float]        = []
        self._lap_just_completed_flag: bool = False
        self._near_miss_flag: bool          = False

    # ── Simulation control ────────────────────────────────────────

    def step(self) -> bool:
        """Advance one TIME_STEP. Returns False when Webots wants to quit.

        Also updates episode-level tracking (distance, lap detection,
        near-miss flag) after each successful step.
        """
        ok = self.robot.step(TIME_STEP) != -1
        if ok:
            self._update_tracking()
        return ok

    def reset(self):
        """Teleport car to spawn, clear physics, step once for fresh reads."""
        self.set_controls(0.0, 0.0)
        self.robot_node.getField("translation").setSFVec3f(self._init_translation)
        self.robot_node.getField("rotation").setSFRotation(self._init_rotation)
        self.robot.simulationResetPhysics()
        self.robot.step(TIME_STEP)
        self._reset_tracking()

    # ── Episode tracking ──────────────────────────────────────────

    def _reset_tracking(self):
        """Clear all episode-level tracking state."""
        self._distance_travelled       = 0.0
        self._last_translation         = list(self._init_translation)
        self._has_departed             = False
        self._lap_start_time           = float(self.robot.getTime())
        self._laps_completed           = 0
        self._lap_times                = []
        self._lap_just_completed_flag  = False
        self._near_miss_flag           = False

    def _update_tracking(self):
        """Update distance, lap state and near-miss flag for the most recent step."""
        now = float(self.robot.getTime())
        current = self.robot_node.getField("translation").getSFVec3f()

        # Path-length integration for total distance travelled.
        dx = current[0] - self._last_translation[0]
        dy = current[1] - self._last_translation[1]
        dz = current[2] - self._last_translation[2]
        self._distance_travelled += float(np.sqrt(dx * dx + dy * dy + dz * dz))
        self._last_translation = list(current)

        # Lap detection: arm after the car is far enough from spawn, then
        # fire exactly once when the car returns close to spawn.
        init = self._init_translation
        dxi = current[0] - init[0]
        dyi = current[1] - init[1]
        dzi = current[2] - init[2]
        dist_from_init = float(np.sqrt(dxi * dxi + dyi * dyi + dzi * dzi))

        self._lap_just_completed_flag = False
        if not self._has_departed:
            if dist_from_init > self._lap_departure_distance:
                self._has_departed = True
        elif dist_from_init < self._lap_return_distance:
            self._lap_times.append(now - self._lap_start_time)
            self._lap_start_time = now
            self._laps_completed += 1
            self._has_departed = False
            self._lap_just_completed_flag = True

        # Near-miss: closest obstacle is inside the warning band but has
        # not crossed the collision threshold.
        d_min = self.get_min_lidar_distance()
        self._near_miss_flag = (
            self._collision_threshold < d_min <= self._near_miss_threshold
        )

    # ── Episode tracking accessors ───────────────────────────────

    def get_lap_completed(self) -> bool:
        """True iff a full lap was completed in the most recent step."""
        return self._lap_just_completed_flag

    def get_last_lap_time(self) -> Optional[float]:
        """Duration (s) of the most recently completed lap, or None if none yet."""
        return self._lap_times[-1] if self._lap_times else None

    def get_lap_times(self) -> List[float]:
        """All lap durations (s) completed since the last reset."""
        return list(self._lap_times)

    def get_laps_completed(self) -> int:
        return self._laps_completed

    def get_distance_travelled(self) -> float:
        """Cumulative path length (m) since the last reset."""
        return self._distance_travelled

    def is_near_miss(self) -> bool:
        """True iff the most recent step registered a near-miss."""
        return self._near_miss_flag

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

        NOTE: This is an image-plane *proxy* for the cross-track error used
        in evaluation metrics — it is a unitless fraction of half-image
        width, NOT a lateral distance in metres. A proper CTE would require
        knowing the world-frame road geometry.
        """
        img    = self.get_camera_image()
        hsv    = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask   = cv2.inRange(hsv, YELLOW_LO, YELLOW_HI)
        cols   = np.where(mask.any(axis=0))[0]
        if len(cols) == 0:
            return 0.0
        cx     = float(cols.mean())
        centre = self.cam_w / 2.0
        return (cx - centre) / centre

    def get_min_lidar_distance(self) -> float:
        return float(self.get_lidar_scan().min())

    # ── Actuator writes ───────────────────────────────────────────

    def set_controls(self, steering: float, throttle: float):
        """
        steering ∈ [-1, 1]  → mapped to ±MAX_STEER radians
        throttle ∈ [-1, 1]  → mapped to ±MAX_SPEED rad/s on front wheels
        """
        angle = float(np.clip(steering, -1.0, 1.0)) * MAX_STEER
        speed = float(np.clip(throttle, -1.0, 1.0)) * MAX_SPEED

        self.left_steer.setPosition(angle)
        self.right_steer.setPosition(angle)
        self.left_wheel.setVelocity(speed)
        self.right_wheel.setVelocity(speed)

    def apply_continuous(self, steering: float, throttle: float):
        """
        steering ∈ [-1, 1]  (negative = turn left)
        throttle ∈ [-1, 1]  (positive = forward)
        """
        self.set_controls(steering, throttle)

    def apply_discrete(self, action: int):
        """
        0 = left   | 1 = right
        2 = straight | 3 = brake
        """
        cmds = {
            0: (-1.0,  0.5),   # steer left,  half throttle
            1: ( 1.0,  0.5),   # steer right, half throttle
            2: ( 0.0,  1.0),   # straight,    full throttle
            3: ( 0.0,  0.0),   # brake
        }
        self.set_controls(*cmds[int(action)])
