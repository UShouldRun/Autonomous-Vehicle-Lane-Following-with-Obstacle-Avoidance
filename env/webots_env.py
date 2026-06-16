from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import cv2
from vehicle import Driver

TIME_STEP       = 10      # ms — matches basicTimeStep in city.wbt
MAX_STEER       = 0.5     # radians — max steering angle
MAX_SPEED       = 50      # km/h — Driver.setCruisingSpeed takes km/h

# Yellow line HSV bounds — widened for robustness across lighting/shadow in Webots.
# Hue  10-40: covers warm golden-yellow through bright pure yellow (OpenCV H = 0-179).
# Sat  60+  : excludes near-white road surface (which has low saturation).
# Val  60+  : handles shadowed sections of the line.
# Road line color in city_level1.wbt: color 0.85 0.75 0.30 → HSV ≈ (25, 165, 216)
# Tight bounds to avoid false positives from car interior, sky, buildings.
# Old bounds [10,60,60]→[40,255,255] matched car upholstery/interior when
# camera was mis-positioned inside the chassis (z was sideways not up).
YELLOW_LO = np.array([18,  80, 120], dtype=np.uint8)
YELLOW_HI = np.array([35, 255, 255], dtype=np.uint8)

BARREL_DEF_PREFIX = "BARREL_DYN"

class WebotsEnv:
    """
    Thin wrapper around the City demo car's Webots devices.
    
    - Enable and read sensors (camera, LiDAR)
    - Write steering and throttle commands scaled to physical properties
    - Track episode-level metrics (distance travelled, lap completion, near-misses)
    """

    def __init__(
        self,
        near_miss_threshold: float = 1.0,
        collision_threshold: float = 0.3,
        lap_departure_distance: float = 20.0,
        lap_return_distance: float = 5.0,
        obstacles_cfg: Optional[Dict[str, Any]] = None,
    ):
        self.driver = Driver()
        self._first_reset = True

        # ── Camera ────────────────────────────────────────────────
        self.camera = self.driver.getDevice("camera")
        self.camera.enable(TIME_STEP)
        self.cam_w = self.camera.getWidth()
        self.cam_h = self.camera.getHeight()

        # ── LiDAR ─────────────────────────────────────────────────
        self.lidar = self.driver.getDevice("lidar")
        self.lidar.enable(TIME_STEP)
        self.lidar_size = (
            self.lidar.getNumberOfLayers() * self.lidar.getHorizontalResolution()
        )

        # ── Steering motors ───────────────────────────────────────
        self.left_steer  = self.driver.getDevice("left_steer")
        self.right_steer = self.driver.getDevice("right_steer")
        for s in (self.left_steer, self.right_steer):
            s.setPosition(0.0)

        self.left_wheel  = self.driver.getDevice("left_front_wheel")
        self.right_wheel = self.driver.getDevice("right_front_wheel")
        for w in (self.left_wheel, self.right_wheel):
            w.setPosition(float("inf"))  # velocity-control mode
            w.setVelocity(0.0)

        # ── Supervisor: needed for teleport-reset ─────────────────
        self.driver_node = self.driver.getSelf()
        self._init_translation = list(
            self.driver_node.getField("translation").getSFVec3f()
        )
        self._init_rotation = list(
            self.driver_node.getField("rotation").getSFRotation()
        )

        # ── Episode tracking config ───────────────────────────────
        self._near_miss_threshold       = float(near_miss_threshold)
        self._collision_threshold       = float(collision_threshold)
        self._lap_departure_distance    = float(lap_departure_distance)
        self._lap_return_distance       = float(lap_return_distance)

        # ── Episode tracking state ────────────────────────────────
        self._distance_travelled: float     = 0.0
        self._forward_distance: float       = 0.0
        self._last_translation: List[float] = list(self._init_translation)
        self._has_departed: bool            = False
        self._lap_start_time: float         = 0.0
        self._laps_completed: int           = 0
        self._lap_times: List[float]        = []
        self._lap_just_completed_flag: bool = False
        self._near_miss_flag: bool          = False

        # ── Obstacle subsystem ────────────────────────────────────
        self._obstacles_cfg: Dict[str, Any]   = dict(obstacles_cfg or {})
        self._obstacles_enabled: bool          = bool(self._obstacles_cfg.get("enabled", False))
        self._obstacle_nodes: List[Dict[str, Any]] = []
        self._next_spawn_distance: float      = 0.0
        self._obstacle_counter: int           = 0

        self.driver.step()          
        self.driver.setGear(1)      
        self.driver.setCruisingSpeed(0.0)
        self.driver.setSteeringAngle(0.0)

    def step(self) -> bool:
        """Advance one TIME_STEP. Returns False when Webots wants to quit."""
        ok = self.driver.step() != -1
        if ok:
            self._update_tracking()
        return ok

    def reset(self):
        if self._first_reset:
            self._first_reset = False
            self._reset_tracking()
            self._reset_obstacle_state()
            return

        self.driver.setCruisingSpeed(0.0)
        self.driver.setSteeringAngle(0.0)

        trans_field = self.driver_node.getField("translation")
        rot_field   = self.driver_node.getField("rotation")
        trans_field.setSFVec3f(self._init_translation)
        rot_field.setSFRotation(self._init_rotation)
        self.driver_node.resetPhysics()

        for _ in range(5):
            self.driver.step()

        self._reset_tracking()
        self._reset_obstacle_state()

    def _reset_tracking(self):
        self._distance_travelled       = 0.0
        self._forward_distance         = 0.0
        self._last_translation         = list(self._init_translation)
        self._has_departed             = False
        self._lap_start_time           = float(self.driver.getTime())
        self._laps_completed           = 0
        self._lap_times                = []
        self._lap_just_completed_flag  = False
        self._near_miss_flag           = False

    def _update_tracking(self):
        now     = float(self.driver.getTime())
        current = self.driver_node.getField("translation").getSFVec3f()

        dx = current[0] - self._last_translation[0]
        dy = current[1] - self._last_translation[1]
        dz = current[2] - self._last_translation[2]

        self._distance_travelled += float(np.sqrt(dx * dx + dy * dy + dz * dz))

        rot = np.array(self.driver_node.getOrientation()).reshape(3, 3)
        forward_world = rot[:, 0]
        self._forward_distance += float(
            dx * forward_world[0] + dy * forward_world[1] + dz * forward_world[2]
        )

        self._last_translation = list(current)

        init = self._init_translation
        dxi  = current[0] - init[0]
        dyi  = current[1] - init[1]
        dzi  = current[2] - init[2]
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

        d_min = self.get_min_lidar_distance()
        self._near_miss_flag = (
            self._collision_threshold < d_min <= self._near_miss_threshold
        )

    def get_lap_completed(self) -> bool:
        return self._lap_just_completed_flag

    def get_last_lap_time(self) -> Optional[float]:
        return self._lap_times[-1] if self._lap_times else None

    def get_lap_times(self) -> List[float]:
        return list(self._lap_times)

    def get_laps_completed(self) -> int:
        return self._laps_completed

    def get_distance_travelled(self) -> float:
        return self._distance_travelled

    def get_forward_distance(self) -> float:
        return self._forward_distance

    def is_near_miss(self) -> bool:
        return self._near_miss_flag

    @property
    def obstacles_enabled(self) -> bool:
        return self._obstacles_enabled

    def update_obstacles(self) -> None:
        if not self._obstacles_enabled:
            return
        self._recycle_passed_obstacles()
        self._maybe_spawn_obstacle()

    def clear_obstacles(self) -> None:
        for entry in self._obstacle_nodes:
            node = entry.get("node")
            if node is None:
                continue
            try:
                node.remove()
            except Exception:
                pass
        self._reset_obstacle_state()

    def _reset_obstacle_state(self) -> None:
        self._obstacle_nodes = []
        interval_min = float(self._obstacles_cfg.get("spawn_interval_min_m", 30.0))
        interval_max = float(self._obstacles_cfg.get("spawn_interval_max_m", 70.0))

        if interval_max < interval_min:
            interval_max = interval_min

        self._next_spawn_distance = float(np.random.uniform(interval_min, interval_max))
        self._obstacle_counter = 0

    def _maybe_spawn_obstacle(self) -> None:
        if len(self._obstacle_nodes) >= int(self._obstacles_cfg.get("max_active", 3)):
            return
        if self._distance_travelled < self._next_spawn_distance:
            return

        spawned = self._spawn_obstacle_ahead()
        if not spawned:
            return

        interval_min = float(self._obstacles_cfg.get("spawn_interval_min_m", 30.0))
        interval_max = float(self._obstacles_cfg.get("spawn_interval_max_m", 70.0))

        if interval_max < interval_min:
            interval_max = interval_min

        self._next_spawn_distance = self._distance_travelled + float(
            np.random.uniform(interval_min, interval_max)
        )

    def _spawn_obstacle_ahead(self) -> bool:
        car_pos = np.array(
            self.driver_node.getField("translation").getSFVec3f(),
            dtype=np.float64,
        )
        rot = np.array(self.driver_node.getOrientation()).reshape(3, 3)
        forward_world = -rot[:, 2]
        right_world   = rot[:, 0]

        d_min = float(self._obstacles_cfg.get("distance_ahead_min", 15.0))
        d_max = float(self._obstacles_cfg.get("distance_ahead_max", 35.0))

        if d_max < d_min:
            d_max = d_min

        distance_ahead = float(np.random.uniform(d_min, d_max))
        distance_ahead = max(
            distance_ahead,
            float(self._obstacles_cfg.get("min_spawn_distance", 10.0)),
        )

        lat_max = float(self._obstacles_cfg.get("lateral_offset_max", 4.0))
        lateral_offset = float(np.random.uniform(-lat_max, lat_max))

        spawn_pos = car_pos + distance_ahead * forward_world + lateral_offset * right_world
        spawn_pos[2] = float(self._obstacles_cfg.get("barrel_z", 0.5))

        self._obstacle_counter += 1
        def_name = f"{BARREL_DEF_PREFIX}_{self._obstacle_counter}"
        vrml = self._barrel_vrml(def_name, spawn_pos)

        try:
            root_children = self.driver.getRoot().getField("children")
            root_children.importMFNodeFromString(-1, vrml)

        except Exception as exc:
            print(f"[obstacles] importMFNodeFromString failed: {exc}")
            return False

        node = self.driver.getFromDef(def_name)
        if node is None:
            print(f"[obstacles] getFromDef('{def_name}') returned None")
            return False

        self._obstacle_nodes.append({
            "node":     node,
            "def_name": def_name,
            "spawn_pos": spawn_pos.tolist(),
        })
        return True

    def _barrel_vrml(self, def_name: str, pos: np.ndarray) -> str:
        radius = float(self._obstacles_cfg.get("barrel_radius", 0.4))
        height = float(self._obstacles_cfg.get("barrel_height", 1.0))
        color  = self._obstacles_cfg.get("barrel_color", [1.0, 0.4, 0.0])
        r, g, b = (float(color[0]), float(color[1]), float(color[2]))
        return (
            f"DEF {def_name} Solid {{ "
            f"translation {pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f} "
            f"rotation 1 0 0 1.5708 "
            f'name "{def_name.lower()}" '
            f"children [ "
            f"Shape {{ "
            f"appearance PBRAppearance {{ "
            f"baseColor {r:.3f} {g:.3f} {b:.3f} "
            f"roughness 0.6 metalness 0 "
            f"}} "
            f"geometry Cylinder {{ height {height} radius {radius} }} "
            f"}} "
            f"] "
            f"boundingObject Cylinder {{ height {height} radius {radius} }} "
            f"}}"
        )

    def _recycle_passed_obstacles(self) -> None:
        if not self._obstacle_nodes:
            return

        car_pos = np.array(
            self.driver_node.getField("translation").getSFVec3f(),
            dtype=np.float64,
        )
        rot = np.array(self.driver_node.getOrientation()).reshape(3, 3)
        forward_world = -rot[:, 2]
        recycle_behind = float(self._obstacles_cfg.get("recycle_behind_distance", 3.0))

        remaining: List[Dict[str, Any]] = []
        for entry in self._obstacle_nodes:
            node = entry["node"]
            try:
                barrel_pos = np.array(
                    node.getField("translation").getSFVec3f(),
                    dtype=np.float64,
                )
            except Exception:
                continue
            forward_distance = float(np.dot(barrel_pos - car_pos, forward_world))
            if forward_distance < -recycle_behind:
                try:
                    node.remove()
                except Exception as exc:
                    print(f"[obstacles] node.remove() failed for {entry['def_name']}: {exc}")
            else:
                remaining.append(entry)
        self._obstacle_nodes = remaining

    def get_camera_image(self) -> np.ndarray:
        raw = self.camera.getImage()
        img = np.frombuffer(raw, dtype=np.uint8).reshape((self.cam_h, self.cam_w, 4))
        bgr = img[:, :, :3]
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def get_lidar_scan(self) -> np.ndarray:
        scan = np.array(self.lidar.getRangeImage(), dtype=np.float32)
        scan = np.clip(np.nan_to_num(scan, nan=30.0, posinf=30.0), 0.0, 30.0)
        scan[scan < 0.27] = 30.0  
        return scan

    def is_collision(self) -> bool:
        contacts = self.driver_node.getContactPoints()
        return len(contacts) > 0

    def get_forward_speed(self) -> float:
        v   = self.driver_node.getVelocity()[:3]
        rot = np.array(self.driver_node.getOrientation()).reshape(3, 3)
        return float(np.dot(v, rot[:, 0]))

    def get_alignment_angle(self) -> Tuple[float, bool]:
        img = self.get_camera_image()
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, w = hsv.shape[:2]

        # Save a debug frame on the very first call so you can inspect what the
        # camera actually sees and tune HSV bounds if needed.
        if not hasattr(self, "_debug_saved"):
            self._debug_saved = True
            try:
                cv2.imwrite("/tmp/debug_camera_rgb.png",
                            cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                cv2.imwrite("/tmp/debug_camera_hsv.png", hsv)
                print("[debug] Saved camera frame → /tmp/debug_camera_rgb.png")
                print("[debug] Saved HSV frame    → /tmp/debug_camera_hsv.png")
            except Exception as exc:
                print(f"[debug] Could not save frame: {exc}")

        # Use the lower 70% of the frame so the lines are captured on curves.
        roi = hsv[int(h * 0.30):, :]

        # Yellow line colour in city_level1.wbt is RoadLine color 0.85 0.75 0.3
        # which maps to HSV ≈ (25, 165, 216).  Bounds are widened for lighting.
        mask = cv2.inRange(roi, YELLOW_LO, YELLOW_HI)

        # DO NOT use MORPH_OPEN — it erodes thin lines (1-2px at 84x84) to zero.
        # MORPH_CLOSE only: dilate then erode — fills tiny gaps without destroying lines.
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        M = cv2.moments(mask)

        # Minimum 50px² — requires a real line blob, not noise or a stray pixel.
        # At 128x128 resolution a thin line across ~30% of the ROI ≈ 100-300px².
        MIN_PIX = 80.0

        # Save the mask on the first call so you can inspect detection quality:
        #   open /tmp/debug_mask_roi.png  ← white = detected yellow
        #   open /tmp/debug_camera_rgb.png ← what the camera actually sees
        if not hasattr(self, "_mask_saved"):
            self._mask_saved = True
            try:
                cv2.imwrite("/tmp/debug_mask_roi.png", mask)
            except Exception:
                pass

        if M["m00"] < MIN_PIX:
            # Full-image fallback for tight curves where line exits the bottom crop.
            mask_full = cv2.inRange(hsv, YELLOW_LO, YELLOW_HI)
            mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_CLOSE, kernel)
            M = cv2.moments(mask_full)
            if M["m00"] < MIN_PIX:
                return 1.0, False

        cx = M["m10"] / M["m00"]
        target_x = w * 0.50
        theta_norm = (cx - target_x) / (w / 2)
        theta_norm = float(np.clip(theta_norm, -1.0, 1.0))

        return theta_norm, True

    def get_min_lidar_distance(self) -> float:
        return float(self.get_lidar_scan().min())

    def set_controls(self, steering: float, throttle: float):
        # Explicit input scaling to translate normalized [-1, 1] policy ranges
        # down into mechanical device safety limits.
        angle = float(np.clip(steering, -1.0, 1.0)) * MAX_STEER
        speed = float(np.clip(throttle, -1.0, 1.0)) * MAX_SPEED
        self.driver.setSteeringAngle(angle)
        self.driver.setCruisingSpeed(speed)

    def apply_continuous(self, steering: float, throttle: float):
        """steering ∈ [-1, 1], throttle ∈ [-1, 1]"""
        self.set_controls(steering, throttle)

    def apply_discrete(self, action: int):
        # 0=hard-left  1=hard-right  2=gentle-left  3=gentle-right  4=straight
        # All actions maintain meaningful forward throttle so DQN never
        # accidentally learns to stall by choosing the old "brake" action.
        cmds = {
            0: (-1.0,  0.7),   # hard left, forward
            1: ( 1.0,  0.7),   # hard right, forward
            2: (-0.3,  1.0),   # gentle left, full throttle
            3: ( 0.3,  1.0),   # gentle right, full throttle
            4: ( 0.0,  1.0),   # straight, full throttle
        }
        self.set_controls(*cmds[int(action)])
