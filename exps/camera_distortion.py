"""
experiments/camera_distortion.py
=================================
Experiment 3 — Camera Input Distortion

Part A: Field of View (FOV) proxy
    The Webots camera FOV is set in the .wbt world file and cannot be changed
    at runtime through the Python API.  As a practical substitute we simulate
    a *short* FOV by cropping the image to a central horizontal band and
    *long* FOV by down-sampling + padding.  Concretely we expose three
    ``fov_mode`` strings:
        "short"   → keep only the top 50 % of rows (near-ground crop)
        "medium"  → full image (baseline, identical to no distortion)
        "long"    → zoom out: shrink image to 50 % then pad back to original size

Part B: Distortion Filters
    Applied to the uint8 RGB frame *before* it reaches the reward function or
    the neural network.  Four filters are implemented:
        "clean"      → no modification
        "fog"        → blend towards a grey haze
        "rain"       → random vertical streaks
        "low_light"  → gamma darkening

How to use
----------
1.  Monkey-patch ``WebotsEnv.get_camera_image`` before building the Gym env:

        from experiments.camera_distortion import apply_distortion, apply_fov
        import env.webots_env as we

        _orig = we.WebotsEnv.get_camera_image

        def patched_get_camera_image(self):
            img = _orig(self)
            img = apply_fov(img, mode="short")
            img = apply_distortion(img, filter_name="fog", intensity=0.5)
            return img

        we.WebotsEnv.get_camera_image = patched_get_camera_image

2.  Or call ``patch_env`` which does the above for you:

        from experiments.camera_distortion import patch_env
        patch_env(fov_mode="short", filter_name="fog", intensity=0.5)

    Calling ``unpatch_env()`` restores the original method.

Notes
-----
* All functions operate on uint8 (H, W, 3) RGB arrays and return uint8.
* Intensity parameters are floats in [0, 1] where 0 = no effect, 1 = maximum.
* These transforms are applied *after* the raw camera read, so reward
  functions and the network both see the distorted image — the intent is to
  measure robustness, not just visual fidelity.
"""
from __future__ import annotations

import numpy as np
import cv2
from typing import Optional


# ── Part A: Field-of-View proxies ──────────────────────────────────────────

def apply_fov(img: np.ndarray, mode: str = "medium") -> np.ndarray:
    """
    Simulate different camera fields of view.

    Parameters
    ----------
    img  : uint8 (H, W, 3) RGB
    mode : "short" | "medium" | "long"

    Returns
    -------
    uint8 (H, W, 3) — always the same spatial size as the input so
    downstream code (alignment angle computation, CNN input shape) stays
    unchanged.
    """
    h, w = img.shape[:2]

    if mode == "medium":
        return img.copy()

    if mode == "short":
        # Keep only the top half of rows (closest, most reactive view).
        # The bottom half is filled with the road-grey colour so the yellow-
        # line detector does not pick up artefacts.
        out = np.full_like(img, 80)          # neutral grey
        crop_h = max(1, h // 2)
        out[:crop_h, :] = img[:crop_h, :]
        return out

    if mode == "long":
        # Shrink to 50 % → pad back to (H, W) with black borders.
        # This simulates a wider / more distant view with more background noise.
        small_h, small_w = max(1, h // 2), max(1, w // 2)
        small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
        out = np.zeros_like(img)
        pad_top  = (h - small_h) // 2
        pad_left = (w - small_w) // 2
        out[pad_top:pad_top + small_h, pad_left:pad_left + small_w] = small
        return out

    raise ValueError(f"Unknown fov mode: '{mode}'. Choose 'short', 'medium', or 'long'.")


# ── Part B: Distortion filters ─────────────────────────────────────────────

def apply_distortion(
    img: np.ndarray,
    filter_name: str = "clean",
    intensity: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Apply a visual distortion filter to a uint8 RGB image.

    Parameters
    ----------
    img         : uint8 (H, W, 3)
    filter_name : "clean" | "fog" | "rain" | "low_light"
    intensity   : float in [0, 1] controlling strength of the effect
    rng         : optional numpy Generator for reproducibility

    Returns
    -------
    uint8 (H, W, 3)
    """
    if rng is None:
        rng = np.random.default_rng()

    intensity = float(np.clip(intensity, 0.0, 1.0))

    if filter_name == "clean":
        return img.copy()
    elif filter_name == "fog":
        return _apply_fog(img, intensity)
    elif filter_name == "rain":
        return _apply_rain(img, intensity, rng)
    elif filter_name == "low_light":
        return _apply_low_light(img, intensity)
    else:
        raise ValueError(
            f"Unknown filter '{filter_name}'. Choose: clean, fog, rain, low_light."
        )


def _apply_fog(img: np.ndarray, intensity: float) -> np.ndarray:
    """
    Blend the image towards a uniform grey haze.

    intensity=0 → no change
    intensity=1 → fully grey (200, 200, 200)

    The grey target mimics a bright overcast sky reflecting off road surfaces.
    A Gaussian blur is also applied proportionally so distant objects lose
    edge definition, matching real fog behaviour.
    """
    fog_colour = np.full_like(img, 200, dtype=np.float32)
    blended = (
        (1.0 - intensity) * img.astype(np.float32)
        + intensity * fog_colour
    )
    # Slight blur to wash out edges
    ksize = max(1, int(intensity * 5)) * 2 + 1   # must be odd
    blurred = cv2.GaussianBlur(blended, (ksize, ksize), sigmaX=intensity * 2.0)
    return np.clip(blurred, 0, 255).astype(np.uint8)


def _apply_rain(
    img: np.ndarray, intensity: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Add semi-transparent vertical streaks to simulate rainfall.

    The number and brightness of streaks scale with intensity.
    """
    out = img.copy().astype(np.float32)
    h, w = img.shape[:2]
    n_streaks = max(1, int(intensity * w * 0.3))   # up to 30 % of columns

    # Each streak: a thin vertical band with a random start row and length
    x_positions = rng.integers(0, w, size=n_streaks)
    streak_lens = rng.integers(h // 4, h, size=n_streaks)
    start_rows  = rng.integers(0, h // 2, size=n_streaks)

    streak_bright = 200.0 + intensity * 55.0  # near-white
    alpha = 0.4 + intensity * 0.4             # blend weight

    for x, length, y0 in zip(x_positions, streak_lens, start_rows):
        y1 = min(h, y0 + length)
        # Narrow streak width (1-2 px) for realism
        x1 = min(w, x + rng.integers(1, 3))
        out[y0:y1, x:x1] = (
            (1.0 - alpha) * out[y0:y1, x:x1]
            + alpha * streak_bright
        )

    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_low_light(img: np.ndarray, intensity: float) -> np.ndarray:
    """
    Darken the image using gamma compression.

    intensity=0 → gamma=1 (no change)
    intensity=1 → gamma≈4 (very dark)

    A small amount of Gaussian noise is added to simulate sensor noise in
    low-light conditions.
    """
    gamma = 1.0 + intensity * 3.0          # range [1, 4]
    lut = np.array(
        [((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8
    )
    darkened = cv2.LUT(img, lut)

    # Additive Gaussian noise (σ scales with how dark the image is)
    noise_sigma = intensity * 15.0
    noise = np.random.normal(0, noise_sigma, img.shape).astype(np.float32)
    noisy = np.clip(darkened.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


# ── Monkey-patching helpers ────────────────────────────────────────────────

_original_get_camera_image = None


def patch_env(
    fov_mode: str = "medium",
    filter_name: str = "clean",
    intensity: float = 0.5,
) -> None:
    """
    Monkey-patch ``WebotsEnv.get_camera_image`` so every camera read goes
    through the chosen FOV and distortion transform.

    Safe to call multiple times — each call replaces the previous patch.
    Call ``unpatch_env()`` to restore the original.

    Parameters
    ----------
    fov_mode    : "short" | "medium" | "long"
    filter_name : "clean" | "fog" | "rain" | "low_light"
    intensity   : float in [0, 1]

    Example
    -------
    >>> patch_env(fov_mode="short", filter_name="fog", intensity=0.6)
    >>> env = WebotsLaneEnv(cfg)   # all camera reads are now distorted
    """
    import env.webots_env as _we

    global _original_get_camera_image

    # Store the original only on the first patch so unpatch works correctly
    if _original_get_camera_image is None:
        _original_get_camera_image = _we.WebotsEnv.get_camera_image

    _fov        = fov_mode
    _filter     = filter_name
    _intensity  = intensity
    _orig       = _original_get_camera_image

    def _patched(self):
        img = _orig(self)
        img = apply_fov(img, mode=_fov)
        img = apply_distortion(img, filter_name=_filter, intensity=_intensity)
        return img

    _we.WebotsEnv.get_camera_image = _patched
    print(
        f"[camera_distortion] patched: fov={fov_mode}  "
        f"filter={filter_name}  intensity={intensity:.2f}"
    )


def unpatch_env() -> None:
    """Restore the original ``WebotsEnv.get_camera_image``."""
    import env.webots_env as _we

    global _original_get_camera_image
    if _original_get_camera_image is not None:
        _we.WebotsEnv.get_camera_image = _original_get_camera_image
        _original_get_camera_image = None
        print("[camera_distortion] unpatched: restored original get_camera_image")
    else:
        print("[camera_distortion] nothing to unpatch.")
