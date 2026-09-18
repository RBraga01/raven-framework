"""Wide additive halo approximation for Raven simulator HUD light leakage.

This intentionally does not reuse Raven's calibrated PSF. It keeps a sharp HUD
core and adds one broad, low-energy Gaussian halo in linear-light byte space
before the existing background blend.

The halos are computed at reduced resolution and scaled back up. That is not a
shortcut taken to save effort -- a wide halo is low-frequency by construction,
so there is nothing in it that a full-resolution blur could represent and a
quarter-resolution one could not. It matters because the simulator only ever
displays the newest blended frame: anything slower than the queue is discarded
before it reaches the screen, so a halo that is beautiful and late is a halo
nobody sees.

Each blurred layer is then scaled so its own peak is `strength` of the source
peak. A Gaussian blur preserves energy, so blurring a two-pixel line spreads it
until the result is about three percent of the line's brightness -- and this
whole interface is two-pixel lines. Without renormalising, a "Glow" control at
its maximum produced roughly 3.7% beside an edge where the reference look has
33%: an order of magnitude short, with no setting able to close it.

With it, the control means what its name says. Glow 20% puts a glow beside a
bright edge that is 20% as bright as the edge, whether that edge is a card
outline, a letter or a dot.

The trade this makes: the halo is scaled against the brightest thing in the
frame, so one very bright element sets the reference for everything else. In a
HUD drawn at one brightness that is what you want. It would not suit a
photograph.
"""

from __future__ import annotations

import cv2
import numpy as np

# Blur at 1/DOWNSCALE resolution. Four keeps a 54px halo smooth to the eye
# while doing about a sixteenth of the work.
DOWNSCALE = 4

# Below this radius the downscale-and-blur round trip costs more than it
# saves, and the halo is tight enough that resampling would show.
MIN_RADIUS_FOR_DOWNSCALE = 8


def _blur(linear_u8: np.ndarray, radius: int) -> np.ndarray:
    """A broad Gaussian, computed where it is cheap to compute.

    Blur in float rather than bytes, or faint wide-halo energy quantises
    to zero before it can be summed. sigmaX is given rather than a kernel
    size so the control maps straight to a spread radius.
    """
    if radius < MIN_RADIUS_FOR_DOWNSCALE:
        return cv2.GaussianBlur(
            linear_u8.astype(np.float32),
            (0, 0),
            sigmaX=float(radius),
            sigmaY=float(radius),
            borderType=cv2.BORDER_REFLECT_101,
        )

    height, width = linear_u8.shape[:2]
    small = cv2.resize(
        linear_u8,
        (max(1, width // DOWNSCALE), max(1, height // DOWNSCALE)),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)

    sigma = max(0.5, float(radius) / DOWNSCALE)
    small = cv2.GaussianBlur(
        small,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT_101,
    )

    # Linear interpolation back up is enough: what is being resampled has
    # already had everything above the sampling frequency blurred out of it.
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


def apply_waveguide_halo(
    hud_linear_u8: np.ndarray,
    radius: int,
    strength: float,
) -> np.ndarray:
    """Return sharp HUD + one broad additive halo, preserving uint8 shape/dtype.

    ``radius``/``strength`` are the fixed HALO_RADIUS/HALO_STRENGTH from
    config.json — this call is only reached when CONSIDER_WAVEGUIDE_HALO is
    on, so there is nothing here to validate on the hot path.
    """
    if strength == 0.0:
        return hud_linear_u8

    source_peak = float(hud_linear_u8.max())
    if source_peak <= 0.0:
        return hud_linear_u8

    layer = _blur(hud_linear_u8, radius)
    layer_peak = float(layer.max())
    if layer_peak <= 0.0:
        return hud_linear_u8

    # Scale so the halo's own peak is `strength` of the source's, not
    # `strength` of the energy the blur spread out. See the module
    # docstring: without this the control cannot reach the look.
    out = (
        hud_linear_u8.astype(np.float32) + layer * (source_peak / layer_peak) * strength
    )
    return np.clip(out, 0.0, 255.0).astype(np.uint8)
