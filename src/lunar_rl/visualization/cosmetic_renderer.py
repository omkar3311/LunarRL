"""Cosmetic Rocket Renderer: Enhances LunarLander visual fidelity with sci-fi spacecraft detailing, dynamic thruster plumes, searchlights, and illuminated helipads."""

import math
import random
from typing import Any, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw

VIEWPORT_W = 600
VIEWPORT_H = 400
SCALE = 30.0

# Pre-generate deterministic background celestial stars
_STAR_SEED = 1337
_RND = random.Random(_STAR_SEED)
STAR_FIELD: List[Tuple[int, int, int, Tuple[int, int, int]]] = [
    (
        _RND.randint(15, VIEWPORT_W - 15),
        _RND.randint(10, 230),
        _RND.choice([1, 1, 2]),
        _RND.choice([
            (255, 255, 255),
            (210, 235, 255),
            (255, 245, 210),
            (160, 210, 255),
        ]),
    )
    for _ in range(85)
]


def apply_rocket_theme(env: Any) -> None:
    """Configures lander Box2D bodies with high-contrast sci-fi aerospace colors."""
    unwrapped = getattr(env, "unwrapped", env)

    if hasattr(unwrapped, "lander") and unwrapped.lander is not None:
        # Hull: Pristine Titanium Ceramic White with Dark Obsidian Trim
        unwrapped.lander.color1 = (248, 250, 255)
        unwrapped.lander.color2 = (20, 30, 48)

    if hasattr(unwrapped, "legs") and len(unwrapped.legs) == 2:
        # Landing gear struts: Aerospace Gold Anodized Titanium
        for leg in unwrapped.legs:
            leg.color1 = (245, 158, 11)
            leg.color2 = (146, 64, 14)


def rotate_point(
    px: float, py: float, cx: float, cy: float, angle_rad: float
) -> Tuple[float, float]:
    """Rotates point (px, py) around center (cx, cy) by angle_rad."""
    dx, dy = px - cx, py - cy
    rx = dx * math.cos(angle_rad) - dy * math.sin(angle_rad) + cx
    ry = dx * math.sin(angle_rad) + dy * math.cos(angle_rad) + cy
    return rx, ry


def enhance_frame_visuals(
    env: Any,
    raw_frame: np.ndarray,
    action: int = 0,
    step: int = 0,
) -> np.ndarray:
    """Decorates the raw environment frame with high-fidelity rocket and space visuals.

    Features:
        - Starry deep space background field.
        - Illuminated landing pad with neon runway beacons and target crosshairs.
        - Sci-fi spacecraft detailing: cyan cockpit glass canopy, mission decal stripes, gold foil base.
        - Dynamic multi-layered plasma engine exhaust with flicker.
        - Cold-gas attitude thruster plumes.
        - Surface searchlight beams.
    """
    unwrapped = getattr(env, "unwrapped", env)
    lander = getattr(unwrapped, "lander", None)

    if lander is None:
        return raw_frame

    img = Image.fromarray(raw_frame).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # 1. Starry Night Sky
    for sx, sy, size, col in STAR_FIELD:
        # Subtle twinkle based on step
        twinkle_alpha = 180 + int(70 * math.sin((sx + step * 4) * 0.05))
        c = (col[0], col[1], col[2], max(100, min(255, twinkle_alpha)))
        if size == 1:
            draw.point((sx, sy), fill=c)
        else:
            draw.rectangle([(sx, sy), (sx + 1, sy + 1)], fill=c)

    # 2. Distant Crescent Moon/Earth in upper-right corner
    moon_cx, moon_cy, moon_r = 530, 60, 22
    draw.ellipse(
        [(moon_cx - moon_r, moon_cy - moon_r), (moon_cx + moon_r, moon_cy + moon_r)],
        fill=(14, 165, 233, 220),
    )
    draw.ellipse(
        [(moon_cx - moon_r + 6, moon_cy - moon_r - 4), (moon_cx + moon_r + 6, moon_cy + moon_r - 4)],
        fill=(0, 0, 0, 240),
    )

    # 3. Landing Pad Runway Beacons (between x=240 and x=360 at y=300)
    pad_y = 300
    # Runway approach lights
    for lx in range(240, 365, 20):
        # Pulsing green runway LED
        glow = int(180 + 75 * math.sin((lx * 0.1) + (step * 0.15)))
        draw.ellipse([(lx - 3, pad_y + 1), (lx + 3, pad_y + 7)], fill=(34, 197, 94, glow))
        draw.ellipse([(lx - 1, pad_y + 2), (lx + 1, pad_y + 4)], fill=(240, 253, 244, 255))

    # Target helipad glowing crosshair
    draw.line([(290, pad_y + 3), (310, pad_y + 3)], fill=(56, 189, 248, 200), width=2)
    draw.line([(300, pad_y - 7), (300, pad_y + 13)], fill=(56, 189, 248, 200), width=2)
    draw.ellipse([(294, pad_y - 3), (306, pad_y + 9)], outline=(56, 189, 248, 220), width=1)

    # 4. Rocket State & Coordinates
    # Box2D to Screen conversion
    pos = lander.position
    cx = float(pos[0] * SCALE)
    cy = float(VIEWPORT_H - (pos[1] * SCALE))
    # Angle is negated due to inverted screen Y coordinates
    angle = -float(lander.angle)

    def rot(px: float, py: float) -> Tuple[float, float]:
        return rotate_point(px, py, cx, cy, angle)

    # 5. Searchlight Beams (when descending towards ground)
    if cy > 160:
        beam_alpha = min(70, int(15 + (cy - 160) * 0.35))
        beam_len = min(180, 400 - cy)
        bl1 = rot(cx - 10, cy + 10)
        bl2 = rot(cx + 10, cy + 10)
        bl3 = rot(cx + 28, cy + 10 + beam_len)
        bl4 = rot(cx - 28, cy + 10 + beam_len)
        draw.polygon([bl1, bl2, bl3, bl4], fill=(186, 230, 253, beam_alpha))

    # 6. Rocket Fuselage Detailing

    # A. Sleek Cyan Pilot Viewport / Cockpit Canopy
    c1 = rot(cx - 7, cy - 8)
    c2 = rot(cx + 7, cy - 8)
    c3 = rot(cx + 6, cy - 2)
    c4 = rot(cx - 6, cy - 2)
    draw.polygon([c1, c2, c3, c4], fill=(6, 182, 212, 245), outline=(14, 116, 144, 255))
    # Canopy specular glint reflection
    g1 = rot(cx - 4, cy - 7)
    g2 = rot(cx + 1, cy - 7)
    draw.line([g1, g2], fill=(255, 255, 255, 220), width=1)

    # B. Central Racing / Mission Thermal Stripe
    s1 = rot(cx - 1.5, cy - 14)
    s2 = rot(cx + 1.5, cy - 14)
    s3 = rot(cx + 1.5, cy + 4)
    s4 = rot(cx - 1.5, cy + 4)
    draw.polygon([s1, s2, s3, s4], fill=(239, 68, 68, 230))

    # C. Gold Thermal Mylar Blanket Band (Lower Body)
    m1 = rot(cx - 13, cy + 4)
    m2 = rot(cx + 13, cy + 4)
    m3 = rot(cx + 13, cy + 9)
    m4 = rot(cx - 13, cy + 9)
    draw.polygon([m1, m2, m3, m4], fill=(217, 119, 6, 210), outline=(180, 83, 9, 255))

    # D. Thruster Nozzle Base
    n1 = rot(cx - 5, cy + 9)
    n2 = rot(cx + 5, cy + 9)
    n3 = rot(cx + 7, cy + 14)
    n4 = rot(cx - 7, cy + 14)
    draw.polygon([n1, n2, n3, n4], fill=(51, 65, 85, 255), outline=(15, 23, 42, 255))

    # 7. Dynamic Plasma Thruster Exhaust
    # Main Engine Exhaust (action == 2)
    if action == 2:
        flicker = random.uniform(0.85, 1.25)
        flame_len = 34 * flicker

        # Outer roaring plume (Solar Amber / Orange)
        fo1 = rot(cx - 7, cy + 14)
        fo2 = rot(cx + 7, cy + 14)
        fo3 = rot(cx + 10, cy + 14 + (flame_len * 0.6))
        fo4 = rot(cx, cy + 14 + flame_len)
        fo5 = rot(cx - 10, cy + 14 + (flame_len * 0.6))
        draw.polygon([fo1, fo2, fo3, fo4, fo5], fill=(249, 115, 22, 210))

        # Mid flame (Vibrant Yellow Plasma)
        fm1 = rot(cx - 5, cy + 14)
        fm2 = rot(cx + 5, cy + 14)
        fm3 = rot(cx, cy + 14 + (flame_len * 0.75))
        draw.polygon([fm1, fm2, fm3], fill=(250, 204, 21, 240))

        # Inner core (Ultra-hot White/Cyan Plasma Core)
        fi1 = rot(cx - 3, cy + 14)
        fi2 = rot(cx + 3, cy + 14)
        fi3 = rot(cx, cy + 14 + (flame_len * 0.45))
        draw.polygon([fi1, fi2, fi3], fill=(255, 255, 255, 255))

    # Left Orientation Thruster (action == 1, fires rightwards / rotates right)
    elif action == 1:
        flicker = random.uniform(0.8, 1.2)
        rj1 = rot(cx - 14, cy - 2)
        rj2 = rot(cx - 14, cy + 4)
        rj3 = rot(cx - 28 * flicker, cy + 1)
        draw.polygon([rj1, rj2, rj3], fill=(56, 189, 248, 220))
        draw.polygon([rot(cx - 14, cy), rot(cx - 14, cy + 2), rot(cx - 20 * flicker, cy + 1)], fill=(255, 255, 255, 255))

    # Right Orientation Thruster (action == 3, fires leftwards / rotates left)
    elif action == 3:
        flicker = random.uniform(0.8, 1.2)
        rj1 = rot(cx + 14, cy - 2)
        rj2 = rot(cx + 14, cy + 4)
        rj3 = rot(cx + 28 * flicker, cy + 1)
        draw.polygon([rj1, rj2, rj3], fill=(56, 189, 248, 220))
        draw.polygon([rot(cx + 14, cy), rot(cx + 14, cy + 2), rot(cx + 20 * flicker, cy + 1)], fill=(255, 255, 255, 255))

    # Composite overlay on base frame
    enhanced = Image.alpha_composite(img, overlay).convert("RGB")
    return np.array(enhanced)
