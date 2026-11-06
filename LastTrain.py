#!/usr/bin/env python3

import argparse
import math
import sys

import numpy as np

W, H = 1280, 720
SEED = 11

# ----------------------------------------------------------------------------
# 1. LAYOUT  (all in pixels, y points down)
# ----------------------------------------------------------------------------
WIN = (640.0, 285.0, 420.0, 205.0, 78.0)          # cx, cy, hx, hy, corner r
FRAME_T = 36.0
OUTER = (WIN[0], WIN[1], WIN[2] + FRAME_T, WIN[3] + FRAME_T, WIN[4] + FRAME_T)
RAIL_Y0, RAIL_Y1, VENT_Y1 = 12.0, 30.0, 42.0
SILL_Y0, SILL_Y1 = 526.0, 548.0
FLOOR_Y = 650.0
LAMPS = np.array([[96.0, 215.0], [1184.0, 215.0]])
HANDLE = (640.0, 563.0, 80.0, 6.0, 6.0)
GRILLE = (640.0, 687.0, 100.0, 27.0, 5.0)


def _seat(mirror):
    mx = (lambda v: W - v) if mirror else (lambda v: v)
    back = (mx(50.0), 570.0, 118.0, 190.0, 58.0)
    cush = (mx(110.0), 700.0, 190.0, 80.0, 52.0)
    arm = (mx(205.0), 590.0, 78.0, 15.0, 15.0)
    post = (mx(268.0), 630.0, 4.0, 26.0, 3.0)
    return back, cush, arm, post


SEATS = [_seat(False), _seat(True)]

(C_CEIL, C_RAIL, C_VENT, C_WALL, C_SILL, C_WAIN, C_FLOOR, C_FRAME, C_GLASS,
 C_HANDLE, C_GRILLE, C_BACK, C_CUSH, C_ARM, C_POST, C_LAMP, C_PLATE) = range(17)
NC = 17

# albedo = the surface's own colour before light hits it
ALB = np.zeros((NC, 3))
ALB[C_CEIL] = (0.20, 0.12, 0.08)
ALB[C_RAIL] = (0.82, 0.64, 0.30)
ALB[C_VENT] = (0.20, 0.13, 0.09)
ALB[C_WALL] = (0.52, 0.28, 0.13)
ALB[C_SILL] = (0.48, 0.25, 0.11)
ALB[C_WAIN] = (0.62, 0.34, 0.16)
ALB[C_FLOOR] = (0.26, 0.15, 0.10)
ALB[C_FRAME] = (0.60, 0.32, 0.14)
ALB[C_HANDLE] = (0.84, 0.66, 0.32)
ALB[C_GRILLE] = (0.13, 0.10, 0.09)
ALB[C_BACK] = (0.36, 0.60, 0.43)      # muted green
ALB[C_CUSH] = (0.40, 0.64, 0.46)
ALB[C_ARM] = (0.40, 0.20, 0.10)
ALB[C_POST] = (0.82, 0.64, 0.30)
ALB[C_LAMP] = (1.00, 0.66, 0.22)
ALB[C_PLATE] = (0.82, 0.62, 0.28)

# dot spacing of the BASE layer per object (0 = no base dots, i.e. glass)
CELL = np.zeros(NC)
for _c, _v in {C_CEIL: 6.2, C_RAIL: 3.6, C_VENT: 4.0, C_WALL: 5.0, C_SILL: 4.2,
               C_WAIN: 4.6, C_FLOOR: 5.6, C_FRAME: 3.8, C_HANDLE: 3.2,
               C_GRILLE: 3.6, C_BACK: 6.4, C_CUSH: 6.8, C_ARM: 4.2,
               C_POST: 3.0, C_LAMP: 3.0, C_PLATE: 3.0}.items():
    CELL[_c] = _v

# chance that a candidate DETAIL dot is kept, per object
DETAIL = np.zeros(NC)
for _c, _v in {C_CEIL: 0.10, C_RAIL: 0.20, C_VENT: 0.10, C_WALL: 0.30,
               C_SILL: 0.30, C_WAIN: 0.26, C_FLOOR: 0.24, C_FRAME: 0.24,
               C_HANDLE: 0.10, C_GRILLE: 0.0, C_BACK: 0.24, C_CUSH: 0.26,
               C_ARM: 0.28, C_POST: 0.0, C_LAMP: 0.0, C_PLATE: 0.0}.items():
    DETAIL[_c] = _v

DENS = np.ones(NC)                     # base-layer density (<1 leaves gaps)
DENS[C_CEIL] = 0.75
DENS[C_FLOOR] = 0.92


def allow(*ids):
    m = np.zeros(NC, bool)
    m[list(ids)] = True
    return m


NONSEAT = allow(C_CEIL, C_RAIL, C_VENT, C_WALL, C_SILL, C_WAIN, C_FLOOR,
                C_HANDLE, C_GRILLE)

# ----------------------------------------------------------------------------
# 2. SMALL MATH HELPERS
# ----------------------------------------------------------------------------


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def hash1(i):
    v = np.sin(np.asarray(i, float) * 12.9898 + 4.1) * 43758.5453
    return v - np.floor(v)


def sdr(x, y, rect):
    """signed distance to a rounded rectangle (negative inside)"""
    cx, cy, hx, hy, r = rect
    qx = np.abs(x - cx) - (hx - r)
    qy = np.abs(y - cy) - (hy - r)
    return (np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
            + np.minimum(np.maximum(qx, qy), 0) - r)


def classify(x, y):
    """which object is the front-most thing at each point (painter's order)"""
    cls = np.full(x.shape, C_WALL, np.int8)
    cls[y < RAIL_Y0] = C_CEIL
    cls[(y >= RAIL_Y0) & (y < RAIL_Y1)] = C_RAIL
    cls[(y >= RAIL_Y1) & (y < VENT_Y1)] = C_VENT
    cls[(y >= SILL_Y0) & (y < SILL_Y1)] = C_SILL
    cls[(y >= SILL_Y1) & (y < FLOOR_Y)] = C_WAIN
    cls[y >= FLOOR_Y] = C_FLOOR
    cls[sdr(x, y, OUTER) < 0] = C_FRAME
    cls[sdr(x, y, WIN) < 0] = C_GLASS
    cls[sdr(x, y, HANDLE) < 0] = C_HANDLE
    cls[sdr(x, y, GRILLE) < 0] = C_GRILLE
    for s in SEATS:
        cls[sdr(x, y, s[0]) < 0] = C_BACK
    for s in SEATS:
        cls[sdr(x, y, s[1]) < 0] = C_CUSH
    for s in SEATS:
        cls[sdr(x, y, s[2]) < 0] = C_ARM
    for s in SEATS:
        cls[sdr(x, y, s[3]) < 0] = C_POST
    for lx, ly in LAMPS:
        d = np.hypot(x - lx, y - ly)
        cls[d < 21] = C_PLATE
        cls[d < 15] = C_LAMP
    return cls


# ----------------------------------------------------------------------------
# 3. LIGHT
# ----------------------------------------------------------------------------
AMB_COL = np.array([0.50, 0.62, 0.95])     # night ambient is cool blue
WARM = np.array([1.00, 0.72, 0.40])        # lamps are amber
COOLC = np.array([0.10, 0.36, 0.72])       # light spilling from the window
AMBIENT = 0.46


def lighting(x, y):
    Il = np.zeros((2,) + x.shape)
    for i, (lx, ly) in enumerate(LAMPS):
        d2 = (x - lx) ** 2 + (y - ly) ** 2
        Il[i] = 1.25 / (1.0 + d2 / 165.0 ** 2)
    d2 = (x - 640.0) ** 2 + (y - 20.0) ** 2
    Ir = 0.30 / (1.0 + d2 / 340.0 ** 2)
    I = Il.sum(0) + Ir
    L = AMBIENT * AMB_COL + I[..., None] * WARM
    glow = np.exp(-np.maximum(sdr(x, y, OUTER), 0) / 75.0) * 0.55
    L = L + glow[..., None] * COOLC
    flick = Il.sum(0) / (AMBIENT + I)
    side = Il[1] / (Il[0] + Il[1] + 1e-6)
    return L, AMBIENT + I, flick, side


def surface_factor(x, y, cls):
    """per-object shading: slat variation, seat roundness, contact shadows"""
    f = np.ones(x.shape)
    m = cls == C_WAIN
    if m.any():
        f[m] = ((0.78 + 0.40 * hash1(np.floor(x[m] / 15.0)))
                * (1 - 0.40 * np.clip((y[m] - SILL_Y1) / (FLOOR_Y - SILL_Y1), 0, 1)))
    m = cls == C_FLOOR
    if m.any():
        f[m] = 0.70 + 0.45 * hash1(np.floor(y[m] / 26.0) + 31)
    for cid, key, sc in ((C_BACK, 0, 40.0), (C_CUSH, 1, 40.0), (C_ARM, 2, 12.0)):
        m = cls == cid
        if m.any():
            depth = np.full(m.sum(), -1e9)
            for s in SEATS:
                depth = np.maximum(depth, -sdr(x[m], y[m], s[key]))
            f[m] *= 0.50 + 0.50 * smoothstep(0, sc, depth)
    m = (cls == C_BACK) | (cls == C_CUSH)
    if m.any():
        sda = np.full(m.sum(), 1e9)
        for s in SEATS:
            sda = np.minimum(sda, sdr(x[m], y[m], s[2]))
        f[m] *= np.where(y[m] > 589, 0.45 + 0.55 * np.clip(sda / 20.0, 0, 1), 1.0)
    m = (cls == C_WALL) | (cls == C_SILL) | (cls == C_WAIN) | (cls == C_VENT)
    if m.any():
        f[m] *= 0.55 + 0.45 * np.clip(sdr(x[m], y[m], OUTER) / 28.0, 0, 1)
    return f


def shade_rgb(x, y, cls):
    L, bright, flick, side = lighting(x, y)
    rgb = ALB[cls] * surface_factor(x, y, cls)[:, None] * L
    em = cls == C_LAMP
    rgb[em] = ALB[C_LAMP]
    return rgb, bright, flick, side


# ----------------------------------------------------------------------------
# 4. DOT TOOLKIT
# ----------------------------------------------------------------------------
DOT_COOL = np.array([0.10, 0.34, 0.46])
DOT_GOLD = np.array([1.00, 0.72, 0.24])
DOT_VIOL = np.array([0.30, 0.20, 0.50])


def pointillize(rgb, bright, rng, amount=1.0, cool=DOT_COOL, gold=DOT_GOLD, viol=DOT_VIOL):
    """optical mixing: instead of blending colours we flip some dots to a neighbour hue"""
    n = len(rgb)
    out = rgb * (1.0 + rng.normal(0, 0.07, (n, 3)))
    bc = np.clip(np.broadcast_to(np.asarray(bright, float), (n,)) / 1.3, 0, 1)
    sh = 1.0 - bc
    r = rng.random(n) / max(amount, 1e-3)
    a = np.where(r < 0.13, 0.22 + 0.35 * sh, 0.0)[:, None]
    out = out * (1 - a) + cool * (0.35 + 0.9 * bc)[:, None] * a
    a = np.where((r >= 0.13) & (r < 0.25), 0.18 + 0.30 * bc, 0.0)[:, None]
    out = out * (1 - a) + gold * (0.35 + 0.8 * bc)[:, None] * a
    a = np.where((r >= 0.25) & (r < 0.29), 0.20 + 0.30 * sh, 0.0)[:, None]
    out = out * (1 - a) + viol * (0.35 + 0.9 * bc)[:, None] * a
    return np.clip(out, 0, 1)


def jitter_grid(cell, rng, bbox=(0, 0, W, H), jit=0.48):
    """hex-ish grid with random jitter = irregular but evenly spread dots"""
    x0, y0, x1, y1 = bbox
    nx = int((x1 - x0) / cell) + 3
    ny = int((y1 - y0) / (cell * 0.9)) + 3
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny))
    px = (gx + 0.5 * (gy % 2)) * cell + x0 - cell
    py = gy * cell * 0.9 + y0 - cell
    px = px + rng.uniform(-jit, jit, px.shape) * cell
    py = py + rng.uniform(-jit, jit, py.shape) * cell
    px, py = px.ravel(), py.ravel()
    m = (px >= x0 - 1) & (px < x1 + 1) & (py >= y0 - 1) & (py < y1 + 1)
    return px[m], py[m]


def rrect_path(cx, cy, hx, hy, r, spacing, rng, jit=0.3):
    """evenly spaced points + outward normals along a rounded rectangle outline"""
    r = max(min(r, hx, hy), 0.01)
    step = 0.5
    P, N = [], []

    def line(x0, y0, x1, y1, nx, ny):
        n = max(int(math.hypot(x1 - x0, y1 - y0) / step), 1)
        t = np.linspace(0, 1, n, endpoint=False)
        P.append(np.column_stack([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t]))
        N.append(np.tile([nx, ny], (n, 1)))

    def arc(ax, ay, a0, a1):
        n = max(int(r * abs(math.radians(a1 - a0)) / step), 1)
        a = np.radians(np.linspace(a0, a1, n, endpoint=False))
        c, s = np.cos(a), np.sin(a)
        P.append(np.column_stack([ax + r * c, ay + r * s]))
        N.append(np.column_stack([c, s]))

    line(cx - hx + r, cy - hy, cx + hx - r, cy - hy, 0, -1)
    arc(cx + hx - r, cy - hy + r, -90, 0)
    line(cx + hx, cy - hy + r, cx + hx, cy + hy - r, 1, 0)
    arc(cx + hx - r, cy + hy - r, 0, 90)
    line(cx + hx - r, cy + hy, cx - hx + r, cy + hy, 0, 1)
    arc(cx - hx + r, cy + hy - r, 90, 180)
    line(cx - hx, cy + hy - r, cx - hx, cy - hy + r, -1, 0)
    arc(cx - hx + r, cy - hy + r, 180, 270)
    P, N = np.vstack(P), np.vstack(N)
    k = max(int(spacing / step), 1)
    idx = np.arange(0, len(P), k) + (rng.uniform(-jit, jit, len(np.arange(0, len(P), k))) * k).astype(int)
    idx = np.clip(idx, 0, len(P) - 1)
    pts = P[idx] + N[idx] * rng.uniform(-0.6, 0.6, (len(idx), 1))
    return pts, N[idx]


def seg_dots(rng, p0, p1, spacing, jit=0.3, wobble=0.35):
    n = max(int(math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / spacing), 1)
    t = np.clip((np.arange(n) + rng.uniform(-jit, jit, n)) / n, 0, 1)
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    return p0 + (p1 - p0) * t[:, None] + rng.normal(0, wobble, (n, 2))


class Emitter:
    """collects dots; columns = x y radius r g b a fx0 fx1 fx2"""

    def __init__(self, rng):
        self.rng = rng
        self.rows = []

    def add(self, x, y, rgb, radius, flick=0.0, side=0.0, alpha=1.0, phase=None):
        x = np.atleast_1d(np.asarray(x, np.float32))
        n = len(x)
        if n == 0:
            return

        def b(v):
            return np.broadcast_to(np.asarray(v, np.float32), (n,)).copy()
        rgb = np.broadcast_to(np.asarray(rgb, np.float32), (n, 3))
        ph = self.rng.uniform(0, 6.2832, n) if phase is None else b(phase)
        self.rows.append(np.column_stack([x, b(y), b(radius), np.clip(rgb, 0, 1),
                                          b(alpha), b(flick), ph, b(side)]).astype(np.float32))

    def array(self):
        return np.vstack(self.rows) if self.rows else np.zeros((0, 10), np.float32)


def contour(em, rng, rect, d, spacing, radius, allowed, bf=1.0, albedo=None,
            test_off=0.0, lit=False, amount=0.5, flick_boost=1.0):
    """dots laid along an outline, offset by d (negative = inside)"""
    cx, cy, hx, hy, r = rect
    pts, nrm = rrect_path(cx, cy, hx + d, hy + d, r + d, spacing, rng)
    x, y = pts[:, 0], pts[:, 1]
    cls = classify(x - nrm[:, 0] * test_off, y - nrm[:, 1] * test_off)
    ok = allowed[cls]
    n = len(x)
    bfv = np.full(n, float(bf))
    if lit:  # rim light: only edges that face the nearest lamp
        dl = LAMPS[None, :, :] - pts[:, None, :]
        dist = np.hypot(dl[..., 0], dl[..., 1]) + 1e-6
        j = np.argmin(dist, axis=1)
        dv = dl[np.arange(n), j] / dist[np.arange(n), j][:, None]
        dv = dv * 0.7 + np.array([0.0, -0.3])
        w = np.clip((nrm * dv).sum(1), 0, 1)
        ok &= rng.random(n) < (0.12 + 0.88 * w ** 0.7)
        bfv = bf * (0.7 + 1.0 * w)
    x, y, cls, bfv = x[ok], y[ok], cls[ok], bfv[ok]
    if len(x) == 0:
        return
    L, bright, flick, side = lighting(x, y)
    alb = ALB[cls] * surface_factor(x, y, cls)[:, None] if albedo is None else np.asarray(albedo)
    rgb = pointillize(alb * L * np.atleast_1d(bfv)[:, None] if np.ndim(alb) > 1 else alb * L * bfv[:, None],
                      bright, rng, amount=amount)
    rad = radius * rng.uniform(0.8, 1.2, len(x))
    em.add(x, y, rgb, rad, flick * flick_boost, side)


def strands(em, rng, n, bbox, lenr, vertical, classes, spacing=3.0, albedo_cls=None):
    """short wood-grain strokes made of dots, dark and light alternating"""
    x0, x1, y0, y1 = bbox
    for _ in range(n):
        x = rng.uniform(x0, x1)
        y = rng.uniform(y0, y1)
        ln = rng.uniform(*lenr)
        p1 = (x, y + ln) if vertical else (x + ln, y)
        pts = seg_dots(rng, (x, y), p1, spacing)
        cls = classify(pts[:, 0], pts[:, 1])
        ok = classes[cls]
        pts, cls = pts[ok], cls[ok]
        if len(pts) == 0:
            continue
        L, bright, flick, side = lighting(pts[:, 0], pts[:, 1])
        bf = 0.42 if rng.random() < 0.5 else 1.55
        rgb = ALB[cls] * surface_factor(pts[:, 0], pts[:, 1], cls)[:, None] * L * bf
        rgb = pointillize(rgb, bright, rng, amount=0.5)
        em.add(pts[:, 0], pts[:, 1], rgb, rng.uniform(1.0, 1.7, len(pts)), flick, side)


def radial_dots(rng, n, cx, cy, r0, r1, scale):
    r = r0 + rng.exponential(scale, n)
    r = r[r < r1]
    a = rng.uniform(0, 2 * math.pi, len(r))
    return cx + r * np.cos(a), cy + r * np.sin(a), r


# ----------------------------------------------------------------------------
# 5. BUILD THE INTERIOR
# ----------------------------------------------------------------------------
def build_interior(rng):
    # ---- layer 1: BASE dots (shuffled so overlaps look like painted mosaic)
    base = Emitter(rng)
    for cell in np.unique(CELL[CELL > 0]):
        gx, gy = jitter_grid(float(cell), rng)
        cls = classify(gx, gy)
        k = np.isclose(CELL[cls], cell)
        gx, gy, cls = gx[k], gy[k], cls[k]
        rgb, bright, flick, side = shade_rgb(gx, gy, cls)
        k = rng.random(len(gx)) < DENS[cls] * (0.82 + 0.18 * np.clip(bright, 0, 1))
        gx, gy, cls, rgb, bright, flick, side = (a[k] for a in (gx, gy, cls, rgb, bright, flick, side))
        rgb = pointillize(rgb, bright, rng)
        rad = cell * 0.67 * rng.uniform(0.82, 1.2, len(gx))
        flick = np.where(cls == C_LAMP, 1.0, flick)
        base.add(gx, gy, rgb, rad, flick, side)
    base_arr = base.array()
    base_arr = base_arr[rng.permutation(len(base_arr))]

    em = Emitter(rng)

    # ---- layer 2: DETAIL dots (grain, weave, sparkle)
    gx, gy = jitter_grid(2.7, rng)
    cls = classify(gx, gy)
    k = rng.random(len(gx)) < DETAIL[cls]
    gx, gy, cls = gx[k], gy[k], cls[k]
    rgb, bright, flick, side = shade_rgb(gx, gy, cls)
    bf = np.clip(np.exp(rng.normal(0, 0.40, len(gx))), 0.45, 2.1)
    rgb = pointillize(rgb * bf[:, None], bright * bf, rng)
    em.add(gx, gy, rgb, rng.uniform(0.8, 1.6, len(gx)), flick, side)

    # wood grain strokes
    strands(em, rng, 300, (0, W, 48, 520), (25, 90), True, allow(C_WALL, C_CEIL))
    strands(em, rng, 420, (0, W, SILL_Y1, FLOOR_Y), (10, 45), True, allow(C_WAIN))
    strands(em, rng, 70, (0, W, SILL_Y0, SILL_Y1), (30, 120), False, allow(C_SILL))
    strands(em, rng, 220, (0, W, FLOOR_Y, H), (40, 170), False, allow(C_FLOOR))
    for s in SEATS:
        cx, cy, hx, hy, _ = s[2]
        strands(em, rng, 14, (cx - hx, cx + hx, cy - hy, cy + hy), (30, 110), False, allow(C_ARM), 2.6)

    # ---- layer 3: STRUCTURE dots
    # wainscot slat grooves + rib highlights
    for k in range(0, 86):
        xk = k * 15.0
        for off, bf_, sp, rr in ((0.0, 0.22, 3.2, 1.6), (2.8, 1.6, 4.2, 1.3)):
            pts = seg_dots(rng, (xk + off, SILL_Y1 + 1), (xk + off, FLOOR_Y - 1), sp)
            cls = classify(pts[:, 0], pts[:, 1])
            pts = pts[cls == C_WAIN]
            if len(pts) == 0:
                continue
            L, bright, flick, side = lighting(pts[:, 0], pts[:, 1])
            alb = ALB[C_WAIN] * surface_factor(pts[:, 0], pts[:, 1], np.full(len(pts), C_WAIN, np.int8))[:, None]
            em.add(pts[:, 0], pts[:, 1], pointillize(alb * L * bf_, bright, rng, 0.4),
                   rng.uniform(rr * 0.8, rr * 1.2, len(pts)), flick, side)

    def hline(y, x0, x1, bf_, sp, rr, cid, amount=0.4):
        pts = seg_dots(rng, (x0, y), (x1, y), sp)
        cls = classify(pts[:, 0], pts[:, 1])
        pts, cls = pts[cls == cid], cls[cls == cid]
        if len(pts) == 0:
            return
        L, bright, flick, side = lighting(pts[:, 0], pts[:, 1])
        alb = ALB[cls] * surface_factor(pts[:, 0], pts[:, 1], cls)[:, None]
        em.add(pts[:, 0], pts[:, 1], pointillize(alb * L * bf_, bright, rng, amount),
               rng.uniform(rr * 0.8, rr * 1.2, len(pts)), flick, side)

    hline(SILL_Y0 + 3.0, 0, W, 1.7, 3.4, 1.4, C_SILL)     # sill top highlight
    hline(SILL_Y1 - 2.0, 0, W, 0.35, 3.2, 1.7, C_SILL)    # sill under-shadow
    hline(SILL_Y1 + 3.0, 0, W, 0.30, 3.4, 1.6, C_WAIN)    # shadow below sill
    hline(RAIL_Y0 + 2.5, 20, W - 20, 1.7, 3.2, 1.3, C_RAIL)
    hline(RAIL_Y1 - 3.0, 20, W - 20, 0.45, 3.2, 1.5, C_RAIL)
    hline(VENT_Y1 + 1.5, 0, W, 0.30, 3.2, 1.6, C_WALL)
    for yy in np.arange(FLOOR_Y + 14, H, 26.0):               # floor planks
        hline(yy, 0, W, 0.35, 3.4, 1.6, C_FLOOR)
        hline(yy + 2.8, 0, W, 1.4, 4.6, 1.2, C_FLOOR)

    # luggage-rail vent: perforated pattern
    vx, vy = np.meshgrid(np.arange(24, W - 20, 7.0), np.arange(RAIL_Y1 + 3, VENT_Y1 - 1, 4.4))
    vx = vx + 3.5 * (np.round((vy - RAIL_Y1) / 4.4) % 2)
    vx, vy = vx.ravel() + rng.normal(0, 0.3, vx.size), vy.ravel() + rng.normal(0, 0.3, vx.size)
    cls = classify(vx, vy)
    k = cls == C_VENT
    L, bright, flick, side = lighting(vx[k], vy[k])
    em.add(vx[k], vy[k], pointillize(ALB[C_RAIL] * L * 0.9, bright[k] if False else bright, rng, 0.3),
           1.5, flick, side)

    # wall panel outlines (the lamps sit inside them)
    for mir in (False, True):
        x0, x1 = (18.0, 166.0)
        if mir:
            x0, x1 = W - x1, W - x0
        panel = ((x0 + x1) / 2, 215.0, (x1 - x0) / 2, 150.0, 8.0)
        contour(em, rng, panel, 0.0, 3.6, 1.7, allow(C_WALL), bf=0.30)
        contour(em, rng, panel, -3.0, 4.6, 1.3, allow(C_WALL), bf=1.5, lit=True)
        seam_x = 48.0 if not mir else W - 48.0
        pts = seg_dots(rng, (seam_x, VENT_Y1 + 4), (seam_x, SILL_Y0 - 2), 3.4)
        cls = classify(pts[:, 0], pts[:, 1])
        pts = pts[cls == C_WALL]

    # window frame: rings of dots following the rounded outline
    rings = [(-2.6, 3.4, 1.7, allow(C_GLASS), 0.18, (0.12, 0.08, 0.10), 0.4),
             (2.2, 4.0, 2.3, allow(C_FRAME), 1.35, (0.92, 0.72, 0.34), 0.4),
             (5.8, 3.4, 1.5, allow(C_FRAME), 0.35, None, 0.4),
             (11.0, 5.0, 2.7, allow(C_FRAME), 1.00, None, 0.7),
             (17.0, 5.0, 2.7, allow(C_FRAME), 1.15, None, 0.7),
             (23.0, 5.0, 2.7, allow(C_FRAME), 0.92, None, 0.7),
             (29.0, 5.0, 2.7, allow(C_FRAME), 1.08, None, 0.7),
             (33.8, 3.8, 2.0, allow(C_FRAME), 1.40, (0.88, 0.58, 0.28), 0.4),
             (37.8, 4.0, 2.4, allow(C_WALL, C_SILL, C_VENT), 0.28, (0.22, 0.12, 0.08), 0.4)]
    for d, sp, rr, al, bf_, alb, amt in rings:
        contour(em, rng, WIN, d, sp, rr, al, bf=bf_, albedo=alb, amount=amt)
    contour(em, rng, WIN, 4.5, 5.5, 1.5, allow(C_FRAME), bf=1.7, lit=True)

    # window handle bar + shadow
    contour(em, rng, HANDLE, -1.6, 3.0, 1.4, allow(C_HANDLE), bf=1.9, lit=False)
    contour(em, rng, HANDLE, 2.5, 3.6, 1.6, allow(C_WAIN, C_SILL), bf=0.22, test_off=0)

    # floor heater grille
    contour(em, rng, GRILLE, -1.5, 3.4, 1.8, allow(C_GRILLE), bf=1.0, albedo=(0.75, 0.58, 0.28), amount=0.3)
    for i in range(6):
        yy = 669.0 + 8.2 * i
        pts = seg_dots(rng, (553, yy), (727, yy), 3.0)
        em.add(pts[:, 0], pts[:, 1], np.tile([0.03, 0.02, 0.03], (len(pts), 1)), rng.uniform(1.7, 2.3, len(pts)))
        warm = rng.random(len(pts)) < 0.40
        em.add(pts[warm, 0], pts[warm, 1] + 3.5, np.tile([0.95, 0.42, 0.12], (warm.sum(), 1)),
               rng.uniform(1.0, 1.6, warm.sum()), 0.8, 0.5, 0.85)

    # seats: piping, stitching, rim light, contact shadows
    for s in SEATS:
        back, cush, arm, post = s
        mirror = back[0] > W / 2
        contour(em, rng, back, 2.2, 4.0, 2.2, NONSEAT, bf=0.22, albedo=(0.10, 0.08, 0.10))
        contour(em, rng, back, -2.2, 3.5, 1.7, allow(C_BACK), bf=1.7, lit=True, test_off=0)
        contour(em, rng, back, -8.5, 3.4, 1.5, allow(C_BACK), bf=0.5, test_off=0)
        contour(em, rng, back, -11.0, 7.5, 1.1, allow(C_BACK), bf=1.0, albedo=(0.85, 0.80, 0.58), amount=0.2)
        contour(em, rng, cush, 2.4, 4.0, 2.3, NONSEAT | allow(C_BACK), bf=0.22, albedo=(0.08, 0.07, 0.09))
        contour(em, rng, cush, -2.2, 3.5, 1.8, allow(C_CUSH), bf=1.8, lit=True, test_off=0)
        contour(em, rng, cush, -9.0, 3.4, 1.5, allow(C_CUSH), bf=0.5, test_off=0)
        contour(em, rng, cush, -12.0, 7.5, 1.1, allow(C_CUSH), bf=1.0, albedo=(0.85, 0.80, 0.58), amount=0.2)
        contour(em, rng, arm, 3.0, 3.4, 2.0, allow(C_BACK, C_CUSH, C_WALL, C_WAIN), bf=0.20, albedo=(0.07, 0.05, 0.07))
        contour(em, rng, arm, -2.0, 3.0, 1.5, allow(C_ARM), bf=2.0, lit=True)
        # vertical seam + headrest seam (dark line + highlight line)
        sx = 100.0 if not mirror else W - 100.0
        for off, bf_, sp, rr in ((0.0, 0.35, 3.2, 1.5), (3.0 if not mirror else -3.0, 1.7, 5.0, 1.2)):
            pts = seg_dots(rng, (sx + off, 392), (sx + off, 618), sp, wobble=0.5)
            cls = classify(pts[:, 0], pts[:, 1])
            pts = pts[cls == C_BACK]
            L, bright, flick, side = lighting(pts[:, 0], pts[:, 1])
            em.add(pts[:, 0], pts[:, 1], pointillize(ALB[C_BACK] * L * bf_, bright, rng, 0.4),
                   rr, flick, side)
        xs = np.linspace(2 if not mirror else W - 168, 168 if not mirror else W - 2, 60)
        for off, bf_, rr in ((0, 0.35, 1.5), (4, 1.6, 1.2)):
            yv = 468.0 + off + 5.0 * np.sin((xs - xs.min()) / 166.0 * math.pi)
            pts = np.column_stack([xs, yv]) + rng.normal(0, 0.4, (len(xs), 2))
            cls = classify(pts[:, 0], pts[:, 1])
            pts = pts[cls == C_BACK]
            L, bright, flick, side = lighting(pts[:, 0], pts[:, 1])
            em.add(pts[:, 0], pts[:, 1], pointillize(ALB[C_BACK] * L * bf_, bright, rng, 0.4), rr, flick, side)
        # armrest post highlight
        pts = seg_dots(rng, (post[0] - 2, post[1] - 24), (post[0] - 2, post[1] + 24), 3.0, wobble=0.2)
        cls = classify(pts[:, 0], pts[:, 1])
        pts = pts[cls == C_POST]
        L, bright, flick, side = lighting(pts[:, 0], pts[:, 1])
        em.add(pts[:, 0], pts[:, 1], pointillize(ALB[C_POST] * L * 1.7, bright, rng, 0.2), 1.2, flick, side)

    # ---- layer 4: GLOW dots (lamp globes and halos)
    for i, (lx, ly) in enumerate(LAMPS):
        side = float(i)
        # brass plate rim
        a = rng.uniform(0, 2 * math.pi, 40)
        em.add(lx + 18 * np.cos(a), ly + 18 * np.sin(a), pointillize(np.tile(ALB[C_PLATE] * 1.2, (40, 1)), 1.0, rng, 0.3),
               1.4, 0.6, side)
        # globe: pale centre -> orange rim
        n = 170
        rr = 14.5 * np.sqrt(rng.random(n))
        a = rng.uniform(0, 2 * math.pi, n)
        t = (rr / 14.5)[:, None]
        col = np.array([1.0, 0.93, 0.66]) * (1 - t ** 1.3) + np.array([1.0, 0.52, 0.12]) * t ** 1.3
        em.add(lx + rr * np.cos(a), ly + rr * np.sin(a), col, rng.uniform(1.3, 2.3, n), 1.0, side)
        # halo
        x, y, r = radial_dots(rng, 1900, lx, ly, 17, 112, 30)
        fall = np.exp(-(r - 17) / 42.0)
        alpha = np.clip(0.95 * fall + 0.07, 0, 1)
        t = np.clip((r - 17) / 90.0, 0, 1)[:, None]
        col = np.array([1.0, 0.78, 0.38]) * (1 - t) + np.array([0.88, 0.32, 0.08]) * t
        teal = rng.random(len(r)) < 0.07
        col[teal] = (0.22, 0.62, 0.72)
        em.add(x, y, col, rng.uniform(0.9, 2.4, len(r)), 1.0, side, alpha)
        # sparse concentric rings (like the lamps in the pointillist reference)
        for R in (25, 35, 47):
            n = int(2 * math.pi * R / 6.0)
            a = np.linspace(0, 2 * math.pi, n, endpoint=False) + rng.normal(0, 0.05, n)
            em.add(lx + R * np.cos(a), ly + R * np.sin(a), (1.0, 0.72, 0.3), 1.2, 1.0, side, 0.55)

    return np.vstack([base_arr, em.array()])


# ----------------------------------------------------------------------------
# 6. OUTSIDE SCENE  (PHASE 2)
#    6a. static dots: sky, Fuji, hills, town, lake + reflections
#        (the water moves in the vertex shader, the dots themselves never change)
#    6b. fireworks: a whole festival timeline baked into one particle buffer,
#        animated entirely on the GPU from the clock
# ----------------------------------------------------------------------------
SHORE = 335.0            # far shore / waterline  (must equal SHORE in the GLSL)
RK = 0.62                # how much the lake squashes reflections (same in GLSL)
FUJI_CX, FUJI_TOP, FUJI_W = 640.0, 112.0, 330.0
K_STATIC, K_STAR, K_REFL, K_WATER = 0.0, 1.0, 2.0, 3.0   # behaviour tag read by the shader

OUT_PAL = dict(cool=np.array([0.12, 0.45, 0.70]), gold=np.array([0.55, 0.72, 0.95]),
               viol=np.array([0.30, 0.25, 0.65]))


def fuji_hw(y):
    """half-width of the mountain at height y: concave, symmetrical, slightly ridged"""
    d = np.clip((y - FUJI_TOP) / (SHORE - FUJI_TOP), 0, 1)
    ridge = (2.2 * np.sin(d * 38.0) + 1.4 * np.sin(d * 91.0 + 1.0)) * d
    return 9.0 + (FUJI_W - 9.0) * d ** 1.16 + ridge


def snow_depth(u):
    """how far down the snow reaches (fraction of height) at distance u from the axis"""
    return 0.27 + 0.10 * np.abs(np.sin(u * 0.062 + 0.7)) ** 1.4 + 0.04 * np.sin(u * 0.19)


def _add(em, rng, x, y, rgb, rad, kind=0.0, weight=0.3, alpha=1.0, shuffle=True):
    n = len(x)
    if n == 0:
        return
    p = rng.permutation(n) if shuffle else np.arange(n)

    def pk(v):
        v = np.asarray(v, np.float32)
        return v[p] if v.ndim >= 1 and len(v) == n else v
    rgb = np.asarray(rgb, np.float32)
    if rgb.ndim == 2 and len(rgb) == n:
        rgb = rgb[p]
    em.add(np.asarray(x)[p], np.asarray(y)[p], rgb, pk(rad), pk(kind), pk(weight), pk(alpha))


def _mirror_y(y):
    return SHORE + (SHORE - y) * RK


def build_outside(rng):
    cx, cy, hx, hy, _ = WIN
    pad = 26.0
    bbox = (cx - hx - pad, cy - hy - pad, cx + hx + pad, cy + hy + pad)
    ybot = bbox[3]
    em = Emitter(rng)

    # ---- sky: deep navy up high, teal haze at the horizon
    gx, gy = jitter_grid(4.6, rng, (bbox[0], bbox[1], bbox[2], SHORE + 3))
    k = (sdr(gx, gy, WIN) < pad) & (gy < SHORE + 3)
    gx, gy = gx[k], gy[k]
    stops = [54, 180, 270, 315, 335]
    cols = np.array([[0.012, 0.03, 0.12], [0.03, 0.09, 0.28], [0.07, 0.19, 0.40],
                     [0.16, 0.31, 0.47], [0.42, 0.40, 0.46]])
    rgb = np.column_stack([np.interp(gy, stops, cols[:, i]) for i in range(3)])
    rgb = pointillize(rgb, np.full(len(gx), 0.6), rng, 0.2, **OUT_PAL)
    _add(em, rng, gx, gy, rgb, 4.6 * 0.80 * rng.uniform(0.85, 1.15, len(gx)), K_STATIC, 0.15)

    # ---- stars (subtle, they twinkle in the shader)
    n = 130
    sx = rng.uniform(bbox[0] + 40, bbox[2] - 40, n)
    sy = rng.uniform(bbox[1] + 10, 262, n)
    sc = np.tile([0.85, 0.92, 1.0], (n, 1)) * rng.uniform(0.55, 1.0, (n, 1))
    _add(em, rng, sx, sy, sc, rng.uniform(0.8, 1.6, n), K_STAR, 0.15, shuffle=False)

    # ---- Mount Fuji, built from dots that follow its own shape
    gx, gy = jitter_grid(3.8, rng, (FUJI_CX - FUJI_W - 20, FUJI_TOP - 4, FUJI_CX + FUJI_W + 20, SHORE + 2))
    u = np.abs(gx - FUJI_CX)
    k = (u <= fuji_hw(gy)) & (gy >= FUJI_TOP) & (gy <= SHORE + 2)
    gx, gy, u = gx[k], gy[k], u[k]
    hw = fuji_hw(gy)
    d = (gy - FUJI_TOP) / (SHORE - FUJI_TOP)
    lit = 0.5 - 0.5 * np.clip((gx - FUJI_CX) / np.maximum(hw, 1), -1, 1)      # moon from the left
    body = np.array([0.04, 0.055, 0.19]) * (1 - lit)[:, None] + np.array([0.16, 0.20, 0.47]) * lit[:, None]
    hz = (smoothstep(0.50, 1.0, d) * 0.75)[:, None]                            # distance haze
    body = body * (1 - hz) + np.array([0.17, 0.30, 0.46]) * hz
    snowcol = np.array([0.40, 0.50, 0.82]) * (1 - lit)[:, None] + np.array([0.94, 0.96, 1.0]) * lit[:, None]
    snow = d < snow_depth(u)
    rgb = np.where(snow[:, None], snowcol, body)
    fpal = dict(cool=np.array([0.30, 0.55, 0.85]), gold=np.array([0.78, 0.82, 1.0]),
                viol=np.array([0.35, 0.30, 0.70]))
    rgb = pointillize(rgb, np.where(snow, 0.9, 0.45), rng, **fpal)
    frad = 3.8 * 0.80 * rng.uniform(0.85, 1.15, len(gx))
    fx_, fy_, frgb, frad_ = gx.copy(), gy.copy(), rgb.copy(), frad.copy()
    _add(em, rng, gx, gy, rgb, frad, K_STATIC, 0.25)
    # ravines and ridges: dotted lines running down the slopes
    for sgn in (-1, 1):
        for i, f in enumerate(np.linspace(0.14, 0.90, 6)):
            ys = np.arange(FUJI_TOP + 70, SHORE - 25, 3.8)
            ys = ys[rng.random(len(ys)) < 0.78]
            xs = FUJI_CX + sgn * f * fuji_hw(ys) + rng.normal(0, 0.5, len(ys))
            dd = (ys - FUJI_TOP) / (SHORE - FUJI_TOP)
            ins = dd < snow_depth(np.abs(xs - FUJI_CX))
            dark = (i % 2 == 0)
            if not dark and sgn > 0:
                continue
            col = np.where(ins[:, None], np.array([0.50, 0.60, 0.90]), np.array([0.03, 0.04, 0.14]) if dark
                           else np.array([0.30, 0.38, 0.68]))
            haze = (smoothstep(0.55, 1.0, dd) * 0.5)[:, None]
            col = col * (1 - haze) + np.array([0.17, 0.30, 0.46]) * haze
            _add(em, rng, xs, ys, col, rng.uniform(1.0, 1.6, len(ys)), K_STATIC, 0.25, shuffle=False)

    # ---- low hills on both sides
    gx, gy = jitter_grid(4.0, rng, (bbox[0], SHORE - 60, bbox[2], SHORE + 3))
    hh = (9 + 26 * np.exp(-((gx - 250) / 110) ** 2) + 20 * np.exp(-((gx - 1040) / 140) ** 2)
          + 4 * np.sin(gx / 17.0))
    k = (gy > SHORE - hh) & (gy <= SHORE + 2) & (sdr(gx, gy, WIN) < pad)
    gx, gy = gx[k], gy[k]
    rgb = np.array([0.035, 0.06, 0.17]) * rng.uniform(0.8, 1.2, (len(gx), 1))
    rgb = pointillize(rgb, np.full(len(gx), 0.35), rng, 0.5, **fpal)
    _add(em, rng, gx, gy, rgb, 4.0 * 0.78 * rng.uniform(0.85, 1.15, len(gx)), K_STATIC, 0.4)
    xs = np.arange(bbox[0], bbox[2], 4.6)
    ys = SHORE - (9 + 26 * np.exp(-((xs - 250) / 110) ** 2) + 20 * np.exp(-((xs - 1040) / 140) ** 2)
                  + 4 * np.sin(xs / 17.0))
    _add(em, rng, xs, ys, np.array([0.18, 0.32, 0.58]), 1.2, K_STATIC, 0.4, shuffle=False)

    # ---- the lakeside town
    wins = []        # (x, y, rgb) for every lit window / lamp, used for reflections

    def fill(bb, mask, rgb, cell=2.8, bright=0.25, weight=0.6, amount=0.5, rad=0.74):
        gx, gy = jitter_grid(cell, rng, bb)
        k = mask(gx, gy)
        gx, gy = gx[k], gy[k]
        c = pointillize(np.tile(rgb, (len(gx), 1)) * rng.uniform(0.8, 1.2, (len(gx), 1)),
                        np.full(len(gx), bright), rng, amount, **fpal)
        _add(em, rng, gx, gy, c, cell * rad * rng.uniform(0.85, 1.15, len(gx)), K_STATIC, weight)

    def lit_window(x, y, big=1.0):
        col = [(1.0, 0.78, 0.38), (1.0, 0.62, 0.25), (0.98, 0.90, 0.58)][rng.integers(0, 3)]
        _add(em, rng, [x], [y], np.array([col]) * 0.30, 3.0 * big, K_STATIC, 0.6, 0.5, shuffle=False)
        _add(em, rng, [x], [y], np.array([col]), rng.uniform(1.15, 1.6) * big, K_STATIC, 0.6, shuffle=False)
        wins.append((x, y, col))

    def building(bx, by, w, h, roof, shade, windows=True):
        body = np.array([0.045, 0.04, 0.11]) * shade
        rh = roof
        cxm = bx + w / 2

        def m(gx, gy):
            inb = (gx >= bx) & (gx <= bx + w) & (gy >= by - h) & (gy <= by)
            t = ((by - h) - gy) / max(rh, 1.0)
            inr = (gy < by - h) & (gy >= by - h - rh) & (np.abs(gx - cxm) <= (w / 2 + 2.5) * (1 - t))
            return inb | inr
        fill((bx - 4, by - h - rh - 2, bx + w + 4, by), m, body)
        if windows:
            for i in range(max(int(w // 6), 1)):
                for j in range(max(int((h - 3) // 6), 1)):
                    if rng.random() < 0.52:
                        lit_window(bx + 3 + 6 * i + rng.normal(0, .3), by - h + 4 + 6 * j + rng.normal(0, .3))

    PAG_X = 395.0
    for row in (0, 1):                     # back row first, then the front row
        x = 282.0 + 6 * row
        while x < 995:
            w = rng.uniform(11, 27)
            if row == 0:
                h, by = rng.uniform(13, 27), SHORE - 5
            else:
                h, by = rng.uniform(7, 17), SHORE
            if abs(x + w / 2 - PAG_X) < 26 and row == 0:
                x += 30
                continue
            building(x, by, w, h, rng.uniform(4, 9) if rng.random() < 0.65 else 0.0,
                     rng.uniform(0.7, 1.3))
            x += w + rng.uniform(1.5, 7.0)

    # five-storey pagoda
    by = SHORE - 5.0
    for tier, bw in enumerate([28, 24, 20, 16, 12]):
        top = by - tier * 10.5 - 8.0
        fill((PAG_X - bw, top - 1, PAG_X + bw, top + 8),
             lambda gx, gy, bw=bw, top=top: (np.abs(gx - PAG_X) <= bw / 2) & (gy >= top) & (gy <= top + 8),
             np.array([0.05, 0.04, 0.11]))
        fill((PAG_X - bw, top - 5, PAG_X + bw, top),
             lambda gx, gy, bw=bw, top=top: (np.abs(gx - PAG_X) <= bw / 2 + 6 - 1.2 * (top - gy)) & (gy >= top - 4) & (gy <= top),
             np.array([0.03, 0.035, 0.09]), bright=0.3)
        if tier < 4:
            lit_window(PAG_X - bw / 4, top + 4)
            lit_window(PAG_X + bw / 4, top + 4)
    spire = seg_dots(rng, (PAG_X, by - 5 * 10.5 - 6), (PAG_X, by - 5 * 10.5 - 20), 2.8, wobble=0.2)
    _add(em, rng, spire[:, 0], spire[:, 1], np.array([0.20, 0.22, 0.38]), 1.1, K_STATIC, 0.6, shuffle=False)

    # street lamps along the shore
    x = 280.0
    while x < 1000:
        _add(em, rng, [x], [SHORE + 0.5], np.array([[1.0, 0.76, 0.36]]), 1.5, K_STATIC, 0.6, shuffle=False)
        _add(em, rng, [x], [SHORE + 0.5], np.array([[1.0, 0.62, 0.25]]) * 0.3, 3.4, K_STATIC, 0.6, 0.5, shuffle=False)
        wins.append((x, SHORE + 0.5, (1.0, 0.76, 0.36)))
        x += rng.uniform(7, 16)

    # ---- the lake: dark water that carries the sky
    gx, gy = jitter_grid(4.6, rng, (bbox[0], SHORE + 1, bbox[2], ybot))
    k = sdr(gx, gy, WIN) < pad
    gx, gy = gx[k], gy[k]
    t = np.clip((gy - SHORE) / (ybot - SHORE), 0, 1)
    lc = np.array([[0.11, 0.24, 0.42], [0.04, 0.10, 0.26], [0.015, 0.04, 0.14]])
    rgb = np.column_stack([np.interp(t, [0, 0.3, 1], lc[:, i]) for i in range(3)])
    rgb = pointillize(rgb, np.full(len(gx), 0.5), rng, 0.5, **OUT_PAL)
    _add(em, rng, gx, gy, rgb, 4.6 * 0.74 * rng.uniform(0.85, 1.15, len(gx)), K_WATER, 0.8)

    # mountain reflection: the same dots mirrored, darker, stretched by the shader
    ry = _mirror_y(fy_)
    k = (ry < ybot) & (rng.random(len(fx_)) < 0.55)
    rr = frgb[k] * 0.55 + np.array([0.01, 0.03, 0.09])
    _add(em, rng, fx_[k], ry[k], rr, frad_[k] * 0.95, K_REFL, 0.8, 0.9)

    # town lights reflected as short vertical shimmering dashes
    for (wx, wy, col) in wins:
        by_ = _mirror_y(wy)
        for kk in range(4):
            yy = by_ + kk * 5.5 + rng.normal(0, 1.0)
            if yy > ybot:
                break
            _add(em, rng, [wx + rng.normal(0, 1.2)], [yy], np.array([col]), 1.5 - 0.2 * kk,
                 K_REFL, 0.8, 0.8 - 0.18 * kk, shuffle=False)

    # moonlit ripples: short horizontal dashes on the water
    for _ in range(330):
        y = rng.uniform(SHORE + 5, ybot - 6)
        x = rng.uniform(bbox[0] + 20, bbox[2] - 60)
        ln = rng.uniform(14, 34 + 70 * (y - SHORE) / (ybot - SHORE))
        pts = seg_dots(rng, (x, y), (x + ln, y), 4.4, wobble=0.5)
        pts = pts[sdr(pts[:, 0], pts[:, 1], WIN) < pad]
        if len(pts) == 0:
            continue
        warm = (300 < x < 1000) and rng.random() < 0.28
        col = np.array([0.95, 0.66, 0.34]) if warm else np.array([0.36, 0.62, 0.88])
        _add(em, rng, pts[:, 0], pts[:, 1], col * rng.uniform(0.40, 0.80), rng.uniform(1.1, 1.8, len(pts)),
             K_WATER, 0.8, 0.8, shuffle=False)

    # two little boats with lanterns
    for bx, by in ((505.0, 372.0), (838.0, 401.0)):
        xs = np.linspace(-10, 10, 14)
        hull = np.column_stack([bx + xs, by + 0.035 * xs ** 2 - 3.5])
        _add(em, rng, hull[:, 0], hull[:, 1], np.array([0.03, 0.03, 0.08]), 1.7, K_WATER, 0.8, shuffle=False)
        _add(em, rng, [bx + 10], [by - 7], np.array([[1.0, 0.72, 0.30]]), 1.7, K_WATER, 0.8, shuffle=False)
        for kk in range(4):
            _add(em, rng, [bx + 10 + rng.normal(0, 1)], [by - 1 + kk * 5], np.array([[1.0, 0.70, 0.30]]),
                 1.4, K_REFL, 0.8, 0.7 - 0.15 * kk, shuffle=False)
    return em.array()


# ---------------------------------------------------------------- fireworks
FW_CYCLE = 150.0                     # the festival loops every 150 s
FW_ROWS = 17
PALETTE = {"gold": (1.00, 0.78, 0.30), "white": (1.00, 0.96, 0.88), "red": (1.00, 0.18, 0.16),
           "violet": (0.62, 0.34, 1.00), "pink": (1.00, 0.42, 0.70), "cyan": (0.22, 0.90, 1.00),
           "green": (0.32, 1.00, 0.50)}
COLOR_NAMES = list(PALETTE)


def _sphere_dirs(n, rng):
    z = rng.uniform(-1, 1, n)
    a = rng.uniform(0, 2 * math.pi, n)
    s = np.sqrt(1 - z * z)
    return np.column_stack([s * np.cos(a), s * np.sin(a)])


def _fw(rows, rng, o, v, tb, life, col, size, drag, grav, tdt, nt, kind):
    """append particles; every particle is stored nt times (head + trail copies)"""
    o = np.atleast_2d(np.asarray(o, np.float32))
    n = max(len(o), len(np.atleast_2d(v)))
    o = np.broadcast_to(o, (n, 2))
    v = np.broadcast_to(np.atleast_2d(np.asarray(v, np.float32)), (n, 2))

    def b(a):
        return np.broadcast_to(np.asarray(a, np.float32), (n,))
    col = np.broadcast_to(np.atleast_2d(np.asarray(col, np.float32)), (n, 3))
    base = np.column_stack([o, v, b(tb), b(life), col, b(size), b(drag), b(grav), b(tdt),
                            b(kind), np.zeros(n, np.float32), b(nt), rng.random(n).astype(np.float32)])
    rep = np.repeat(base, nt, axis=0)
    rep[:, 14] = np.tile(np.arange(nt, dtype=np.float32), n)
    rows.append(rep)


def _burst(rows, rng, pattern, ex, ey, tb, A, B):
    A, B = np.array(A), np.array(B)
    if pattern == "peony":                                   # spherical shell, two-tone core
        n = 360
        S0 = rng.uniform(165, 190)
        d = _sphere_dirs(n, rng)
        sp = S0 * (1 + rng.normal(0, 0.025, n))
        core = rng.random(n) < 0.30
        sp[core] *= 0.55
        col = np.where(core[:, None], B, A) * rng.uniform(0.85, 1.1, (n, 1))
        _fw(rows, rng, (ex, ey), d * sp[:, None], tb, rng.uniform(2.6, 3.3, n), col,
            rng.uniform(1.5, 2.4, n), 1.3, 40.0, 0.05, 4, 0)
        R = 135.0
    elif pattern == "chrysanthemum":                         # radial spokes of dots
        spokes, per = 30, 12
        S0 = rng.uniform(185, 215)
        th = np.repeat(2 * math.pi * np.arange(spokes) / spokes + rng.normal(0, 0.02, spokes), per)
        frac = np.tile(np.linspace(0.30, 1.0, per), spokes) * (1 + rng.normal(0, 0.01, spokes * per))
        d = np.column_stack([np.cos(th), np.sin(th)])
        mix = (1 - frac)[:, None] / 0.7
        col = np.array(A) * (1 - mix) + np.array(B) * mix
        _fw(rows, rng, (ex, ey), d * (S0 * frac)[:, None], tb, rng.uniform(3.0, 3.6, len(th)),
            np.clip(col, 0, 1), rng.uniform(1.3, 2.0, len(th)), 1.5, 30.0, 0.058, 6, 0)
        R = 140.0
    elif pattern == "willow":                                # slow golden droop with long trails
        n = 280
        d = _sphere_dirs(n, rng)
        sp = rng.uniform(120, 150) * rng.uniform(0.45, 1.0, n)
        _fw(rows, rng, (ex, ey), d * sp[:, None], tb, rng.uniform(4.2, 5.0, n), A * rng.uniform(0.85, 1.1, (n, 1)),
            rng.uniform(1.3, 2.0, n), 0.9, 60.0, 0.09, 6, 0)
        R = 100.0
    else:                                                    # cascade: fountain that rains down
        n = 300
        ang = -math.pi / 2 + np.radians(rng.uniform(-75, 75, n))
        sp = rng.uniform(70, 200, n)
        d = np.column_stack([np.cos(ang), np.sin(ang)])
        col = np.where((rng.random(n) < 0.3)[:, None], B, A)
        _fw(rows, rng, (ex, ey), d * sp[:, None], tb, rng.uniform(3.2, 3.8, n), col,
            rng.uniform(1.4, 2.2, n), 0.7, 110.0, 0.07, 5, 0)
        R = 80.0
    # soft glow particles around the explosion (additive blobs) + one big flash
    ng = 40
    o = np.column_stack([ex + rng.normal(0, 18, ng), ey + rng.normal(0, 18, ng)])
    _fw(rows, rng, o, rng.normal(0, 14, (ng, 2)), tb, rng.uniform(1.0, 2.0, ng), A * 0.07,
        rng.uniform(8, 18, ng), 1.0, 0.0, 0.0, 1, 1)
    of = np.column_stack([ex + rng.normal(0, 6, 6), np.full(6, ey)])
    _fw(rows, rng, of, (0, 0), tb, rng.uniform(0.35, 0.6, 6), A * 0.045, rng.uniform(26, 40, 6),
        1.0, 0.0, 0.0, 1, 1)
    # late glitter: tiny white-gold crackles blinking inside the burst
    nc = 20 if pattern == "peony" else 46
    r = R * np.sqrt(rng.random(nc)) * 0.9
    a = rng.uniform(0, 2 * math.pi, nc)
    oc = np.column_stack([ex + r * np.cos(a), ey + r * np.sin(a) + (40 if pattern == "willow" else 0)])
    _fw(rows, rng, oc, (0, 0), tb + rng.uniform(0.6, 2.2, nc), rng.uniform(0.14, 0.30, nc),
        (1.0, 0.92, 0.70), rng.uniform(1.8, 2.8, nc), 1.0, 0.0, 0.0, 1, 3)


def build_fireworks(rng):
    """Bakes the whole festival: launch -> rise -> burst -> droop & fade, one after another,
    with the occasional overlapping pair. The GPU animates it from the clock."""
    rows = []
    t, prev_x, prev_overlap = 1.4, 640.0, False
    while t < FW_CYCLE - 9.0:
        pattern = rng.choice(["peony", "chrysanthemum", "willow", "cascade"], p=[0.28, 0.27, 0.22, 0.23])
        lx = rng.uniform(320, 960)
        while abs(lx - prev_x) < 150:
            lx = rng.uniform(320, 960)
        prev_x = lx
        ex = float(np.clip(lx + rng.uniform(-70, 70), 300, 980))
        ey = {"peony": rng.uniform(135, 215), "chrysanthemum": rng.uniform(150, 215),
              "willow": rng.uniform(125, 190), "cascade": rng.uniform(185, 228)}[pattern]
        tr = rng.uniform(1.15, 1.55)
        ly = SHORE - 6.0
        names = list(rng.choice(COLOR_NAMES, 2, replace=False))
        if pattern == "willow":
            names = ["gold" if rng.random() < 0.7 else "white", "white"]
        A, B = PALETTE[names[0]], PALETTE[names[1]]
        if pattern == "chrysanthemum" and names[0] == "white":
            B = PALETTE["gold"]
        # launch phase: bright rocket with a dotted tail + sparks shed along the way
        _fw(rows, rng, (lx, ly), (ex - lx, ey - ly), t, tr, (1.0, 0.88, 0.60), 2.4, 1.0, 0.0, 0.032, 9, 2)
        s = rng.uniform(0.05, 0.95, 14)
        e = 1 - (1 - s) ** 2
        os_ = np.column_stack([lx + (ex - lx) * e, ly + (ey - ly) * e])
        _fw(rows, rng, os_, rng.normal(0, 12, (14, 2)) + [0, 14], t + s * tr, rng.uniform(0.5, 0.9, 14),
            (1.0, 0.70, 0.35), rng.uniform(1.1, 1.6, 14), 1.5, 40.0, 0.04, 2, 0)
        # explosion phase
        _burst(rows, rng, pattern, ex, ey, t + tr, A, B)
        # sequencing: mostly one at a time, now and then a second one overlaps
        if (not prev_overlap) and rng.random() < 0.24:
            t += rng.uniform(0.3, 0.75)
            prev_overlap = True
        else:
            t += rng.uniform(1.7, 3.0)
            prev_overlap = False
    return np.ascontiguousarray(np.vstack(rows), np.float32)


# ----------------------------------------------------------------------------
# 6c. UNDERWATER WORLD (PHASE 3)
#     Every element is its own dot layer. Positions are fixed here; the vertex
#     shader moves them (currents, falling snow, rising bubbles, drifting
#     silhouettes), so the world stays alive at almost no CPU cost.
# ----------------------------------------------------------------------------
UX0, UX1, UY0, UY1 = 194.0, 1086.0, 54.0, 516.0       # window box (+ margin); wrap range of particles
UXR, UYR = UX1 - UX0, UY1 - UY0
K_BODY, K_SNOW, K_PLANK, K_BUBBLE, K_BIO, K_SIL, K_SURF, K_RAY, K_FRAG, K_WHALE = (
    4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0)
UW_PAL = dict(cool=np.array([0.10, 0.78, 0.88]),       # cyan
              gold=np.array([0.85, 0.78, 0.45]),       # pale gold
              viol=np.array([0.38, 0.26, 0.74]))       # violet


def seabed_h(xr):
    """height of the distant sea floor; periodic so it can scroll forever"""
    a = 2 * np.pi * xr / UXR
    h = 34 + 16 * np.sin(2 * a + 0.5) + 11 * np.sin(5 * a + 2.0) + 8 * np.sin(9 * a)
    for c, w, hh in ((0.18, 0.018, 70.0), (0.86, 0.02, 58.0)):
        dx = ((xr / UXR - c + 0.5) % 1.0) - 0.5
        h = h + hh * np.exp(-(dx / w) ** 2)
    return h


def _uw_body(em, rng):
    gx, gy = jitter_grid(5.0, rng, (UX0, UY0, UX1, UY1))
    k = sdr(gx, gy, WIN) < 26
    gx, gy = gx[k], gy[k]
    t = (gy - UY0) / UYR
    lc = np.array([[0.08, 0.45, 0.58], [0.04, 0.22, 0.46], [0.025, 0.12, 0.32]])
    rgb = np.column_stack([np.interp(t, [0, 0.35, 1], lc[:, i]) for i in range(3)])
    rgb = pointillize(rgb, np.full(len(gx), 0.55), rng, 0.30, **UW_PAL)
    _add(em, rng, gx, gy, rgb, 5.0 * 0.86 * rng.uniform(0.85, 1.15, len(gx)), K_BODY, rng.random(len(gx)))


def _uw_rays(em, rng):
    """light shafts: soft slanted bands of pale dots, densest on their axis"""
    for _ in range(7):
        xs = rng.uniform(UX0 - 250, UX1 - 100)
        width = rng.uniform(34, 90)
        n = int(1500 * width / 60)
        along = rng.random(n)
        yy = UY0 - 40 + along * (UYR + 80)
        xx = xs + 0.36 * (yy - UY0) + rng.normal(0, width / 3.4, n)
        gold = rng.random(n) < 0.10
        col = np.where(gold[:, None], (0.95, 0.88, 0.55), (0.55, 0.95, 1.0))
        alpha = rng.uniform(0.16, 0.42, n) * (0.5 + 0.5 * (1 - along))
        em.add(xx, yy, col, rng.uniform(1.2, 2.6, n), K_RAY, along, alpha, rng.uniform(0, 6.28))


def _uw_silhouettes(em, rng):
    # --- sea floor ridges + a sunken torii gate, far away and barely there
    gx, gy = jitter_grid(5.5, rng, (UX0, UY1 - 140, UX1, UY1 + 10))
    k = gy > UY1 - seabed_h(gx - UX0)
    gx, gy = gx[k], gy[k]
    rgb = np.array([0.004, 0.016, 0.065]) * rng.uniform(0.8, 1.3, (len(gx), 1))
    rgb = pointillize(rgb, np.full(len(gx), 0.15), rng, 0.5, **UW_PAL)
    _add(em, rng, gx, gy, rgb, 5.5 * 0.76 * rng.uniform(0.85, 1.15, len(gx)), K_SIL, 0.15, 0.80)
    xs = UX0 + np.arange(0, UXR, 4.2)
    _add(em, rng, xs, UY1 - seabed_h(xs - UX0), np.array([0.12, 0.40, 0.62]), 1.2, K_SIL, 0.15, 0.30,
         shuffle=False)
    tx = UX0 + 0.62 * UXR
    base = UY1 - float(seabed_h(np.array([tx - UX0]))[0]) + 4.0
    parts = [(tx - 34, base - 78, tx - 27, base), (tx + 27, base - 78, tx + 34, base),     # pillars
             (tx - 52, base - 86, tx + 52, base - 76), (tx - 38, base - 62, tx + 38, base - 55)]  # beams
    for x0, y0, x1, y1 in parts:
        gx, gy = jitter_grid(3.2, rng, (x0, y0, x1, y1))
        rgb = pointillize(np.tile([0.005, 0.014, 0.06], (len(gx), 1)), np.full(len(gx), 0.2), rng, 0.4, **UW_PAL)
        _add(em, rng, gx, gy, rgb, 2.6, K_SIL, 0.15, 0.85)
        _add(em, rng, gx[gx < x0 + 3.5], gy[gx < x0 + 3.5], np.array([0.12, 0.42, 0.65]), 1.1, K_SIL, 0.15, 0.30)
    # --- one vast whale, crossing the whole window very slowly (shader moves it)
    L = 420.0
    gx, gy = jitter_grid(5.5, rng, (-L - 50, -90, 50, 90))
    u = -gx / L
    hh = 46 * np.sin(np.pi * np.clip(u * 0.72 + 0.14, 0, 1)) ** 0.8 * (1 - 0.78 * u)
    body = (u >= 0) & (u <= 1) & (np.abs(gy) <= hh)
    fluke = (((gx + L + 14) / 11.0) ** 2 + (gy / 34.0) ** 2) < 1
    fin = (((gx + 0.32 * L) / 10.0) ** 2 + ((gy - 34) / 22.0) ** 2) < 1
    k = body | fluke | fin
    gx, gy, u = gx[k], gy[k], np.clip(-gx[k] / L, 0, 1.1)
    rgb = pointillize(np.tile([0.005, 0.02, 0.08], (len(gx), 1)), np.full(len(gx), 0.2), rng, 0.4, **UW_PAL)
    ph = 1.7
    em.add(-150 + gx, 262 + gy, rgb, 5.5 * 0.74 * rng.uniform(0.85, 1.15, len(gx)), K_WHALE, u, 0.55, ph)
    xs = np.arange(-L, 0, 6.0)
    hb = 46 * np.sin(np.pi * np.clip((-xs / L) * 0.72 + 0.14, 0, 1)) ** 0.8 * (1 - 0.78 * (-xs / L))
    em.add(-150 + xs, 262 - hb, np.array([0.18, 0.52, 0.72]), 1.3, K_WHALE, -xs / L, 0.22, ph)


def _uw_particles(em, rng):
    # marine snow: sparse, slow, many sizes
    n = 300
    layer = rng.random(n) ** 0.8
    col = np.tile([0.70, 0.88, 0.97], (n, 1)) * rng.uniform(0.55, 1.0, (n, 1))
    col = pointillize(col, np.full(n, 0.8), rng, 0.5, **UW_PAL)
    em.add(rng.uniform(UX0, UX1, n), rng.uniform(UY0, UY1, n), col, 0.8 + 1.5 * layer,
           K_SNOW, layer, rng.uniform(0.35, 0.85, n))
    # plankton: soft clusters of luminous dots that move as one
    palette = [(0.15, 0.85, 0.95), (0.20, 0.90, 0.72), (0.50, 0.40, 0.95), (0.90, 0.82, 0.50), (0.55, 0.85, 1.0)]
    for _ in range(14):
        c = np.array([rng.uniform(UX0, UX1), rng.uniform(UY0 + 30, UY1)])
        layer = rng.uniform(0.1, 1.0)
        ph = rng.uniform(0, 6.28)
        m = int(rng.integers(14, 40))
        sg = rng.uniform(14, 38)
        base = np.array(palette[rng.integers(0, len(palette))])
        pts = c + rng.normal(0, sg, (m, 2))
        col = np.clip(base * rng.uniform(0.7, 1.15, (m, 1)), 0, 1)
        em.add(pts[:, 0], pts[:, 1], col, rng.uniform(1.1, 2.4, m) * (0.7 + 0.6 * layer),
               K_PLANK, layer, rng.uniform(0.45, 0.95, m), ph)
        h = max(m // 6, 1)
        em.add(pts[:h, 0], pts[:h, 1], base * 0.8, rng.uniform(8, 12, h), K_PLANK, layer, 0.07, ph)
    # leftover firework colours, softened by the sea: the fireworks' afterlife
    nf = 70
    cols = np.array([PALETTE[c] for c in rng.choice(COLOR_NAMES, nf)])
    cols = cols * 0.65 + np.array([0.10, 0.45, 0.55]) * 0.35
    pos = np.column_stack([rng.normal(640, 230, nf), rng.normal(230, 90, nf)])
    pos[:, 0] = np.clip(pos[:, 0], UX0, UX1 - 1)
    pos[:, 1] = np.clip(pos[:, 1], UY0, UY1 - 1)
    layer = rng.uniform(0.2, 0.8, nf)
    em.add(pos[:, 0], pos[:, 1], cols, rng.uniform(1.4, 2.8, nf), K_FRAG, layer, 0.85, rng.uniform(0, 6.28, nf))
    em.add(pos[:20, 0], pos[:20, 1], cols[:20], rng.uniform(8, 13, 20), K_FRAG, layer[:20], 0.08,
           rng.uniform(0, 6.28, 20))
    # faint bioluminescence in the lower dark water: small core + soft halo
    for _ in range(26):
        x, y = rng.uniform(UX0 + 20, UX1 - 20), rng.uniform(250, UY1 - 20)
        layer, ph = rng.uniform(0.05, 0.6), rng.uniform(0, 6.28)
        col = np.array([(0.25, 0.95, 1.0), (0.60, 0.45, 1.0), (0.95, 0.85, 0.5), (0.30, 1.0, 0.75)][rng.integers(0, 4)])
        em.add([x], [y], col, rng.uniform(1.6, 2.8), K_BIO, layer, 0.9, ph)
        em.add([x], [y], col * 0.8, rng.uniform(9, 14), K_BIO, layer, 0.13, ph)
    # bubbles: little rising chains + a few loners
    for _ in range(12):
        x0, layer, ph = rng.uniform(UX0 + 30, UX1 - 30), rng.uniform(0.2, 0.9), rng.uniform(0, 6.28)
        m = int(rng.integers(7, 13))
        ys = UY0 + (rng.uniform(UY0, UY1) + np.arange(m) * rng.uniform(14, 26) - UY0) % UYR
        em.add(x0 + rng.normal(0, 1.2, m), ys, (0.75, 0.95, 1.0), (1.8 + 2.4 * layer) * rng.uniform(0.7, 1.2, m),
               K_BUBBLE, layer, 0.95, ph)
    n = 50
    layer = rng.random(n)
    em.add(rng.uniform(UX0, UX1, n), rng.uniform(UY0, UY1, n), (0.75, 0.95, 1.0), 1.8 + 2.8 * layer,
           K_BUBBLE, layer, rng.uniform(0.6, 0.95, n))


def _uw_surface(em, rng):
    """the waterline itself: rows of foam / cyan / turquoise dots; y is an offset from the line"""
    rows = [(-9, (0.85, 0.97, 1.0), 0.45), (-5, (0.90, 1.0, 1.0), 0.6), (-1, (0.80, 0.98, 1.0), 0.9),
            (3, (0.35, 0.90, 0.95), 0.9), (7, (0.20, 0.75, 0.85), 0.75), (11, (0.12, 0.60, 0.72), 0.6),
            (16, (0.08, 0.45, 0.62), 0.4)]
    for dy, col, a in rows:
        xs = np.arange(UX0 - 12, UX1 + 12, 3.4) + rng.uniform(-1.2, 1.2, int((UXR + 24) / 3.4) + 1)[:int((UXR + 24) / 3.4) + 1][:len(np.arange(UX0 - 12, UX1 + 12, 3.4))]
        n = len(xs)
        em.add(xs, dy + rng.normal(0, 0.8, n), np.array(col) * rng.uniform(0.8, 1.1, (n, 1)),
               rng.uniform(1.3, 2.7, n), K_SURF, 0.0, a * rng.uniform(0.6, 1.0, n))


def build_underwater(rng):
    em = Emitter(rng)
    _uw_body(em, rng)
    _uw_rays(em, rng)
    _uw_silhouettes(em, rng)
    _uw_particles(em, rng)
    _uw_surface(em, rng)
    return em.array()


def build_spill(rng):
    """faint blue light from the sea landing on the wall, sill and seats (additive). The wooden window
    frame (C_FRAME) is deliberately excluded - its warm brown/amber must never be recoloured."""
    em = Emitter(rng)
    gx, gy = jitter_grid(4.6, rng, (60, 10, W - 60, 680))
    sd = sdr(gx, gy, OUTER)
    cls = classify(gx, gy)
    p = np.exp(-np.maximum(sd, 0) / 42.0)
    # sd > 2.6 keeps every dot (radius <= 2.4) fully clear of the frame; the class tests exclude glass, frame, lamps
    ok = ((sd > 2.6) & (sd < 100) & (cls != C_GLASS) & (cls != C_FRAME) & (cls != C_LAMP) & (cls != C_PLATE)
          & (rng.random(len(gx)) < 0.85 * p))
    gx, gy, p = gx[ok], gy[ok], p[ok]
    pick = rng.random(len(gx))
    col = np.where((pick < 0.70)[:, None], (0.10, 0.45, 0.75),
                   np.where((pick < 0.90)[:, None], (0.10, 0.75, 0.70), (0.40, 0.30, 0.85)))
    em.add(gx, gy, col, rng.uniform(1.2, 2.4, len(gx)), 5.0, 0.0, rng.uniform(0.18, 0.42, len(gx)) * (0.5 + 0.5 * p))
    return em.array()


# ----------------------------------------------------------------------------
# 6d. THE JELLYFISH GARDEN (PHASE 4)
#     A self-contained system, in two halves:
#       - jelly_dots(): builds each animal out of individual dots, once. Every dot stores
#         *where it lives on the animal* (bell surface angle, tentacle number + distance
#         along it, ...), never a screen position.
#       - JellySystem: a small simulation (drift, pulse timing, lean, respawn) that hands
#         the GPU a few numbers per animal each frame. The vertex shader turns those numbers
#         plus each dot's anatomy into a position, so bells breathe and tentacles flow.
# ----------------------------------------------------------------------------
NJ = 17                                  # jellyfish slots (must equal NJ in the GLSL)
JR0, JH0 = 46.0, 28.0                    # bell radius / height at scale 1 (pixels)
GARDEN_DEPTH = 0.80                      # depth used by the G key / --garden
(JK_SURF, JK_RIM, JK_CANAL, JK_CORE, JK_ARM, JK_TENT, JK_FIL, JK_MOTE, JK_HALO) = (
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)

# (name, edge colour, centre colour): cyan, turquoise, pale blue, lavender, violet, soft pink
JPALETTES = [
    ("cyan", (0.08, 0.82, 0.95), (0.55, 1.00, 0.95)),
    ("turquoise", (0.10, 0.80, 0.70), (0.60, 1.00, 0.88)),
    ("pale blue", (0.42, 0.70, 1.00), (0.85, 0.95, 1.00)),
    ("lavender", (0.62, 0.45, 1.00), (0.88, 0.78, 1.00)),
    ("violet", (0.50, 0.25, 0.95), (0.78, 0.62, 1.00)),
    ("pink", (1.00, 0.48, 0.76), (1.00, 0.86, 0.93)),
    ("aqua-violet", (0.12, 0.72, 0.90), (0.72, 0.55, 1.00)),
]

# Everything that differs between the four size classes. Sizes are relative to the previous
# build's large animals (scale ~2.0): tiny 10-13%, small 14-21%, medium 25-39%, large 45-55%.
# Dot counts / dot sizes / tentacle density follow the area, so every animal keeps the same
# delicate pointillist texture at its new size.
JTIERS = {
    # one or two bigger animals, drifting in the middle distance / background
    "large": dict(scale=(0.90, 1.10), layer=0.32, speed=9.0, period=(3.8, 5.0), dv=8.0, bright=0.95,
                  surf=600, sr=(0.9, 1.5), sa=(0.20, 0.40), rim=(78, 2), canals=(12, 14),
                  gon=(16, 90), arms=(4, 40), tent=(16, 22.0, (1.8, 3.8), 3), fil=(5, 14.0, (3.2, 5.4)),
                  tr=(0.75, 1.15), motes=30, halo=10, y=(150, 330), rim_a=0.75, cloud=(200, 0.30), ds=1.0),
    # medium animals: the closest ones, nearest the glass
    "medium": dict(scale=(0.50, 0.78), layer=0.10, speed=12.0, period=(3.0, 4.2), dv=6.5, bright=1.05,
                   surf=260, sr=(0.8, 1.2), sa=(0.22, 0.42), rim=(52, 2), canals=(8, 10),
                   gon=(12, 60), arms=(4, 26), tent=(12, 15.0, (1.8, 3.8), 2), fil=(3, 10.0, (3.0, 5.0)),
                   tr=(0.70, 1.05), motes=16, halo=5, y=(130, 420), rim_a=0.70, cloud=(110, 0.30), ds=0.9),
    # small, farther away
    "small": dict(scale=(0.28, 0.42), layer=0.58, speed=7.0, period=(2.2, 3.2), dv=4.0, bright=0.88,
                  surf=120, sr=(0.7, 1.1), sa=(0.24, 0.44), rim=(34, 1), canals=(0, 0),
                  gon=(0, 50), arms=(0, 0), tent=(8, 10.0, (2.0, 4.0), 1), fil=(0, 0.0, (0, 0)),
                  tr=(0.60, 0.90), motes=8, halo=2, y=(100, 450), rim_a=0.55, cloud=(50, 0.32), ds=0.85),
    # tiny distant glimmers
    "tiny": dict(scale=(0.18, 0.26), layer=0.88, speed=4.0, period=(1.8, 2.6), dv=2.5, bright=0.72,
                 surf=70, sr=(0.6, 0.95), sa=(0.26, 0.46), rim=(24, 1), canals=(0, 0),
                 gon=(0, 30), arms=(0, 0), tent=(6, 7.0, (2.0, 4.0), 0), fil=(0, 0.0, (0, 0)),
                 tr=(0.55, 0.80), motes=4, halo=0, y=(90, 470), rim_a=0.50, cloud=(30, 0.32), ds=0.80),
}

# (tier, x, y, scale, palette) - hand-placed in loose groups with dark water between them.
# 2 large, 6 medium, 5 small, 4 tiny. Two start off-screen on the right and drift in.
JSTART = [
    ("large", 455, 235, 1.00, 2), ("large", 885, 300, 0.92, 3),
    ("medium", 600, 335, 0.62, 0), ("medium", 350, 178, 0.55, 6), ("medium", 770, 170, 0.68, 5),
    ("medium", 1000, 180, 0.58, 1), ("medium", 1190, 230, 0.70, 4),
    ("small", 540, 140, 0.36, 0), ("small", 300, 150, 0.32, 3), ("small", 700, 412, 0.34, 2),
    ("small", 960, 200, 0.38, 6), ("medium", 655, 240, 0.52, 1), ("small", 1110, 130, 0.34, 5),
    ("tiny", 655, 118, 0.22, 4), ("tiny", 845, 345, 0.24, 0), ("tiny", 395, 112, 0.20, 5),
    ("tiny", 945, 228, 0.22, 2),
]


def _jrows(kind, jid, u, v, b, rad, alpha, cmix, rng):
    """one block of dots; columns = kind, jelly, u, v, b0..b3, radius, alpha, colour-mix, phase"""
    n = max(np.size(u), np.size(v))
    full = lambda a: np.broadcast_to(np.asarray(a, np.float32), (n,))
    u = full(u)
    b = np.broadcast_to(np.asarray(b, np.float32), (n, 4)) if np.ndim(b) else np.zeros((n, 4), np.float32)
    return np.column_stack([np.full(n, kind, np.float32), np.full(n, jid, np.float32), u, full(v), b,
                            full(rad), full(alpha), full(cmix), rng.uniform(0, 6.2832, n)]).astype(np.float32)


def jelly_dots(rng, jid, T):
    """Build one jellyfish out of dots. Pointillism: every part is its own population of dots
    with its own size, density and brightness - there is no mesh and no image anywhere."""
    out = []
    TAU = 2 * math.pi
    # --- translucent bell: dots spread evenly over the dome, so the silhouette edges end up
    #     denser and brighter on their own (the dome is seen edge-on there)
    n = T["surf"]
    phi = np.arccos(rng.random(n) * 0.985)
    th = rng.uniform(0, TAU, n)
    cm = np.clip(0.95 - phi / 1.57 * 0.85 + rng.normal(0, 0.10, n), 0, 1)
    cm[rng.random(n) < 0.07] = 2.0                                  # white-blue sparkles
    out.append(_jrows(JK_SURF, jid, phi, th, 0, rng.uniform(*T["sr"], n), rng.uniform(*T["sa"], n), cm, rng))
    # --- glowing margin: a flowing ring of brighter, larger dots (+ a hanging fringe row)
    nr, rows = T["rim"]
    for r in range(rows):
        th = (np.arange(nr) + rng.uniform(-0.4, 0.4, nr)) / nr * TAU
        cm = rng.uniform(0.0, 0.30, nr)
        cm[rng.random(nr) < 0.06] = 2.0
        out.append(_jrows(JK_RIM, jid, 0, th, [2.4 * r, 0, 0, 0], rng.uniform(1.5, 2.5, nr) * (1 - 0.15 * r),
                          rng.uniform(0.50, 0.85, nr) * (1 - 0.25 * r) * T["rim_a"], cm, rng))
    # --- delicate radial canals running apex -> margin just inside the surface
    nc, per = T["canals"]
    for k in range(nc):
        tk = (k + rng.uniform(-0.25, 0.25)) / nc * TAU
        ph = np.sort(rng.uniform(0.25, 1.50, per))
        out.append(_jrows(JK_CANAL, jid, ph, tk, 0, rng.uniform(0.9, 1.4, per), rng.uniform(0.28, 0.48, per), 0.55, rng))
    # --- bright core: four gonad rings (seen through the bell) + a soft cloud of centre dots
    ng, ncloud = T["gon"]
    if ng:
        for k in range(4):
            psi = math.pi / 4 + k * math.pi / 2 + rng.uniform(-0.2, 0.2)
            a = rng.uniform(0, TAU, ng)
            x = 0.30 * math.cos(psi) + 0.13 * np.cos(a)
            z = 0.30 * math.sin(psi) + 0.13 * np.sin(a)
            out.append(_jrows(JK_CORE, jid, x, z, [0.38, 1.0, 0, 0], rng.uniform(1.1, 1.6, ng),
                              rng.uniform(0.40, 0.65, ng), 0.75, rng))
    if ncloud:
        rho = np.abs(rng.normal(0, 0.17, ncloud))
        a = rng.uniform(0, TAU, ncloud)
        out.append(_jrows(JK_CORE, jid, rho * np.cos(a), rho * np.sin(a),
                          np.column_stack([rng.uniform(0.2, 0.65, ncloud), rng.uniform(0.6, 1.0, ncloud),
                                           np.zeros(ncloud), np.zeros(ncloud)]),
                          rng.uniform(1.0, 1.7, ncloud), rng.uniform(0.20, 0.38, ncloud), 0.7, rng))
    # a wide, dim glow of tiny dots filling the bell: the soft colour gradient inside the translucent body
    nfill, sig = T["cloud"]
    if nfill:
        rho = np.minimum(np.abs(rng.normal(0, sig, nfill)), 0.85)
        a = rng.uniform(0, TAU, nfill)
        out.append(_jrows(JK_CORE, jid, rho * np.cos(a), rho * np.sin(a),
                          np.column_stack([rng.uniform(0.10, 0.80, nfill), rng.uniform(0.35, 0.60, nfill),
                                           np.zeros(nfill), np.zeros(nfill)]),
                          rng.uniform(0.9, 1.5, nfill), rng.uniform(0.14, 0.28, nfill), rng.uniform(0.2, 0.8, nfill), rng))
    # --- ruffled oral arms: ribbons of dots hanging from the centre
    na, per = T["arms"]
    for k in range(na):
        ang = k * TAU / na + rng.uniform(-0.3, 0.3)
        t = np.clip((np.arange(per) + rng.uniform(-0.4, 0.4, per)) / per, 0, 1)
        w = rng.uniform(-1, 1, per)
        out.append(_jrows(JK_ARM, jid, t, ang, np.column_stack([np.full(per, rng.uniform(0.9, 1.4)), w,
                          np.full(per, rng.uniform(0, 6.28)), np.zeros(per)]),
                          rng.uniform(0.9, 1.5, per), rng.uniform(0.35, 0.60, per), 0.8, rng))
    # --- long, sparse tentacles: each with its own length, wave speed and wavelength,
    #     plus a few luminous beads sliding along it
    nt, per_unit, (l0, l1), nbeads = T["tent"]
    for k in range(nt):
        th = (k + rng.uniform(-0.45, 0.45)) / nt * TAU
        L = rng.uniform(l0, l1)
        prm = [L, rng.uniform(0, 6.28), rng.uniform(0.6, 1.4), rng.uniform(1.0, 2.0)]
        n = max(int(per_unit * L), 8)
        t = np.clip((np.arange(n) + rng.uniform(-0.35, 0.35, n)) / n, 0, 1)
        rad = rng.uniform(*T["tr"], n) * (1 - 0.4 * t)
        out.append(_jrows(JK_TENT, jid, t, th, prm, rad, rng.uniform(0.30, 0.50, n), 0.15 + 0.5 * t, rng))
        if nbeads:
            tb = rng.uniform(0.1, 0.95, nbeads)
            out.append(_jrows(JK_TENT, jid, tb, th, prm, rng.uniform(1.6, 2.4, nbeads), 0.75, 1.0, rng))
    # --- trailing filaments: very faint, very long, whipping around more than the tentacles
    nf, per_unit, (l0, l1) = T["fil"]
    for k in range(nf):
        th = rng.uniform(0, TAU)
        L = rng.uniform(l0, l1)
        prm = [L, rng.uniform(0, 6.28), rng.uniform(0.5, 1.1), rng.uniform(1.2, 2.4)]
        n = max(int(per_unit * L), 8)
        t = np.clip((np.arange(n) + rng.uniform(-0.4, 0.4, n)) / n, 0, 1)
        out.append(_jrows(JK_FIL, jid, t, th, prm, rng.uniform(0.7, 1.1, n), rng.uniform(0.15, 0.28, n), 0.3, rng))
    # --- tiny luminous motes orbiting the animal
    nm = T["motes"]
    if nm:
        r = rng.uniform(0.9, 2.3, nm)
        a = rng.uniform(0, TAU, nm)
        cm = rng.uniform(0.0, 1.0, nm)
        cm[rng.random(nm) < 0.2] = 2.0
        out.append(_jrows(JK_MOTE, jid, r * np.cos(a), r * np.sin(a) * 1.2 + 0.4, 0,
                          rng.uniform(0.9, 1.8, nm), rng.uniform(0.40, 0.80, nm), cm, rng))
    # --- soft halo: a loose scatter of very faint, large blurry dots (never one solid disc)
    nh = T["halo"]
    if nh:
        r = rng.uniform(0.2, 1.7, nh)
        a = rng.uniform(0, TAU, nh)
        out.append(_jrows(JK_HALO, jid, r * np.cos(a), r * np.sin(a) * 1.1 - 0.1, 0,
                          rng.uniform(15, 30, nh) * (0.6 if T["layer"] > 0.5 else 1.0),
                          rng.uniform(0.030, 0.060, nh), 0.4, rng))
    arr = np.vstack(out)
    arr[arr[:, 0] != JK_HALO, 8] *= T["ds"]
    return arr


def pulse_shape(x):
    """0 = relaxed, 1 = fully contracted: a quick squeeze, then a long graceful relax"""
    if x < 0.30:
        return math.sin(0.5 * math.pi * x / 0.30)
    return math.cos(0.5 * math.pi * (x - 0.30) / 0.70) ** 2


class Jelly:
    """state of one animal; every one has its own rhythm, size, colour and character"""
    __slots__ = ("slot", "tier", "pos", "drift", "pvel", "scale", "layer", "period", "dv", "phase", "lean",
                 "bclk", "tclk", "tspeed", "sway", "pal", "bright", "y_pref", "seed", "speed", "extent")


class JellySystem:
    def __init__(self, seed=SEED):
        self.seed = seed
        rng = np.random.default_rng(seed + 101)
        self.tiers = [s[0] for s in JSTART]
        self.crest = None
        self.dots = np.ascontiguousarray(
            np.vstack([jelly_dots(rng, i, JTIERS[t]) for i, t in enumerate(self.tiers)]), np.float32)
        self.reset()

    def set_crest(self, crest):
        """give the animals the skyline of the near corals so they drift above it"""
        self.crest = crest

    def reset(self):
        self.rng = np.random.default_rng(self.seed + 7)
        self.t = 0.0
        self.jellies = []
        for i, (tier, x, y, s, pal) in enumerate(JSTART):
            j = Jelly()
            j.slot, j.tier = i, tier
            j.pos = np.array([x, y], float)
            j.scale, j.pal = s, pal
            self._personalise(j)
            j.y_pref = y
            j.phase = self.rng.random()                       # not synchronised
            self.jellies.append(j)

    # ---- per-animal character
    def _personalise(self, j):
        r, T = self.rng, JTIERS[j.tier]
        j.layer = max(0.0, T["layer"] + r.uniform(-0.05, 0.05))
        j.period = r.uniform(*T["period"])
        j.dv = T["dv"] * r.uniform(0.8, 1.2)
        j.speed = T["speed"] * r.uniform(0.75, 1.25)
        j.bright = T["bright"] * r.uniform(0.85, 1.1)
        j.sway = r.uniform(0.75, 1.3)
        j.tspeed = r.uniform(0.45, 0.85)
        j.seed = r.uniform(0, 100)
        j.drift = np.array([-j.speed, 0.0])
        j.pvel = np.zeros(2)
        j.lean = r.uniform(-0.3, 0.3)
        j.bclk = r.uniform(0, 50)
        j.tclk = r.uniform(0, 50)
        j.extent = JR0 * j.scale * 1.35

    def _respawn(self, j):
        """an animal that has drifted out on the left re-enters later from the right, as a
        'new' one: new size, colour and rhythm, in a free spot with dark water around it"""
        r, T = self.rng, JTIERS[j.tier]
        j.scale = r.uniform(*T["scale"])
        j.pal = int(r.choice([p for p in range(len(JPALETTES)) if p != j.pal]))
        self._personalise(j)
        best, best_score = None, -1.0
        for _ in range(18):
            y = r.uniform(*T["y"])
            x = 1060 + j.extent + r.uniform(40, 380)
            d = min(math.hypot(x - o.pos[0], y - o.pos[1]) / (o.extent + j.extent) for o in self.jellies if o is not j)
            if d > best_score:
                best, best_score = (x, y), d
        j.pos = np.array(best, float)
        j.y_pref = best[1]

    # ---- simulation (all rates use dt)
    def update(self, dt):
        if dt <= 0:
            return
        self.t += dt
        for j in self.jellies:
            self._step(j, dt)

    def simulate(self, t_end, step=1 / 30.0):
        self.reset()
        while self.t < t_end:
            self.update(min(step, t_end - self.t))

    def _step(self, j, dt):
        t, x, y = self.t, j.pos[0], j.pos[1]
        # the slow current (shared, so neighbours move together) pulls the animal along
        cur = np.array([-(1.0 + 0.35 * math.sin(t * 0.05 + y * 0.006 + 1.0)), 0.20 * math.sin(t * 0.07 + x * 0.005)])
        j.drift += (cur * j.speed - j.drift) * (1 - math.exp(-dt * 0.6))
        # the pulse: a squeeze pushes the animal along its own axis, then it glides
        j.phase = (j.phase + dt / j.period) % 1.0
        if j.phase < 0.30:
            thrust = j.dv / (0.191 * j.period) * math.sin(math.pi * j.phase / 0.30)
            j.pvel += np.array([math.sin(j.lean), -math.cos(j.lean)]) * thrust * dt
        j.pvel[1] += (2.2 + (j.y_pref - y) * 0.012) * dt              # slow sink + altitude steering
        if self.crest is not None:                                   # float above the near corals, not behind them
            cap = self.crest[min(max(int(x // 8), 0), len(self.crest) - 1)] - (JR0 * j.scale * 0.9 + 10.0)
            if cap < 1e8:
                j.y_pref = min(j.y_pref, max(cap, 95.0))
                if y > cap:
                    j.pvel[1] -= (y - cap) * 0.05 * dt
        ymax = JTIERS[j.tier]["y"][1] + 70
        if y > ymax:
            j.pvel[1] -= (y - ymax) * 0.08 * dt                     # a gentle push back up
        j.pvel *= math.exp(-dt * 0.35)
        j.y_pref = min(max(j.y_pref + self.rng.normal(0, 3.0) * dt, JTIERS[j.tier]["y"][0] - 30),
                       JTIERS[j.tier]["y"][1] + 30)
        j.pos += (j.drift + j.pvel) * dt
        target = max(-0.5, min(0.5, j.drift[0] * 0.02)) + 0.2 * math.sin(t * 0.13 + j.seed)
        j.lean += (target - j.lean) * (1 - math.exp(-dt * 0.6))
        j.bclk += dt * 0.9
        j.tclk += dt * j.tspeed
        if j.pos[0] + j.extent < 190:
            self._respawn(j)

    def uniforms(self):
        """per-frame numbers for the shaders (arrays of NJ vec4)"""
        A, B, C, D, E, G = (np.zeros((NJ, 4), np.float32) for _ in range(6))
        for i, j in enumerate(self.jellies):
            c = pulse_shape(j.phase)
            c1, c2 = JPALETTES[j.pal][1], JPALETTES[j.pal][2]
            A[i] = (j.pos[0], j.pos[1], j.scale, j.layer)
            B[i] = (c, j.bclk, j.lean, j.sway)
            C[i] = (*c1, 1.0)
            D[i] = (*c2, j.tclk)
            v = j.drift + j.pvel
            E[i] = (v[0], v[1], j.seed, j.bright)
            mix = [0.6 * a + 0.4 * b for a, b in zip(c1, c2)]
            G[i] = (*mix, j.bright * (0.8 + 0.4 * c))                 # light they give to the water
        return dict(u_jA=A, u_jB=B, u_jC=C, u_jD=D, u_jE=E, u_jG=G)


# ----------------------------------------------------------------------------
# 6e. THE LUMINOUS CORAL GARDEN (PHASE 5)
#     Same idea as the jellyfish system: nothing here is a mesh or an image.
#       - gen_*() functions grow each coral out of individually placed dots: ribs and
#         branches are polylines sampled into dots, with thickness, lit centre lines,
#         optical colour mixing, translucent "membrane" dots and glowing tip dots.
#       - every dot remembers a SWAY WEIGHT (how far it is from the coral's root) plus the
#         phase of its own branch, so the vertex shader can make each branch move on its own.
#       - CoralGarden lays everything out: big near clusters at the left/right edges, mid and
#         far clusters behind them, an open sandy path through the middle, rocks, light shafts,
#         drifting motes and the list of glow sources that light nearby particles.
# ----------------------------------------------------------------------------
NK = 14                                    # glow sources (must equal NK in the GLSL)
CK_BODY, CK_TIP, CK_MOTE, CK_HALO, CK_RAY, CK_FLOOR = 0.0, 1.0, 2.0, 3.0, 4.0, 5.0
LY_NEAR, LY_MID, LY_FAR = 0.12, 0.50, 0.85  # depth layers: 0 = right at the glass, 1 = far away
CORAL_FADE = (0.56, 0.80)                  # depth range over which the garden fades in
PATH_CX, PATH_Y0 = 640.0, 402.0            # the sandy corridor: centre line + where it starts
# light shafts that fall onto selected corals: x where the shaft starts (y=54), width, strength, phase
CORAL_BEAMS = np.array([[168.0, 70.0, 1.0, 0.5], [825.0, 64.0, 0.9, 2.1], [490.0, 56.0, 0.5, 4.0]], np.float32)


def path_hw(y):
    """half-width of the open corridor at screen height y (it opens up toward the viewer)"""
    return 38.0 + np.maximum(np.asarray(y, float) - PATH_Y0, 0.0) * 1.45


# base / mid / highlight colours along a limb, and the colour of its glowing tips
CPAL = {
    "fan_violet": dict(base=(0.30, 0.14, 0.55), mid=(0.58, 0.38, 0.92), hi=(0.92, 0.40, 0.74), tip=(1.00, 0.78, 0.42)),
    "fan_pink": dict(base=(0.40, 0.12, 0.50), mid=(0.85, 0.40, 0.80), hi=(1.00, 0.62, 0.82), tip=(1.00, 0.70, 0.50)),
    "fan_lav": dict(base=(0.26, 0.18, 0.58), mid=(0.52, 0.42, 0.95), hi=(0.74, 0.62, 1.00), tip=(0.98, 0.82, 0.52)),
    "branch_purple": dict(base=(0.20, 0.10, 0.42), mid=(0.50, 0.26, 0.78), hi=(0.95, 0.45, 0.72), tip=(1.00, 0.78, 0.45)),
    "branch_pink": dict(base=(0.34, 0.12, 0.40), mid=(0.80, 0.32, 0.62), hi=(1.00, 0.60, 0.74), tip=(1.00, 0.80, 0.55)),
    "anem_pink": dict(base=(0.40, 0.10, 0.45), mid=(0.90, 0.35, 0.70), hi=(1.00, 0.65, 0.85), tip=(1.00, 0.76, 0.48)),
    "anem_cyan": dict(base=(0.10, 0.25, 0.55), mid=(0.30, 0.65, 0.95), hi=(0.65, 0.95, 1.00), tip=(0.72, 1.00, 0.96)),
    "anem_violet": dict(base=(0.25, 0.12, 0.55), mid=(0.58, 0.40, 0.95), hi=(0.85, 0.70, 1.00), tip=(1.00, 0.72, 0.88)),
    "soft_pink": dict(base=(0.42, 0.18, 0.50), mid=(0.95, 0.48, 0.74), hi=(1.00, 0.68, 0.84), tip=(1.00, 0.84, 0.58)),
    "soft_peach": dict(base=(0.42, 0.18, 0.42), mid=(0.96, 0.55, 0.62), hi=(1.00, 0.74, 0.70), tip=(1.00, 0.84, 0.56)),
    "mush_lav": dict(base=(0.30, 0.18, 0.60), mid=(0.62, 0.46, 0.96), hi=(0.90, 0.55, 0.88), tip=(0.58, 0.92, 1.00)),
    "mush_teal": dict(base=(0.10, 0.28, 0.50), mid=(0.28, 0.62, 0.85), hi=(0.62, 0.95, 1.00), tip=(0.86, 0.80, 1.00)),
    "plant": dict(base=(0.08, 0.30, 0.40), mid=(0.15, 0.58, 0.62), hi=(0.55, 0.92, 0.90), tip=(0.76, 1.00, 0.95)),
    "pearl": dict(base=(0.40, 0.45, 0.75), mid=(0.70, 0.78, 1.00), hi=(0.96, 0.95, 1.00), tip=(1.00, 0.86, 0.95)),
    "rock": dict(base=(0.045, 0.035, 0.13), mid=(0.11, 0.08, 0.26), hi=(0.28, 0.30, 0.56), tip=(0.45, 0.65, 0.90)),
}
CPAL_MIX = dict(cool=np.array([0.20, 0.55, 0.85]), gold=np.array([1.00, 0.75, 0.40]), viol=np.array([0.55, 0.30, 0.85]))


def _grad(pal, s):
    s = np.clip(np.asarray(s, float), 0, 1)[..., None]
    b, m, h = (np.array(pal[k], float) for k in ("base", "mid", "hi"))
    lo = b + (m - b) * np.clip(s / 0.55, 0, 1)
    up = m + (h - m) * np.clip((s - 0.55) / 0.45, 0, 1)
    return np.where(s < 0.55, lo, up)


def _crow(x, y, rad, alpha, rgb, glow, sw, amp, ph, fr, pulse, layer, kind, extra=0.0):
    """one block of coral dots: x y radius alpha | r g b glow | sway-weight amp phase freq | pulse layer kind extra"""
    x = np.atleast_1d(np.asarray(x, np.float32))
    n = len(x)
    B = lambda a: np.broadcast_to(np.asarray(a, np.float32), (n,))
    rgb = np.broadcast_to(np.asarray(rgb, np.float32), (n, 3))
    return np.column_stack([x, B(y), B(rad), B(alpha), np.clip(rgb, 0, 1.5), B(glow), B(sw), B(amp), B(ph), B(fr),
                            B(pulse), B(layer), B(kind), B(extra)]).astype(np.float32)


def _sample(P, spacing, rng, jit=0.3):
    """dots at (jittered) even spacing along a polyline -> positions, 0..1 progress, unit tangents"""
    P = np.asarray(P, float)
    d = np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1]))
    cum = np.r_[0.0, np.cumsum(d)]
    total = cum[-1]
    if total < 1.0:
        return np.zeros((0, 2)), np.zeros(0), np.zeros((0, 2))
    n = max(int(total / spacing), 2)
    s = np.clip((np.arange(n) + rng.uniform(-jit, jit, n)) / n, 0, 1) * total
    f = lambda q, k: np.interp(np.clip(q, 0, total), cum, P[:, k])
    x, y = f(s, 0), f(s, 1)
    tx, ty = f(s + 1.5, 0) - f(s - 1.5, 0), f(s + 1.5, 1) - f(s - 1.5, 1)
    ln = np.hypot(tx, ty) + 1e-6
    return np.column_stack([x, y]), s / total, np.column_stack([tx / ln, ty / ln])


def _bez(p0, p1, p2, n=28):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2 = (np.asarray(p, float) for p in (p0, p1, p2))
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def _rib(base, ang, L, bend):
    """a gently curved limb leaving `base` at angle `ang` (0 = straight up, + = to the right)"""
    base = np.asarray(base, float)
    d = np.array([math.sin(ang), -math.cos(ang)])
    nrm = np.array([-d[1], d[0]])
    return _bez(base, base + d * L * 0.55 + nrm * bend * L, base + d * L)


class _Ctx:
    """everything one coral needs while it is being grown"""

    def __init__(self, G, front, layer, pal):
        self.G, self.rng, self.front, self.layer = G, G.rng, front, layer
        self.pal = CPAL[pal]
        self.ds = 1.0 - 0.45 * layer                       # far corals use finer, smaller dots
        self.phase = G.rng.uniform(0, 6.28)
        self.pulse = G.rng.uniform(0, 6.28)                # each coral glows on its own slow rhythm
        self.tips = []
        self.name = pal
        self.box = [1e9, 1e9, -1e9, -1e9]                  # footprint (x0, y0, x1, y1), used by the layout review
        self.cells = []                                     # occupied 8 px cells, for overlap measurements

    def _grow(self, rows):
        b = self.box
        b[0], b[1] = min(b[0], float(rows[:, 0].min())), min(b[1], float(rows[:, 1].min()))
        b[2], b[3] = max(b[2], float(rows[:, 0].max())), max(b[3], float(rows[:, 1].max()))
        if rows[:, 14].max() < CK_MOTE:
            self.cells.append((rows[:, 0] // 8).astype(np.int32) * 1000 + (rows[:, 1] // 8).astype(np.int32) + 500)

    def body(self, rows):
        self._grow(rows)
        (self.G.fb if self.front else self.G.bb).append(rows)

    def glow(self, rows):
        self._grow(rows)
        (self.G.fg if self.front else self.G.bg).append(rows)

    def done(self, weight=1.0, cluster=True):
        self.G.formations.append(dict(name=self.name, layer=self.layer, front=self.front, box=tuple(self.box),
                                      cells=np.unique(np.concatenate(self.cells)) if self.cells else np.zeros(0, np.int32)))
        if self.tips:
            t = np.array(self.tips)
            c = t.mean(0)
            R = 55.0 * self.ds + float(np.hypot(*(t - c).T).max()) * 0.8
            col = np.array(self.pal["tip"]) * 0.65 + np.array(self.pal["hi"]) * 0.35
            self.G.sources.append((float(c[0]), float(c[1]), R, weight * min(1.0, len(t) / 8.0) * (1.0 - 0.4 * self.layer),
                                   col, self.pulse))
        if cluster and self.tips:
            c = np.array(self.tips).mean(0)
            self.G.clusters.append((float(c[0]), float(c[1]), self.layer))


def limb(C, P, spacing=3.6, w0=4.0, w1=1.2, s0=0.0, s1=1.0, amp=6.0, sw0=0.0, sw1=1.0, ph=None, fr=None,
         rad=(1.8, 2.8), alpha=(0.80, 0.96), hi=0.2):
    """a stroke of dots with thickness: lit centre line, darker flanks, optical colour mixing"""
    rng, ds = C.rng, C.ds
    pts, s, tan = _sample(P, spacing * ds, rng)
    if len(pts) == 0:
        return None
    m = len(pts)
    w = (w0 + (w1 - w0) * s) * ds
    k = np.maximum(1, np.round(w / (2.4 * ds)).astype(int))
    idx = np.repeat(np.arange(m), k)
    j = np.arange(len(idx)) - np.repeat(np.cumsum(k) - k, k)
    lat = ((j + 0.5) / k[idx] - 0.5) * w[idx]
    nrm = np.column_stack([-tan[:, 1], tan[:, 0]])
    n = len(idx)
    pos = pts[idx] + nrm[idx] * lat[:, None] + rng.normal(0, 0.35 * ds, (n, 2))
    sg = s0 + (s1 - s0) * s[idx]
    edge = np.abs(lat) / np.maximum(w[idx] / 2, 0.6)
    rgb = _grad(C.pal, sg) * (1.0 + hi * (1 - edge))[:, None]
    rgb = pointillize(rgb, 0.65 + 0.5 * (1 - edge), rng, 0.6, **CPAL_MIX)
    rgb[rng.random(n) < 0.10] *= 0.55                                  # shadow dots
    ph = C.phase + rng.normal(0, 0.5) if ph is None else ph
    fr = rng.uniform(0.5, 0.9) if fr is None else fr
    sw = sw0 + (sw1 - sw0) * s[idx]
    C.body(_crow(pos[:, 0], pos[:, 1], rng.uniform(*rad, n) * ds, rng.uniform(*alpha, n), rgb,
                 (sg > 0.78) * 0.8, sw, amp, ph, fr, C.pulse, C.layer, CK_BODY))
    return dict(end=pts[-1], ph=ph, fr=fr, amp=amp, sw=sw1)


def tip(C, x, y, n=4, spread=2.4, col=None, rad=(2.0, 3.2), halo=0.30, sw=1.0, amp=6.0, ph=0.0, fr=0.7, strength=1.0):
    """a small cluster of luminous dots (+ sometimes a faint halo): the glowing coral tip"""
    rng, ds = C.rng, C.ds
    col = np.array(C.pal["tip"] if col is None else col) * 0.82
    pts = np.array([x, y]) + rng.normal(0, spread * ds, (n, 2))
    C.glow(_crow(pts[:, 0], pts[:, 1], rng.uniform(*rad, n) * ds, rng.uniform(0.45, 0.80, n) * strength,
                 col * rng.uniform(0.85, 1.1, (n, 1)), 1.0, sw, amp, ph, fr, C.pulse, C.layer, CK_TIP))
    if rng.random() < halo:
        C.glow(_crow([x], [y], rng.uniform(9, 16) * ds, rng.uniform(0.05, 0.09) * strength, col, 1.0, sw, amp, ph, fr,
                     C.pulse, C.layer, CK_HALO))
    C.tips.append((x, y))


def tuft(C, x, y, n=16, r=8.0, amp=8.0, sw=1.0, ph=0.0, fr=0.8):
    """a fuzzy puff of polyps (soft corals): tiny dots sprayed around a branch end"""
    rng, ds = C.rng, C.ds
    a = rng.uniform(0, 2 * math.pi, n)
    d = r * ds * np.sqrt(rng.random(n))
    pos = np.column_stack([x + d * np.cos(a), y + d * np.sin(a) * 0.9])
    rgb = pointillize(_grad(C.pal, rng.uniform(0.55, 1.0, n)), 0.9, rng, 0.8, **CPAL_MIX)
    C.body(_crow(pos[:, 0], pos[:, 1], rng.uniform(1.0, 1.7, n) * ds, rng.uniform(0.55, 0.9, n), rgb, 0.5, sw, amp,
                 ph + rng.normal(0, 0.3, n), fr, C.pulse, C.layer, CK_BODY))
    k = max(n // 7, 1)
    C.glow(_crow(pos[:k, 0], pos[:k, 1], rng.uniform(1.3, 2.0, k) * ds, rng.uniform(0.28, 0.50, k),
                 np.array(C.pal["tip"]) * rng.uniform(0.85, 1.1, (k, 1)), 1.0, sw, amp, ph, fr, C.pulse, C.layer, CK_TIP))


# ------------------------------------------------------------------ the corals
def gen_fan(C, cx, cy, H, W, tilt=0.0):
    """broad flowing sea fan: radial ribs + side branches + a lattice of cross-links + translucent web"""
    rng, ds = C.rng, C.ds
    half = float(np.clip(math.atan2(W / 2, H) * 1.55, 0.5, 1.25))
    n = int(np.clip(10 + W / (20 / max(ds, 0.6)), 9, 21))
    a = np.sort(np.linspace(-1, 1, n) + rng.uniform(-0.3, 0.3, n) * (2.0 / n))
    amp = 7.0 * (0.5 + 0.5 * H / 200.0)
    fr0 = rng.uniform(0.5, 0.75)
    ribs = []
    for ai in a:
        ang = tilt + ai * half
        L = H * (0.62 + 0.38 * math.cos(ai * math.pi / 2)) * rng.uniform(0.94, 1.04)
        P = _rib((cx, cy), ang, L, rng.uniform(-0.10, 0.10))
        ph = C.phase + ai * 0.7 + rng.normal(0, 0.12)            # phases flow smoothly across the fan
        ribs.append((P, ph))
        info = limb(C, P, 3.2, 7.5, 2.2, amp=amp, ph=ph, fr=fr0, hi=0.28)
        tj = 0.20
        while tj < 0.90:
            bp = P[int(tj * (len(P) - 1))]
            for side in (-1, 1):
                if rng.random() < 0.8:
                    bl = L * 0.30 * (1 - tj) ** 0.55 * rng.uniform(0.8, 1.15)
                    if bl < 8:
                        continue
                    PB = _rib(bp, ang + side * rng.uniform(0.35, 0.70), bl, rng.uniform(-0.1, 0.1))
                    inf = limb(C, PB, 3.5, 3.2, 1.2, s0=tj, s1=1.0, amp=amp, sw0=tj, sw1=min(1.0, tj + 0.2),
                               ph=ph + rng.normal(0, 0.25), fr=fr0, rad=(1.5, 2.3), alpha=(0.7, 0.9))
                    if inf is not None and rng.random() < 0.55:
                        tip(C, inf["end"][0], inf["end"][1], 2, 1.6, rad=(1.6, 2.4), halo=0.1, sw=inf["sw"],
                            amp=amp, ph=inf["ph"], fr=fr0, strength=0.9)
            tj += rng.uniform(0.12, 0.17)
        if info:
            tip(C, info["end"][0], info["end"][1], 5, 2.6, sw=1.0, amp=amp, ph=ph, fr=fr0)
    for r in (0.30, 0.44, 0.58, 0.72, 0.85, 0.96):                       # delicate lattice between the ribs
        for i in range(len(ribs) - 1):
            Pa, Pb = ribs[i][0], ribs[i + 1][0]
            A, B = Pa[int(r * (len(Pa) - 1))], Pb[int(r * (len(Pb) - 1))]
            mid = (A + B) / 2
            out = mid - np.array([cx, cy])
            mid = mid + out / (np.hypot(*out) + 1e-6) * 0.10 * np.hypot(*(B - A))
            limb(C, _bez(A, mid, B, 8), 3.8, 2.0, 2.0, s0=r * 0.5 + 0.2, s1=r * 0.5 + 0.2, amp=amp, sw0=r, sw1=r,
                 ph=(ribs[i][1] + ribs[i + 1][1]) / 2, fr=fr0, rad=(1.4, 2.1), alpha=(0.55, 0.80), hi=0.1)
    nm = int(min(0.012 * 0.5 * H * H * 2 * half / (ds ** 2), 1900))   # translucent web: flowing, flower-like body
    tt = rng.uniform(0.25, 0.97, nm) ** 0.8
    aa = rng.uniform(-1, 1, nm)
    ang = tilt + aa * half
    L_ = H * (0.62 + 0.38 * np.cos(aa * math.pi / 2))
    rgb = pointillize(_grad(C.pal, 0.30 + 0.45 * tt), 0.7, rng, 0.7, **CPAL_MIX)
    C.body(_crow(cx + np.sin(ang) * L_ * tt, cy - np.cos(ang) * L_ * tt, rng.uniform(1.5, 2.4, nm) * ds,
                 rng.uniform(0.32, 0.58, nm), rgb, 0.3, tt, amp, C.phase + aa * 0.7, fr0, C.pulse, C.layer, CK_BODY))
    limb(C, np.array([[cx, cy + 12], [cx, cy - 0.12 * H]]), 3.6, 11, 7, s0=0.0, s1=0.2, amp=0.0, sw0=0, sw1=0, hi=0.2)
    C.done(1.0)


def gen_branching(C, cx, cy, H, tilt=0.0, levels=4, amp=4.0):
    """a bushy branching coral: thick trunk, forking limbs, pink-gold glowing tips"""
    rng = C.rng

    def rec(p0, ang, L, w, lvl, ph):
        P = _rib(p0, ang, L, rng.uniform(-0.12, 0.12))
        h0 = float(np.clip((cy - p0[1]) / H, 0, 1))
        h1 = float(np.clip((cy - P[-1][1]) / H, 0, 1))
        fr = rng.uniform(0.5, 0.9)
        limb(C, P, 3.5, w, w * 0.72, s0=lvl / levels, s1=(lvl + 1) / levels, amp=amp, sw0=h0, sw1=h1, ph=ph, fr=fr, hi=0.4)
        end = P[-1]
        if lvl >= levels - 1 or L < 14:
            tip(C, end[0], end[1], 3, 2.0, sw=h1, amp=amp, ph=ph, fr=fr)
            tuft(C, end[0], end[1], 7, 5.0, amp, h1, ph, fr)
            return
        nch = 3 if rng.random() < 0.35 else 2
        offs = np.linspace(-1, 1, nch) * rng.uniform(0.35, 0.65) + rng.normal(0, 0.1, nch)
        for o in offs:
            rec(end, ang + o, L * rng.uniform(0.62, 0.80), w * 0.7, lvl + 1, ph + rng.normal(0, 0.35))
    rec((cx, cy), tilt, H * 0.34, 9.0, 0, C.phase)
    C.done(0.9)


def gen_soft_tree(C, cx, cy, H, tilt=0.0):
    """soft coral: thin swaying limbs ending in fluffy pastel polyp tufts"""
    rng = C.rng
    amp = 12.0 * (0.5 + 0.5 * H / 150.0)

    def rec(p0, ang, L, w, lvl, ph):
        P = _rib(p0, ang, L, rng.uniform(-0.18, 0.18))
        h0 = float(np.clip((cy - p0[1]) / H, 0, 1))
        h1 = float(np.clip((cy - P[-1][1]) / H, 0, 1))
        fr = rng.uniform(0.6, 1.0)
        limb(C, P, 3.6, w, w * 0.7, s0=0.1 + lvl * 0.2, s1=0.3 + lvl * 0.2, amp=amp, sw0=h0, sw1=h1, ph=ph, fr=fr,
             rad=(1.2, 2.0), alpha=(0.6, 0.85))
        end = P[-1]
        if lvl >= 3 or L < 16:
            tuft(C, end[0], end[1], int(22 * (0.6 + 0.4 * (1 - C.layer))), 11.0, amp, h1, ph, fr)
            tip(C, end[0], end[1], 2, 2.0, halo=0.25, sw=h1, amp=amp, ph=ph, fr=fr)
            return
        if lvl >= 1 and rng.random() < 0.5:
            tuft(C, P[len(P) // 2][0], P[len(P) // 2][1], 8, 6.0, amp, (h0 + h1) / 2, ph, fr)
        for o in np.linspace(-1, 1, 3 if rng.random() < 0.4 else 2) * rng.uniform(0.4, 0.7):
            rec(end, ang + o + rng.normal(0, 0.1), L * rng.uniform(0.60, 0.78), w * 0.7, lvl + 1, ph + rng.normal(0, 0.4))
    rec((cx, cy), tilt, H * 0.36, 6.5, 0, C.phase)
    C.done(0.9)


def gen_anemone(C, cx, cy, R):
    """sea anemone: a squat column crowned with a ring of independently swaying, glowing-tipped tentacles"""
    rng, ds = C.rng, C.ds
    n = int((18 + R * 1.3) / max(ds, 0.6))
    a = rng.uniform(0, 2 * math.pi, n)
    r = np.sqrt(rng.random(n))
    x, y = cx + np.cos(a) * r * R * 0.9, cy - 0.12 * R + np.sin(a) * r * R * 0.2
    rgb = pointillize(_grad(C.pal, 0.15 + 0.35 * r), 0.5, rng, 0.6, **CPAL_MIX)
    C.body(_crow(x, y, rng.uniform(1.5, 2.6, n) * ds, rng.uniform(0.7, 0.95, n), rgb, 0.0, 0.0, 0.0, 0.0, 0.0,
                 C.pulse, C.layer, CK_BODY))
    limb(C, np.array([[cx, cy + 4], [cx, cy - 0.28 * R]]), 3.2, R * 0.55, R * 0.42, s0=0.0, s1=0.2, amp=1.0, hi=0.2)
    for _ in range(int(18 + R * 0.9)):
        ring, phi = rng.uniform(0.15, 1.0), rng.uniform(0, 2 * math.pi)
        bx, by = cx + ring * R * 0.75 * math.cos(phi), cy - 0.25 * R + ring * R * 0.16 * math.sin(phi)
        ang = math.copysign(1, math.cos(phi)) * ring * rng.uniform(0.25, 0.7) + rng.normal(0, 0.12)
        P = _rib((bx, by), ang, R * rng.uniform(0.85, 1.5), rng.uniform(-0.25, 0.25))
        amp = rng.uniform(7, 13) * (0.5 + 0.5 * R / 50)
        ph, fr = rng.uniform(0, 6.28), rng.uniform(0.7, 1.3)
        inf = limb(C, P, 3.2, 3.6, 1.4, s0=0.1, s1=1.0, amp=amp, sw0=0.2, sw1=1.0, ph=ph, fr=fr, hi=0.35, rad=(1.3, 2.2))
        if inf:
            tip(C, inf["end"][0], inf["end"][1], 2, 1.0, rad=(1.7, 2.5), halo=0.05, sw=1.0, amp=amp, ph=ph, fr=fr, strength=0.65)
    C.done(1.0)


def gen_mushroom(C, cx, cy, w):
    """mushroom coral: a short stem under a ridged disc cap with a glowing rim"""
    rng, ds = C.rng, C.ds
    sh = w * rng.uniform(0.28, 0.5)
    limb(C, np.array([[cx + rng.normal(0, 1), cy + 4], [cx, cy - sh]]), 3.2, w * 0.30, w * 0.22, s0=0.0, s1=0.35,
         amp=0.0, hi=0.2)
    top = np.array([cx, cy - sh])
    rx = w / 2
    ry = rx * 0.34
    m = int(18 + w / 2)
    for k in range(m):                                                     # ridged gills, alternating colours
        a = 2 * math.pi * k / m + rng.normal(0, 0.04)
        r = np.linspace(0.08, 1.0, max(int(rx / (3.0 * ds)), 3))
        x = top[0] + math.cos(a) * rx * r
        y = top[1] + math.sin(a) * ry * r - 0.45 * ry * (1 - r ** 2)
        rgb = pointillize(_grad(C.pal, (0.35 + 0.55 * r) * (1.0 if k % 2 else 0.8)), 0.8, rng, 0.6, **CPAL_MIX)
        C.body(_crow(x, y, rng.uniform(1.2, 2.0, len(r)) * ds, rng.uniform(0.7, 0.95, len(r)), rgb, 0.4, 0.15, 1.0,
                     C.phase, 0.6, C.pulse, C.layer, CK_BODY))
    for t in (0.35, 0.7):                                                  # underside skirt gives the cap thickness
        a = np.linspace(0, math.pi, int(2 * rx / (3.2 * ds)))
        C.body(_crow(top[0] + np.cos(a) * rx, top[1] + np.sin(a) * ry + t * ry * 0.9, 1.7 * ds, 0.8,
                     _grad(C.pal, 0.12), 0.0, 0.1, 0.5, C.phase, 0.6, C.pulse, C.layer, CK_BODY))
    a = np.linspace(0, 2 * math.pi, int(2 * math.pi * rx / (7.0 * ds)), endpoint=False)
    C.glow(_crow(top[0] + np.cos(a) * rx, top[1] + np.sin(a) * ry - 0.0, rng.uniform(1.5, 2.2, len(a)) * ds,
                 rng.uniform(0.5, 0.8, len(a)), np.array(C.pal["tip"]), 1.0, 0.1, 0.5, C.phase, 0.6, C.pulse, C.layer, CK_TIP))
    C.tips.append((cx, top[1]))
    C.done(0.7)


def gen_bubble(C, cx, cy, n, R):
    """bubble coral: a cluster of small pearly spheres - tiny, intricate, no sway"""
    rng, ds = C.rng, C.ds
    for _ in range(n):
        c = np.array([cx + rng.normal(0, R * 1.2), cy - abs(rng.normal(0, R * 0.8))])
        r = R * rng.uniform(0.7, 1.2)
        m = max(int(2 * math.pi * r / (3.0 * ds)), 6)
        a = np.linspace(0, 2 * math.pi, m, endpoint=False)
        C.body(_crow(c[0] + np.cos(a) * r, c[1] + np.sin(a) * r, 1.4 * ds, 0.85,
                     pointillize(_grad(C.pal, 0.75 + 0.2 * np.sin(a)), 0.9, rng, 0.5, **CPAL_MIX), 0.2, 0.2, 1.5, C.phase, 0.5,
                     C.pulse, C.layer, CK_BODY))
        k = max(int(r * r / (5 * ds * ds)), 4)
        q = np.sqrt(rng.random(k)) * r * 0.9
        b = rng.uniform(0, 2 * math.pi, k)
        x, y = c[0] + q * np.cos(b), c[1] + q * np.sin(b)
        lit = 1 - np.clip(np.hypot(x - (c[0] - 0.35 * r), y - (c[1] - 0.4 * r)) / (1.4 * r), 0, 1)
        C.body(_crow(x, y, rng.uniform(1.0, 1.6, k) * ds, rng.uniform(0.35, 0.65, k),
                     pointillize(_grad(C.pal, 0.35 + 0.6 * lit), 0.8, rng, 0.5, **CPAL_MIX), 0.2, 0.2, 1.5, C.phase, 0.5,
                     C.pulse, C.layer, CK_BODY))
        C.glow(_crow([c[0] - 0.35 * r], [c[1] - 0.4 * r], 1.6 * ds, 0.8, (1.0, 0.97, 1.0), 1.0, 0.2, 1.0, C.phase, 0.5,
                     C.pulse, C.layer, CK_TIP))
    C.done(0.4, cluster=False)


def gen_plants(C, cx, cy, n, H, spread):
    """delicate sea plants: slender blades that sway the most"""
    rng = C.rng
    for _ in range(n):
        x0 = cx + rng.uniform(-spread, spread)
        P = _rib((x0, cy), rng.normal(0, 0.15), H * rng.uniform(0.55, 1.3), rng.uniform(-0.2, 0.2))
        ph, fr = rng.uniform(0, 6.28), rng.uniform(0.5, 0.95)
        amp = rng.uniform(10, 16)
        inf = limb(C, P, 3.8, 3.0, 0.9, amp=amp, sw0=0.0, sw1=1.0, ph=ph, fr=fr, hi=0.3, rad=(1.1, 1.8), alpha=(0.55, 0.85))
        if inf and rng.random() < 0.5:
            tip(C, inf["end"][0], inf["end"][1], 2, 1.2, rad=(1.4, 2.0), halo=0.0, sw=1.0, amp=amp, ph=ph, fr=fr, strength=0.8)
    C.done(0.3, cluster=False)


def gen_whip(C, cx, cy, H):
    """a sea whip: one long thin stem studded with little glowing polyps"""
    rng = C.rng
    P = _rib((cx, cy), rng.normal(0, 0.12), H, rng.uniform(-0.15, 0.15))
    ph, fr, amp = rng.uniform(0, 6.28), rng.uniform(0.5, 0.8), 11.0
    limb(C, P, 3.6, 2.2, 1.0, amp=amp, sw0=0.0, sw1=1.0, ph=ph, fr=fr, hi=0.3)
    for t in np.arange(0.25, 1.0, 0.09):
        q = P[int(t * (len(P) - 1))]
        tuft(C, q[0], q[1], 5, 3.5, amp, t, ph, fr)
    tip(C, P[-1][0], P[-1][1], 3, 1.6, sw=1.0, amp=amp, ph=ph, fr=fr)
    C.done(0.5, cluster=False)


def gen_rock(C, cx, cy, w, h):
    """a dark indigo rock mound that anchors a cluster, with a cyan rim light on its crest"""
    rng, ds = C.rng, C.ds
    gx, gy = jitter_grid(3.8 * ds, rng, (cx - w / 2 - 6, cy - h - 6, cx + w / 2 + 6, cy + 8))
    u = (gx - cx) / (w / 2)
    prof = h * np.clip(1 - u ** 2, 0, 1) ** 0.65 * (1 + 0.12 * np.sin(u * 9 + C.phase))
    k = (gy > cy - prof) & (gy < cy + 6) & (np.abs(u) < 1.02)
    gx, gy, u, prof = gx[k], gy[k], u[k], prof[k]
    f = np.clip((cy - gy) / max(h, 1), 0, 1)
    rgb = _grad(C.pal, 0.15 + 0.35 * f + 0.2 * rng.random(len(gx)))
    crest = gy < cy - prof + 3.5
    rgb[crest] = np.array(C.pal["hi"]) * rng.uniform(0.55, 0.9, (crest.sum(), 1))
    rgb = pointillize(rgb, 0.4, rng, 0.5, **CPAL_MIX)
    C.body(_crow(gx, gy, 3.8 * ds * 0.62 * rng.uniform(0.85, 1.2, len(gx)), 0.95, rgb, 0.0, 0.0, 0.0, 0.0, 0.0,
                 C.pulse, C.layer, CK_BODY))


class CoralGarden:
    """lays out the whole garden. Left/right edges get big layered clusters, the middle stays an open
    sandy corridor, and distant small corals fade into the blue far away."""

    def __init__(self, seed=SEED):
        self.rng = np.random.default_rng(seed + 303)
        self.bb, self.bg, self.fb, self.fg = [], [], [], []
        self.sources, self.clusters, self.formations = [], [], []
        self._floor()
        self._layout()
        self._motes()
        self._rays()
        stack = lambda L: self._cull(np.ascontiguousarray(np.vstack(L) if L else np.zeros((0, 16)), np.float32))
        self.back_body, self.back_glow, self.front_body, self.front_glow = (stack(self.bb), stack(self.bg),
                                                                           stack(self.fb), stack(self.fg))
        self.beams = CORAL_BEAMS.copy()
        self.crest = self._crest()
        src = sorted(self.sources, key=lambda s: -s[3])[:NK]
        self.kA = np.zeros((NK, 4), np.float32)
        self.kC = np.zeros((NK, 4), np.float32)
        for i, (x, y, R, st, col, pulse) in enumerate(src):
            self.kA[i] = (x, y, R, st)
            self.kC[i] = (*col, pulse)

    def C(self, front, layer, pal):
        return _Ctx(self, front, layer, pal)

    def _crest(self):
        """height of the near corals' skyline in every 8 px column (1e9 = open water). Solid = >= 5 dots in a cell.
        The jellyfish use it to float above the near corals instead of disappearing behind them."""
        f = self.front_body
        cx, cy = (f[:, 0] // 8).astype(int), (f[:, 1] // 8).astype(int)
        key = cx * 4096 + cy
        u, cnt = np.unique(key, return_counts=True)
        u = u[cnt >= 5]
        cols = u // 4096
        rows = u % 4096
        crest = np.full(W // 8 + 4, 1e9)
        for c, r in zip(cols, rows):
            if 0 <= c < len(crest):
                crest[c] = min(crest[c], r * 8.0)
        out = crest.copy()
        for d in (1, 2):                                   # widen by 2 columns so jellyfish clear the very tips
            out[d:] = np.minimum(out[d:], crest[:-d])
            out[:-d] = np.minimum(out[:-d], crest[d:])
        return out

    @staticmethod
    def _cull(a):
        """Drop dots that provably can never reach the glass (the fragment shader would clip every pixel of
        them anyway). The margin covers the dot radius, its maximum sway (<= 1.8 x amplitude), camera shake and
        edge refraction, so the picture is pixel-identical - it just stops submitting dead instances.
        Motes (they rise ~110 px) get extra room; light-shaft dots are positioned by the shader, so they stay."""
        if len(a) == 0:
            return a
        sd = sdr(a[:, 0].astype(float), a[:, 1].astype(float), WIN)
        margin = a[:, 2] + 1.8 * a[:, 9] + 12.0 + np.where(a[:, 14] == CK_MOTE, 125.0, 0.0)
        keep = (sd <= margin) | (a[:, 14] == CK_RAY)
        return np.ascontiguousarray(a[keep])

    # --------------------------------------------------------------- ground + path
    def _floor(self):
        rng = self.rng
        gx, gy = jitter_grid(4.4, rng, (UX0, 398, UX1, 530))
        hw = path_hw(gy)
        dx = np.abs(gx - PATH_CX)
        edge = np.clip((dx - hw) / 22.0, 0, 1)                           # 0 on the path .. 1 beside it
        ty = np.clip((gy - 398) / 132.0, 0, 1)[:, None]
        sand = np.array([0.22, 0.24, 0.50]) * (1 - ty) + np.array([0.52, 0.48, 0.78]) * ty
        dark = np.array([0.06, 0.05, 0.17]) * (1 - ty) + np.array([0.12, 0.08, 0.26]) * ty
        rgb = sand * (1 - edge)[:, None] + dark * edge[:, None]
        alpha = smoothstep(398, 440, gy) * 0.92
        k = (alpha > 0.04) & (sdr(gx, gy, WIN) < 26)
        gx, gy, rgb, alpha = gx[k], gy[k], rgb[k], alpha[k]
        rgb = pointillize(rgb, 0.45, rng, 0.6, **CPAL_MIX)
        self.bb.append(_crow(gx, gy, 4.4 * 0.7 * rng.uniform(0.85, 1.15, len(gx)), alpha, rgb, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.55, CK_FLOOR))
        n = 170                                                           # pebbles + glints along the path
        py = rng.uniform(430, 520, n)
        px = PATH_CX + rng.uniform(-1.05, 1.05, n) * path_hw(py)
        self.bb.append(_crow(px, py, rng.uniform(1.6, 3.0, n), rng.uniform(0.5, 0.8, n),
                             np.array([0.62, 0.60, 0.90]) * rng.uniform(0.6, 1.0, (n, 1)), 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.5, CK_FLOOR))
        m = 60
        py = rng.uniform(420, 505, m)
        px = PATH_CX + rng.uniform(-1.0, 1.0, m) * path_hw(py)
        self.bg.append(_crow(px, py, rng.uniform(1.2, 2.0, m), rng.uniform(0.4, 0.75, m), (0.55, 0.92, 1.0), 1.0,
                             0.0, 0.0, 0.0, 0.0, rng.uniform(0, 6.28, m), 0.5, CK_TIP))

    # --------------------------------------------------------------- the planting plan
    def _layout(self):
        F, M, N = LY_FAR, LY_MID, LY_NEAR
        # Depth layers are separated in x as well as in size: the NEAR clusters own the outer ~220 px on each side
        # (big, saturated, cropped by the frame), the MIDDLE corals fill the gap between them and the corridor, and
        # small FAR corals flank the far end of the path. The corridor |x-640| < 0.85*path_hw(y) stays open.
        # (A first layout buried most middle/far corals behind the near ones; this one was checked by measuring
        #  how much of each formation stays visible - see the composition checks.)
        # ---- far: small, hazy, flanking the end of the path
        for x in (552, 730):
            gen_rock(self.C(False, F, "rock"), x, 433, 52, 12)
        gen_fan(self.C(False, F, "fan_lav"), 542, 431, 50, 46)
        gen_branching(self.C(False, F, "branch_purple"), 566, 434, 38)
        gen_fan(self.C(False, F, "fan_violet"), 732, 432, 50, 46)
        gen_fan(self.C(False, F, "fan_pink"), 712, 434, 38, 36)
        gen_plants(self.C(False, F, "plant"), 528, 438, 5, 30, 8)
        gen_plants(self.C(False, F, "plant"), 754, 438, 5, 30, 8)
        gen_fan(self.C(False, F, "fan_pink"), 584, 414, 28, 28)
        gen_mushroom(self.C(False, F, "mush_teal"), 698, 414, 15)
        gen_plants(self.C(False, F, "plant"), 596, 416, 4, 24, 6)
        # ---- middle distance: the bands between the near clusters and the corridor
        for (x, w, h) in ((487, 112, 24), (808, 120, 26)):
            gen_rock(self.C(False, M, "rock"), x, 472, w, h)
        gen_fan(self.C(False, M, "fan_lav"), 490, 464, 106, 110, 0.05)
        gen_soft_tree(self.C(False, M, "soft_peach"), 449, 478, 76)
        gen_mushroom(self.C(False, M, "mush_lav"), 474, 485, 26)
        gen_mushroom(self.C(False, M, "mush_teal"), 505, 487, 22)
        gen_bubble(self.C(False, M, "pearl"), 492, 491, 5, 6)
        gen_anemone(self.C(False, M, "anem_violet"), 452, 494, 20)
        gen_branching(self.C(False, M, "branch_purple"), 770, 464, 86)
        gen_fan(self.C(False, M, "fan_pink"), 812, 467, 96, 100, -0.05)
        gen_mushroom(self.C(False, M, "mush_teal"), 762, 487, 24)
        gen_mushroom(self.C(False, M, "mush_lav"), 800, 489, 22)
        gen_anemone(self.C(False, M, "anem_cyan"), 846, 494, 20)
        gen_bubble(self.C(False, M, "pearl"), 776, 492, 5, 6)
        # ---- near: big, saturated, cropped by the window frame, framing the left and right
        for (x, w, h) in ((292, 250, 58), (1010, 250, 56)):
            gen_rock(self.C(True, N, "rock"), x, 522, w, h)
        gen_fan(self.C(True, N, "fan_violet"), 292, 529, 300, 310, 0.06)
        gen_soft_tree(self.C(True, N, "soft_pink"), 232, 522, 150)
        gen_anemone(self.C(True, N, "anem_violet"), 322, 514, 38)
        gen_anemone(self.C(True, N, "anem_pink"), 386, 512, 44)
        gen_anemone(self.C(True, N, "anem_cyan"), 436, 514, 26)
        gen_mushroom(self.C(True, N, "mush_teal"), 270, 503, 30)
        gen_bubble(self.C(True, N, "pearl"), 352, 502, 6, 9)
        gen_plants(self.C(True, N, "plant"), 252, 514, 9, 120, 24)
        gen_whip(self.C(True, N, "soft_peach"), 270, 518, 160)
        gen_fan(self.C(True, N, "fan_lav"), 1030, 531, 290, 310, -0.06)
        gen_branching(self.C(True, N, "branch_purple"), 874, 514, 140)
        gen_anemone(self.C(True, N, "anem_cyan"), 1052, 503, 44)
        gen_anemone(self.C(True, N, "anem_pink"), 962, 512, 34)
        gen_anemone(self.C(True, N, "anem_violet"), 906, 514, 26)
        gen_soft_tree(self.C(True, N, "soft_peach"), 1092, 524, 180)
        gen_mushroom(self.C(True, N, "mush_lav"), 1002, 504, 34)
        gen_bubble(self.C(True, N, "pearl"), 932, 506, 6, 9)
        gen_plants(self.C(True, N, "plant"), 1064, 514, 9, 110, 24)

    # --------------------------------------------------------------- glowing motes around the clusters
    def _motes(self):
        rng = self.rng
        pal = np.array([(1.0, 0.82, 0.50), (1.0, 0.60, 0.80), (0.55, 0.92, 1.0), (0.80, 0.70, 1.0)])
        for (cx, cy, layer) in self.clusters:
            n = int(26 * (1.0 - 0.5 * layer))
            x = cx + rng.normal(0, 70 * (1.0 - 0.4 * layer), n)
            y = cy + rng.normal(0, 55, n)
            rows = _crow(x, y, rng.uniform(0.9, 1.9, n) * (1.0 - 0.3 * layer), rng.uniform(0.35, 0.80, n),
                         pal[rng.integers(0, 4, n)], 1.0, 0.0, 0.0, rng.uniform(0, 6.28, n), rng.uniform(4, 10, n),
                         0.0, layer, CK_MOTE)
            (self.fg if layer < 0.3 else self.bg).append(rows)

    # --------------------------------------------------------------- shafts of light that land on the corals
    def _rays(self):
        rng = self.rng
        for bi, b in enumerate(CORAL_BEAMS):
            n = 520
            lat = np.clip(rng.normal(0, 0.45, n), -1, 1)
            y = rng.uniform(54, 480, n)
            gold = rng.random(n) < 0.12
            col = np.where(gold[:, None], (1.0, 0.92, 0.62), (0.55, 0.92, 1.0))
            self.bg.append(_crow(lat, y, rng.uniform(1.2, 2.4, n), rng.uniform(0.10, 0.28, n), col, 0.0, 0.0, 0.0,
                                 rng.uniform(0, 6.28, n), 0.0, 0.0, 0.5, CK_RAY, bi))


# ----------------------------------------------------------------------------
# 6f. THE FISH (PHASE 5)
#     Same split as the jellyfish: fish_dots() builds each fish once from dots (body, tail,
#     fins, eye, a glowing lateral line - positions are in the fish's own space); FishSystem
#     steers them (schools follow a leader that wanders between the corals) and the vertex
#     shader bends the body / beats the tail from a per-fish tail phase.
# ----------------------------------------------------------------------------
NF = 18                                       # fish slots (must equal NF in the GLSL)
(FK_BODY, FK_TAIL, FK_DORSAL, FK_PELVIC, FK_EYE, FK_GLOW) = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
# (count, half-length px, cruising speed px/s, body colour, belly colour, layer)
FISH_SPECS = [
    (8, 6.5, 34.0, (0.25, 0.80, 1.00), (0.75, 0.96, 1.00), 0.55),    # little cyan school
    (2, 9.0, 24.0, (1.00, 0.70, 0.34), (1.00, 0.90, 0.62), 0.40),    # golden pair
    (3, 8.0, 26.0, (0.60, 0.38, 1.00), (0.90, 0.78, 1.00), 0.50),    # violet trio
    (2, 7.0, 30.0, (1.00, 0.42, 0.74), (1.00, 0.84, 0.92), 0.55),    # pink pair
    (1, 15.0, 15.0, (0.45, 0.72, 1.00), (0.92, 0.98, 1.00), 0.25),   # three unhurried loners
    (1, 14.0, 14.0, (0.70, 0.50, 1.00), (0.95, 0.88, 1.00), 0.30),
    (1, 13.0, 16.0, (0.20, 0.85, 0.80), (0.80, 1.00, 0.95), 0.28),
]


def fish_dots(rng, fid, size):
    """one fish from dots; (u, v) are coordinates along / across each part"""
    rows = []

    def add(kind, u, v, rad, alpha, cm):
        u = np.atleast_1d(np.asarray(u, np.float32))
        n = len(u)
        B = lambda a: np.broadcast_to(np.asarray(a, np.float32), (n,))
        rows.append(np.column_stack([B(kind), B(fid), u, B(v), B(rad), B(alpha), B(cm), rng.uniform(0, 6.28, n)]))
    r0 = 0.65 + size * 0.045
    nb = int(14 + size * 3.2)
    add(FK_BODY, rng.uniform(-1, 1, nb), rng.uniform(-1, 1, nb), rng.uniform(r0, r0 * 1.5, nb), rng.uniform(0.75, 0.95, nb), 0)
    ne = int(8 + size * 1.2)                                   # outline dots: crisper silhouette
    ue = np.linspace(-0.95, 0.95, ne)
    add(FK_BODY, np.r_[ue, ue], np.r_[np.full(ne, -1.0), np.full(ne, 1.0)], r0 * 1.15, 0.9, 0)
    nt = int(6 + size)
    add(FK_TAIL, rng.uniform(0, 1, nt), rng.uniform(-1, 1, nt), rng.uniform(r0 * 0.8, r0 * 1.3, nt), rng.uniform(0.6, 0.85, nt), 1)
    nd = int(4 + size * 0.5)
    add(FK_DORSAL, rng.uniform(-0.35, 0.45, nd), rng.uniform(0, 1, nd), r0 * 0.9, 0.7, 1)
    add(FK_PELVIC, rng.uniform(0, 1, 3), rng.uniform(0, 1, 3), r0 * 0.9, 0.7, 1)
    add(FK_EYE, 0.0, 0.0, r0 * 1.3, 1.0, 0)
    ng = int(5 + size * 0.5)
    add(FK_GLOW, np.linspace(-0.6, 0.6, ng), 0.0, r0 * 0.9, 0.55, 0)
    return np.vstack(rows).astype(np.float32)


class FishSystem:
    def __init__(self, seed=SEED):
        self.seed = seed
        rng = np.random.default_rng(seed + 404)
        self.specs = []
        fid = 0
        parts = []
        for si, (cnt, size, spd, c1, c2, layer) in enumerate(FISH_SPECS):
            for k in range(cnt):
                self.specs.append((si, k, size * rng.uniform(0.92, 1.08), spd, c1, c2, layer))
                parts.append(fish_dots(rng, fid, self.specs[-1][2]))
                fid += 1
        self.dots = np.ascontiguousarray(np.vstack(parts), np.float32)
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.seed + 9)
        self.t = 0.0
        r = self.rng
        self.groups = {}
        self.fish = []
        for i, (si, k, size, spd, c1, c2, layer) in enumerate(self.specs):
            if k == 0:
                g = dict(wp=np.array([r.uniform(300, 980), r.uniform(260, 470)]), pos=np.array([r.uniform(260, 1000), r.uniform(270, 460)]),
                         vel=np.array([r.choice([-1, 1]) * spd * 0.8, 0.0]), spd=spd * r.uniform(0.85, 1.1))
                self.groups[si] = g
            g = self.groups[si]
            off = r.normal(0, 22, 2) if k else np.zeros(2)
            self.fish.append(dict(si=si, k=k, size=size, pos=g["pos"] + off, vel=g["vel"].copy(), off=off, tail=r.uniform(0, 6.28),
                                  dir=1.0 if g["vel"][0] >= 0 else -1.0, slope=0.0, spd=spd * r.uniform(0.9, 1.15), c1=c1, c2=c2,
                                  layer=layer, wob=r.uniform(0, 6.28)))

    def update(self, dt):
        if dt <= 0:
            return
        self.t += dt
        r = self.rng
        for g in self.groups.values():                       # leaders wander between waypoints among the corals
            d = g["wp"] - g["pos"]
            dist = float(np.hypot(*d))
            if dist < 35:
                far = r.random() < 0.12                       # now and then swim out of view and come back
                g["wp"] = np.array([r.choice([140.0, 1140.0]) if far else r.uniform(260, 1020), r.uniform(250, 470)])
            want = d / (dist + 1e-6) * g["spd"]
            g["vel"] += (want - g["vel"]) * (1 - math.exp(-dt * 1.2))
            g["pos"] = g["pos"] + g["vel"] * dt
        for f in self.fish:
            g = self.groups[f["si"]]
            if f["k"] == 0:
                f["pos"], f["vel"] = g["pos"], g["vel"]
            else:                                              # followers hold a loose, wobbling formation
                off = f["off"] + 9.0 * np.array([math.sin(self.t * 0.7 + f["wob"]), math.cos(self.t * 0.9 + f["wob"] * 1.3)])
                want = (g["pos"] + off - f["pos"]) * 1.1
                sp = float(np.hypot(*want))
                lim = f["spd"] * 1.3
                if sp > lim:
                    want *= lim / sp
                f["vel"] = f["vel"] + (want - f["vel"]) * (1 - math.exp(-dt * 2.5))
                f["pos"] = f["pos"] + f["vel"] * dt
            vx, vy = f["vel"]
            if abs(vx) > 4:
                f["dir"] = 1.0 if vx > 0 else -1.0
            f["slope"] += (max(-0.5, min(0.5, vy / (abs(vx) + 10.0))) - f["slope"]) * (1 - math.exp(-dt * 3.0))
            f["tail"] += dt * 2 * math.pi * (1.4 + 0.045 * float(np.hypot(vx, vy)))

    def simulate(self, t_end, step=1 / 30.0):
        self.reset()
        while self.t < t_end:
            self.update(min(step, t_end - self.t))

    def uniforms(self):
        A, B, C, D = (np.zeros((NF, 4), np.float32) for _ in range(4))
        for i, f in enumerate(self.fish):
            A[i] = (f["pos"][0], f["pos"][1], f["size"], f["dir"])
            B[i] = (f["tail"], float(np.hypot(*f["vel"])), f["layer"], 1.0)
            C[i] = (*f["c1"], f["slope"])
            D[i] = (*f["c2"], 0.0)
        return dict(u_fA=A, u_fB=B, u_fC=C, u_fD=D)


# ----------------------------------------------------------------------------
# 7. GLASS REFLECTIONS (additive, extremely faint)
# ----------------------------------------------------------------------------
def build_reflect(rng):
    em = Emitter(rng)
    cx, cy, hx, hy, _ = WIN
    bbox = (cx - hx, cy - hy, cx + hx, cy + hy)
    # kind 0: diagonal sheen that slowly sweeps across
    gx, gy = jitter_grid(14, rng, bbox)
    k = sdr(gx, gy, WIN) < -8
    gx, gy = gx[k], gy[k]
    em.add(gx, gy, (0.50, 0.62, 0.85), rng.uniform(2.0, 4.8, len(gx)), 0.0, 0.0, 0.11)
    # (the two amber lamp "ghost" reflections that used to live here were removed on request.
    #  Their random draws are still consumed into a scratch emitter that is never drawn, so every
    #  layer generated afterwards - fireworks, underwater world, spill light - stays exactly as before.)
    scratch = Emitter(rng)
    for i, (px, py) in enumerate(((335.0, 215.0), (955.0, 228.0))):
        x, y, r = radial_dots(rng, 650, px, py, 0, 95, 26)
        scratch.add(x, y, (0, 0, 0), rng.uniform(1.2, 3.0, len(x)), 1.0, float(i), 0.0)
        x, y, r = radial_dots(rng, 90, px, py, 0, 12, 5)
        scratch.add(x, y, (0, 0, 0), rng.uniform(1.0, 2.0, len(x)), 1.0, float(i), 0.0)
    # kind 2: faint ghosts of the seat backs behind the viewer
    for mirror in (False, True):
        sx = W - 330.0 if mirror else 330.0
        rect = (sx, 505.0, 95.0, 150.0, 55.0)
        gx, gy = jitter_grid(7, rng, (sx - 100, 355, sx + 100, 660))
        k = (sdr(gx, gy, rect) < 0) & (sdr(gx, gy, WIN) < -6)
        em.add(gx[k], gy[k], (0.22, 0.42, 0.34), rng.uniform(1.8, 3.0, k.sum()), 2.0, 0.0, 0.085)
    arr = em.array()
    return arr


# ----------------------------------------------------------------------------
# 8. SCENE ASSEMBLY
# ----------------------------------------------------------------------------
def build_scene(seed=SEED):
    rng = np.random.default_rng(seed)
    scene = {"outside": build_outside(rng), "interior": build_interior(rng), "reflect": build_reflect(rng),
             "fireworks": build_fireworks(rng), "underwater": build_underwater(rng),
             "spill": build_spill(rng)}
    jelly = JellySystem(seed)
    scene["jelly"] = jelly
    scene["jelly_dots"] = jelly.dots
    scene["coral"] = CoralGarden(seed)        # Phase 5: built on its own random stream
    jelly.set_crest(scene["coral"].crest)       # ...and the jellyfish know where the near corals end
    scene["fish"] = FishSystem(seed)
    return scene


# ----------------------------------------------------------------------------
# 9. TRAIN MOTION + FLICKER (all tiny on purpose)
# ----------------------------------------------------------------------------
def train_shake(t):
    sway_x = 1.4 * math.sin(t * 0.9) + 0.6 * math.sin(t * 0.37 + 1.3)
    sway_y = 0.6 * math.sin(t * 1.7 + 0.4)
    rattle = 0.45 * math.sin(t * 23.0) + 0.35 * math.sin(t * 31.7 + 1.0)
    tt = t % 1.35                                         # rail-joint "clack"
    clack = 1.5 * math.exp(-tt * 7.0) * math.sin(tt * 38.0)
    return sway_x + 0.25 * rattle, sway_y + 0.4 * rattle + clack


def lamp_flicker(t):
    out = []
    for seed in (0.0, 2.3):
        v = (0.5 * math.sin(t * 7.3 + seed) + 0.3 * math.sin(t * 13.1 + seed * 2.1)
             + 0.2 * math.sin(t * 3.1 + seed * 0.7))
        v = 0.07 * v + 0.05 * math.sin(t * 0.9 + seed)
        if math.sin(t * 0.37 + seed * 5.0) > 0.985:        # rare tiny dip
            v -= 0.10
        out.append(v)
    return tuple(out)


# ----------------------------------------------------------------------------
# 10. GPU RENDERER
# ----------------------------------------------------------------------------
JELLY_LIGHT_GLSL = '''
#define NJ 17
uniform float u_garden;
uniform vec4 u_jA[NJ];   // x, y, scale, layer
uniform vec4 u_jG[NJ];   // glow colour, glow strength
#define NK 14
uniform float u_coral;
uniform vec4 u_kA[NK];   // coral glow sources: x, y, radius, strength
uniform vec4 u_kC[NK];   // colour, pulse phase
vec3 jellyGlow(vec2 p, out float wsum) {          // nearby water brightens and takes their colour
    vec3 acc = vec3(0.0);
    wsum = 0.0;
    for (int i = 0; i < NJ; i++) {
        float g = u_jG[i].w * u_garden;
        if (g < 0.01) continue;
        vec2 d = p - u_jA[i].xy;
        float R = 95.0 * u_jA[i].z;
        float w = exp(-dot(d, d) / (R * R)) * g;
        acc += u_jG[i].rgb * w;
        wsum += w;
    }
    for (int k = 0; k < NK; k++) {                    // glowing corals pulse slowly; passing particles pick it up
        float g = u_kA[k].w * u_coral;
        if (g < 0.01) continue;
        vec2 d = p - u_kA[k].xy;
        float R = u_kA[k].z;
        float w = exp(-dot(d, d) / (R * R)) * g * (0.55 + 0.45 * sin(u_time * 0.42 + u_kC[k].w));
        acc += u_kC[k].rgb * w;
        wsum += w;
    }
    return acc;
}
float winSdf(vec2 p) {
    vec2 q = abs(p - vec2(640.0, 285.0)) - (vec2(420.0, 205.0) - 78.0);
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - 78.0;
}
const float SPILL_GAIN = 0.07;   // interior colours are locked: the sea only whispers on the room
vec3 jellySpill(vec2 p, out float wsum) {         // light from animals inside the glass reaching the room
    vec3 acc = vec3(0.0);
    wsum = 0.0;
    for (int i = 0; i < NJ; i++) {
        float g = u_jG[i].w * u_garden;
        if (g < 0.01) continue;
        vec4 A = u_jA[i];
        float inside = 1.0 - smoothstep(-30.0, 60.0, winSdf(A.xy));
        float d = max(length(p - A.xy) - 55.0 * A.z, 0.0);
        float w = exp(-d / 115.0) * g * inside;
        acc += u_jG[i].rgb * w;
        wsum += w;
    }
    return acc;
}
'''

CLIP_GLSL = """
float winMask(vec2 p) {                 // 1 inside the panoramic glass, 0 outside
    vec2 q = abs(p - vec2(640.0, 285.0)) - (vec2(420.0, 205.0) - 78.0);
    float sd = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - 78.0;
    return 1.0 - smoothstep(-1.6, -0.4, sd);    // reaches 0 half a pixel inside the glass: no exterior colour on wood
}
"""

WATER_GLSL = """
const float SHORE = 335.0;
const float UX0 = 194.0, UY0 = 54.0, UXR = 892.0, UYR = 462.0, UY1 = 516.0;
float waveY(float x, float t) {         // the moving waterline (shared by every shader)
    return 4.0 * sin(x * 0.018 + t * 1.1) + 2.2 * sin(x * 0.047 - t * 1.7) + 1.2 * sin(x * 0.11 + t * 2.3);
}
bool isK(float k, float v) { return abs(k - v) < 0.5; }
"""

VERT = """
#version 330
in vec2 in_quad;
in vec2 in_pos;
in float in_radius;
in vec4 in_col;
in vec3 in_fx;
uniform vec2 u_res;
uniform float u_time;
uniform int u_mode;       // 0 interior, 1 outside world, 2 glass reflections, 5 blue spill light
uniform vec2 u_shake;
uniform vec2 u_lamps;
uniform float u_wl;       // waterline height (pixels), driven by depth
uniform float u_depth;    // 0 surface .. 1 deepest
uniform float u_spill;    // strength of blue light on the interior
""" + JELLY_LIGHT_GLSL + WATER_GLSL + """
out vec2 v_local;
out float v_rad;
out vec4 v_col;
out vec2 v_world;
out float v_stretch;
out float v_soft;

vec2 refractEdge(vec2 p) {              // gentle refraction that grows near the glass edge
    vec2 c = vec2(640.0, 285.0);
    vec2 q = abs(p - c) - (vec2(420.0, 205.0) - 78.0);
    float sd = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - 78.0;
    float e = smoothstep(-70.0, 0.0, sd);
    return normalize(c - p + 0.001) * e * 3.5 * sin(u_time * 0.9 + p.x * 0.03 + p.y * 0.045);
}

void main() {
    vec2 p = in_pos;
    vec4 col = in_col;
    float ph = in_fx.y;
    float stretch = 1.0;
    float soft = 0.0;
    float t = u_time;
    if (u_mode == 0) {
        // interior is locked in place: only a sub-pixel shimmer + lamp flicker
        p += 0.35 * vec2(sin(t * 1.7 + ph), cos(t * 1.3 + ph * 1.7));
        float f = mix(u_lamps.x, u_lamps.y, in_fx.z);
        col.rgb *= 1.0 + in_fx.x * f;
        col.rgb *= 1.0 + 0.035 * sin(t * 2.3 + ph * 3.0) * step(0.15, in_fx.x);
    } else if (u_mode == 1) {
        float kind = in_fx.x;
        if (kind < 3.5) {
            // ===== the lakeside world: sways as the train nears the water, then sinks behind it
            float approach = smoothstep(0.0, 0.2, u_depth) * (1.0 - smoothstep(0.35, 0.5, u_depth));
            p += u_shake * in_fx.z * (1.0 + 1.5 * approach);
            p.x += 4.0 * approach * sin(t * 0.45);
            p.y += 2.5 * approach * sin(t * 0.7 + p.x * 0.006);
            if (kind > 0.5 && kind < 1.5) {               // star twinkle
                col.a *= 0.55 + 0.45 * sin(t * (1.2 + fract(ph) * 1.8) + ph * 9.0);
            } else if (kind > 1.5) {                      // lake: drift + shimmer
                float dw = clamp((p.y - SHORE) / 160.0, 0.0, 1.0);
                p.x += (1.0 + 2.0 * approach) * ((0.8 + 2.4 * dw) * sin(t * 1.1 + p.y * 0.09 + ph)
                     + 0.6 * sin(t * 2.3 + p.y * 0.21 + ph * 3.0));
                col.rgb *= 0.84 + 0.26 * sin(t * 1.9 + ph * 5.0 + p.y * 0.25);
                if (kind < 2.5) stretch = 1.9;
            }
            // --- water covering the scene: wobble, scatter, cool colours, fade
            float sub = clamp((p.y - (u_wl + waveY(p.x, t))) / 230.0, 0.0, 1.0);
            if (sub > 0.0) {
                p.x += sub * (9.0 * sin(t * 1.3 + p.y * 0.07 + ph) + 5.0 * sin(t * 2.9 + p.y * 0.19 + ph * 2.0));
                p.y += sub * 3.0 * sin(t * 1.7 + p.x * 0.05);
                p += sub * sub * 14.0 * vec2(sin(ph * 7.0), cos(ph * 5.0 + t * 0.3));
                col.rgb = mix(col.rgb, col.rgb * vec3(0.5, 0.9, 1.2) + vec3(0.0, 0.05, 0.10), sub * 0.8);
                col.a *= 1.0 - smoothstep(0.30, 1.0, sub);
            }
        } else {
            // ===== the underwater world
            float layer = in_fx.z;
            if (isK(kind, 4.0)) {                         // water body: currents
                p.x += (1.0 + 1.4 * layer) * sin(t * 0.45 + p.y * 0.028 + ph) + 0.9 * sin(t * 0.9 + p.y * 0.07 + ph * 2.0);
                p.y += 0.8 * sin(t * 0.6 + p.x * 0.03 + ph);
                col.rgb *= 0.88 + 0.22 * sin(p.y * 0.045 - t * 0.35 + p.x * 0.008 + ph * 0.5);
            } else if (isK(kind, 5.0)) {                  // marine snow
                p.y = UY0 + mod(in_pos.y - UY0 + t * (4.0 + 10.0 * layer), UYR);
                p.x = UX0 + mod(in_pos.x - UX0 - t * (1.5 + 3.5 * layer) + (5.0 + 9.0 * layer) * sin(t * 0.2 + ph), UXR);
                col.a *= 0.7 + 0.3 * sin(t * 1.3 + ph * 11.0);
            } else if (isK(kind, 6.0) || isK(kind, 12.0)) {   // plankton clusters / firework fragments
                p.x = UX0 + mod(in_pos.x - UX0 - t * (2.0 + 5.0 * layer), UXR);
                p += (8.0 + 26.0 * layer) * vec2(sin(t * 0.13 + ph * 3.0), cos(t * 0.11 + ph * 5.0));
                p += 1.4 * vec2(sin(t * 0.9 + in_pos.x), cos(t * 0.8 + in_pos.y));
                col.a *= 0.8 + 0.2 * sin(t * 0.9 + ph * 7.0);
                if (isK(kind, 12.0)) col.a *= 1.0 - 0.5 * smoothstep(0.6, 1.0, u_depth);
            } else if (isK(kind, 7.0)) {                  // bubbles rise, wobbling
                p.y = UY1 - mod(UY1 - in_pos.y + t * (12.0 + 34.0 * layer), UYR);
                p.x += 3.0 * sin(t * 1.4 + ph + p.y * 0.05) + (3.0 + 6.0 * layer) * sin(t * 0.2 + ph);
            } else if (isK(kind, 8.0)) {                  // bioluminescence: slow pulses, deep only
                p.x = UX0 + mod(in_pos.x - UX0 - t * (1.0 + 2.5 * layer), UXR);
                p.y += 4.0 * sin(t * 0.12 + ph * 3.0);
                float pulse = pow(0.5 + 0.5 * sin(t * 0.33 + ph * 5.0), 3.0);
                col.a *= (0.15 + 0.85 * pulse) * smoothstep(0.50, 0.85, u_depth);
            } else if (isK(kind, 9.0)) {                  // far sea floor + torii: scroll slowly
                p.x = UX0 + mod(in_pos.x - UX0 - t * (1.2 + 2.0 * layer), UXR);
                col.a *= smoothstep(0.52, 0.82, u_depth);
            } else if (isK(kind, 13.0)) {                 // the whale: slow swim, tail sway
                p.x = UX0 - 450.0 + mod(in_pos.x - UX0 + 450.0 + t * 5.5, 1900.0);
                p.y += 6.0 * sin(t * 0.2 + ph) + 7.0 * sin(t * 0.5 - layer * 3.0) * layer * layer;
                col.a *= smoothstep(0.52, 0.82, u_depth);
            } else if (isK(kind, 10.0)) {                 // the waterline: foam and cyan rows
                p.x += 5.0 * sin(t * 0.7 + ph);
                p.y = u_wl + waveY(p.x, t) + in_pos.y + 1.2 * sin(t * 2.0 + ph * 3.0);
                col.a *= 0.65 + 0.35 * sin(t * 2.7 + ph * 4.0);
            } else {                                      // 11: light shafts, fade as we sink
                p.x += 10.0 * sin(t * 0.18 + ph);
                float breath = 0.55 + 0.45 * sin(t * 0.5 + ph * 2.0);
                col.a *= 2.2 * breath * exp(-max(285.0 - u_wl, 0.0) / 700.0) * (1.0 - 0.85 * smoothstep(0.65, 1.0, u_depth)) * (1.0 - layer);
            }
            p += refractEdge(p) + u_shake * 0.35;
            float subu = p.y - (u_wl + waveY(p.x, t));
            float lightF = exp(-max(subu, 0.0) / 650.0) * mix(1.0, 0.5, smoothstep(0.5, 1.0, u_depth));
            float uw = isK(kind, 10.0) ? 1.0 : smoothstep(0.0, 45.0, subu);
            if (isK(kind, 4.0) || isK(kind, 5.0) || isK(kind, 6.0) || isK(kind, 7.0))
                col.rgb = mix(vec3(0.006, 0.02, 0.085), col.rgb, 0.30 + 0.70 * lightF);
            if (isK(kind, 4.0)) {                         // turquoise light just under the surface
                float near = exp(-max(subu, 0.0) / 110.0);
                col.rgb = mix(col.rgb, vec3(0.14, 0.70, 0.74), near * 0.26) * (1.0 + 0.25 * near);
            }
            if (isK(kind, 5.0) || isK(kind, 7.0)) col.a *= 1.0 - 0.35 * u_garden;
            if (isK(kind, 7.0)) { col.a *= 0.55 + 0.45 * lightF; col.rgb += vec3(0.05, 0.10, 0.12); }
            if (isK(kind, 4.0) || isK(kind, 5.0) || isK(kind, 6.0) || isK(kind, 7.0) || isK(kind, 12.0)) {
                float jw;
                vec3 jl = jellyGlow(p, jw);
                if (isK(kind, 4.0)) {
                    col.rgb += jl * 0.10;                               // the water itself glows softly
                } else {                                                // drifting particles light up as they pass
                    col.rgb = col.rgb * (1.0 + 0.9 * jw) + jl * 0.30;
                    col.a = min(col.a * (1.0 + 1.0 * jw), 1.0);
                }
            }
            col.a *= uw;
            if (isK(kind, 4.0)) col.a *= 1.0 - 0.45 * exp(-max(subu, 0.0) / 130.0);
            if ((isK(kind, 6.0) || isK(kind, 8.0) || isK(kind, 12.0)) && in_radius > 6.0) soft = 1.0;
            if (isK(kind, 7.0)) soft = 2.0;
        }
    } else if (u_mode == 5) {
        // blue light from the sea dappling the frame, wall and seats
        float caustic = 0.5 + 0.5 * sin(p.x * 0.035 + t * 0.8 + sin(p.y * 0.05 + t * 0.5) * 1.5);
        float jw;
        vec3 jl = jellySpill(p, jw);
        float base = col.a * u_spill * (0.30 + 0.70 * caustic);
        col.rgb = mix(col.rgb, 0.7 * jl / max(jw, 0.001), clamp(jw * 1.0, 0.0, 0.45));
        col.a = (base * (1.0 + 0.8 * min(jw, 1.5)) + col.a * 0.22 * min(jw, 1.2)) * SPILL_GAIN;
    } else {
        float kind = in_fx.x;
        if (kind < 0.5) {
            float b = p.x * 0.9 + p.y * 0.45 - t * 38.0;
            float s = fract(b / 1100.0);
            col.a *= smoothstep(0.0, 0.18, s) * smoothstep(0.42, 0.24, s);
            p += u_shake * 0.15;
        } else {
            p.x += 3.0 * sin(t * 0.17 + ph * 0.2);
            p += u_shake * 0.2;
            col.a *= 0.8 + 0.2 * sin(t * 0.4 + ph);
        }
    }
    vec2 q = p / u_res - 0.5;
    float vig = 1.0 - 0.5 * pow(clamp(length(q * vec2(1.0, 1.1)) * 1.45, 0.0, 1.4), 2.3);
    col.rgb *= vig;
    vec2 ext = vec2(in_radius + 1.0, in_radius * stretch + 1.0);
    vec2 world = p + in_quad * ext;
    gl_Position = vec4(world.x / u_res.x * 2.0 - 1.0, 1.0 - world.y / u_res.y * 2.0, 0.0, 1.0);
    v_local = in_quad * ext;
    v_rad = in_radius;
    v_col = col;
    v_world = world;
    v_stretch = stretch;
    v_soft = soft;
}
"""

FRAG = """
#version 330
in vec2 v_local;
in float v_rad;
in vec4 v_col;
in vec2 v_world;
in float v_stretch;
in float v_soft;
uniform int u_clip;
out vec4 f_color;
""" + CLIP_GLSL + """
void main() {
    float d = length(vec2(v_local.x, v_local.y / v_stretch));
    float r = max(v_rad, 0.5);
    float n = d / r;
    float a = clamp(v_rad + 0.5 - d, 0.0, 1.0);          // 1px soft edge
    float shade = 1.0 + 0.16 * (0.35 - n * n) + 0.07 * dot(v_local / r, vec2(-0.5, -0.6));
    if (v_soft > 1.5) {                                   // bubble: ring + glint
        float ring = smoothstep(0.55, 0.88, n) * (1.0 - smoothstep(0.92, 1.05, n));
        float glint = smoothstep(0.45, 0.0, length(v_local / r - vec2(-0.35, -0.4)));
        a = clamp(ring * 0.85 + 0.16 * (1.0 - n) + glint * 0.9, 0.0, 1.0) * clamp(v_rad + 0.5 - d, 0.0, 1.0);
        shade = 1.0;
    } else if (v_soft > 0.5) {                            // luminous halo
        a = pow(clamp(1.0 - n, 0.0, 1.0), 1.6);
        shade = 1.0;
    }
    if (u_clip == 1) a *= winMask(v_world);              // outside world lives only inside the glass
    if (a <= 0.0) discard;
    f_color = vec4(v_col.rgb * shade, v_col.a * a);
}
"""

# ---- fireworks: positions are computed from the clock, so thousands of dots cost almost nothing
FW_VERT = """
#version 330
in vec2 in_quad;
in vec2 in_o;      // burst origin (or rocket start)
in vec2 in_v;      // launch velocity (or rocket start->apex vector)
in vec2 in_t;      // burst time, lifetime
in vec3 in_col;
in vec4 in_p;      // size, drag, gravity, trail spacing (s)
in vec4 in_k;      // kind (0 spark, 1 glow, 2 rocket, 3 glitter), trail index, trail count, seed
uniform vec2 u_res;
uniform float u_tc;      // festival clock, wraps every FW_CYCLE
uniform float u_time;
uniform vec2 u_shake;
uniform int u_refl;      // 0 = sky, 1 = reflection in the lake
uniform float u_wl;      // waterline: fireworks sinking behind it fragment and fade
""" + WATER_GLSL + """
const float RK = 0.62;
out vec2 v_local;
out float v_rad;
out vec4 v_col;
out vec2 v_world;
out float v_stretch;
out float v_soft;
void cull() {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    v_local = vec2(0.0); v_rad = 1.0; v_col = vec4(0.0); v_world = vec2(0.0); v_stretch = 1.0; v_soft = 0.0;
}
void main() {
    float kind = in_k.x, j = in_k.y, nt = max(in_k.z, 1.0), seed = in_k.w;
    float age = u_tc - in_t.x - j * in_p.w;          // trail copies lag behind the head
    float life = in_t.y;
    if (age < 0.0 || age > life) { cull(); return; }
    float u = age / life;
    vec2 pos;
    float bright;
    bool glow = (kind > 0.5 && kind < 1.5);
    if (kind > 1.5 && kind < 2.5) {                   // rocket: eases out as it climbs
        float e = 1.0 - (1.0 - u) * (1.0 - u);
        pos = in_o + in_v * e + vec2(1.6 * sin(age * 14.0 + seed * 6.0) * (1.0 - u), 0.0);
        bright = 1.0 - smoothstep(0.85, 1.0, u);
    } else if (kind > 2.5) {                          // glitter: a blink in the air
        pos = in_o;
        bright = (1.0 - u) * (0.55 + 0.45 * sin(age * 70.0 + seed * 20.0)) * 1.5;
    } else {
        float k = max(in_p.y, 0.001);
        float f = (1.0 - exp(-k * age)) / k;          // drag
        pos = in_o + in_v * f;
        pos.y += in_p.z / k * (age - f);              // gravity with terminal velocity
        float flash = 1.0 + 0.8 * exp(-age * 6.0);
        float tw = 0.7 + 0.3 * sin(age * (16.0 + seed * 12.0) + seed * 40.0);
        bright = pow(1.0 - u, 1.4) * flash * mix(1.0, tw, smoothstep(0.3, 0.9, u));
        if (glow) bright = pow(1.0 - u, 2.0) * (1.0 + 0.8 * exp(-age * 5.0));
    }
    float tf = 1.0 - j / nt;
    bright *= tf * tf;
    vec3 col = mix(in_col, in_col * vec3(1.0, 0.62, 0.35), smoothstep(0.55, 1.0, u) * 0.6);
    float rad = in_p.x * (1.0 - 0.45 * j / nt);
    float stretch = 1.0;
    pos += u_shake * 0.4;
    if (u_refl == 0) {
        if (pos.y > SHORE - 1.0) { cull(); return; } // sparks die behind the town
    } else {
        if (pos.y > SHORE) { cull(); return; }
        pos.y = SHORE + (SHORE - pos.y) * RK;         // mirror in the lake
        float dd = clamp((pos.y - SHORE) / 160.0, 0.0, 1.0);
        pos.x += (1.2 + 2.6 * dd) * sin(u_time * 1.7 + pos.y * 0.13 + seed * 6.28)
               + 1.0 * sin(u_time * 4.1 + seed * 9.0);
        bright *= 0.55 * (0.62 + 0.38 * sin(u_time * 8.0 + seed * 40.0 + pos.y * 0.4)) * (1.0 - 0.5 * dd);
        stretch = glow ? 2.2 : 2.7;                   // vertically stretched, shimmering dots
    }
    // ---- the rising water: trails fragment, particles scatter and drift, colours go soft and cyan
    float sub = clamp((pos.y - (u_wl + waveY(pos.x, u_time))) / 130.0, 0.0, 1.0);
    if (sub > 0.0) {
        float h1 = fract(sin(seed * 91.7 + j * 12.3) * 43758.5);
        float h2 = fract(sin(seed * 17.3 + j * 7.1) * 24634.6);
        pos.x += sub * ((h1 - 0.5) * 34.0 + 7.0 * sin(u_time * 1.4 + pos.y * 0.08 + seed * 6.0));
        pos.y += sub * (h2 - 0.5) * 16.0 + sub * sub * 12.0;
        float lum = dot(col, vec3(0.333));
        col = mix(col, vec3(lum * 0.55, lum, lum * 1.15), sub * 0.75);
        bright *= (1.0 - 0.92 * smoothstep(0.35, 1.0, sub)) * mix(1.0, 1.0 - j / nt, sub);
    }
    float a = clamp(bright, 0.0, 1.0);
    col *= max(bright, 1.0);
    vec2 ext = vec2(rad + 1.0, rad * stretch + 1.0);
    vec2 world = pos + in_quad * ext;
    gl_Position = vec4(world.x / u_res.x * 2.0 - 1.0, 1.0 - world.y / u_res.y * 2.0, 0.0, 1.0);
    v_local = in_quad * ext;
    v_rad = rad;
    v_col = vec4(col, a);
    v_world = world;
    v_stretch = stretch;
    v_soft = glow ? 1.0 : 0.0;
}
"""

FW_FRAG = """
#version 330
in vec2 v_local;
in float v_rad;
in vec4 v_col;
in vec2 v_world;
in float v_stretch;
in float v_soft;
out vec4 f_color;
""" + CLIP_GLSL + """
void main() {
    float d = length(vec2(v_local.x, v_local.y / v_stretch));
    float a;
    if (v_soft > 0.5) a = pow(clamp(1.0 - d / v_rad, 0.0, 1.0), 1.6);   // luminous halo blobs
    else a = clamp(v_rad + 0.5 - d, 0.0, 1.0);                          // crisp dots
    a *= winMask(v_world);
    if (a <= 0.003) discard;
    f_color = vec4(v_col.rgb, v_col.a * a);
}
"""

JELLY_VERT = '''
#version 330
#define NJ 17
in vec2 in_quad;
in vec4 in_a;      // kind, jelly id, u, v        (where this dot lives on the animal)
in vec4 in_b;      // per-kind parameters
in vec4 in_c;      // radius, alpha, colour mix, phase
uniform vec2 u_res;
uniform float u_time;
uniform vec2 u_shake;
uniform float u_garden;
uniform vec4 u_jA[NJ];   // x, y, scale, layer (0 near .. 1 far)
uniform vec4 u_jB[NJ];   // contraction, bell clock, lean, sway
uniform vec4 u_jC[NJ];   // edge colour, presence
uniform vec4 u_jD[NJ];   // centre colour, tentacle clock
uniform vec4 u_jE[NJ];   // velocity xy, seed, brightness
const float R0 = 46.0;
const float H0 = 28.0;
out vec2 v_local;
out float v_rad;
out vec4 v_col;
out vec2 v_world;
out float v_stretch;
out float v_soft;

vec2 rot(vec2 v, float a) { float c = cos(a), s = sin(a); return vec2(c * v.x - s * v.y, s * v.x + c * v.y); }

vec2 refractEdge(vec2 p) {
    vec2 c = vec2(640.0, 285.0);
    vec2 q = abs(p - c) - (vec2(420.0, 205.0) - 78.0);
    float sd = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - 78.0;
    float e = smoothstep(-70.0, 0.0, sd);
    return normalize(c - p + 0.001) * e * 3.5 * sin(u_time * 0.9 + p.x * 0.03 + p.y * 0.045);
}

void cull() {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    v_local = vec2(0.0); v_rad = 1.0; v_col = vec4(0.0); v_world = vec2(0.0); v_stretch = 1.0; v_soft = 0.0;
}

void main() {
    float kind = in_a.x;
    int id = int(in_a.y + 0.5);
    vec4 A = u_jA[id], B = u_jB[id], C = u_jC[id], D = u_jD[id], E = u_jE[id];
    float pres = C.w * u_garden;
    if (pres < 0.003) { cull(); return; }
    float s = A.z, layer = A.w;
    float c = B.x, bclk = B.y, lean = B.z, sway = B.w, tclk = D.w;
    float ph = in_c.w;

    // ---- the bell breathes: it narrows and lengthens as it squeezes, its margin flares when relaxed
    float Rb = R0 * (1.0 - 0.20 * c);
    float Hb = H0 * (1.0 + 0.16 * c);
    float flare = 0.12 * (1.0 - 1.5 * c);
    vec2 piv = vec2(0.0, -0.35 * Hb);

    vec2 loc = vec2(0.0);       // position in animal space (pixels at scale 1)
    vec2 hang = vec2(0.0);      // extra offset that is NOT rotated with the bell (things that dangle)
    float bright = 1.0, tpar = 0.0, soft = 0.0;
    bool rigid = true;          // rotates with the bell
    float rad = in_c.x * (0.80 + 0.20 * s);

    if (kind < 0.5 || (kind > 1.5 && kind < 2.5)) {                 // bell surface / radial canals
        bool canal = kind > 1.5;
        float phi = in_a.z, th = in_a.w;
        float sp = sin(phi), cp = cos(phi);
        float shrink = canal ? 0.93 : 1.0;
        float r = Rb * sp * (1.0 + flare * pow(sp, 5.0)) * (1.0 + 0.03 * sin(7.0 * th + bclk * 0.8) * pow(sp, 4.0)) * shrink;
        float x3 = r * cos(th), z3 = r * sin(th);
        float y3 = -Hb * pow(cp, 0.9) * shrink + 0.18 * R0 * pow(sp, 7.0) * (1.0 + 0.6 * c);
        loc = vec2(x3, y3 + z3 * 0.15);
        float fres = pow(1.0 - abs(sp * sin(th)), 2.0);              // glow where the dome is seen edge-on
        bright = canal ? 0.55 : (0.22 + 0.95 * fres + 0.55 * pow(1.0 - sp, 2.0));
        if (sin(th) < 0.0) bright *= 0.65;                           // far wall is dimmer
    } else if (kind < 1.5) {                                         // glowing margin
        float th = in_a.w;
        float r = Rb * (1.0 + flare) * (1.0 + 0.03 * sin(7.0 * th + bclk * 0.8));
        float y = 0.18 * R0 * (1.0 + 0.6 * c) + in_b.x * (1.0 + 0.3 * c) + 0.045 * R0 * sin(5.0 * th + bclk * 1.1);
        loc = vec2(r * cos(th), y + r * sin(th) * 0.15);
        bright = (1.05 + 0.40 * (0.5 + 0.5 * sin(bclk * 1.5 + th * 3.0))) * (sin(th) < 0.0 ? 0.55 : 1.0);   // glow travels round the edge
    } else if (kind < 3.5) {                                         // bright core + gonad rings
        float x3 = in_a.z * Rb, z3 = in_a.w * Rb;
        loc = vec2(x3, -Hb * in_b.x + z3 * 0.15);
        bright = (0.65 + 0.45 * c) * in_b.y;
    } else if (kind < 4.5) {                                         // ruffled oral arms
        float t = in_a.z, ang = in_a.w;
        float len = in_b.x * R0 * (1.0 - 0.18 * c);
        vec2 base = rot(vec2(0.14 * Rb * cos(ang), 0.05 * R0 + 0.03 * Rb * sin(ang)) - piv, lean) + piv;
        float wave = sin(3.2 * (tclk * 0.9 - t * 2.0) + in_b.z);
        float ruf = sin(t * 26.0 + ang * 3.0 + tclk * 0.6) * 0.05 * R0 * t;
        loc = base;
        hang = vec2(wave * 0.12 * R0 * t * sway + ruf + in_b.y * 0.10 * R0 * (1.0 - 0.5 * t), t * len);
        bright = 0.9 * (1.0 - 0.5 * t);
        rigid = false;
    } else if (kind < 6.5) {                                         // tentacles + trailing filaments
        bool fil = kind > 5.5;
        float t = in_a.z, th = in_a.w;
        tpar = t;
        float L = in_b.x, ph2 = in_b.y, spd = in_b.z, lam = in_b.w;
        float r = Rb * (1.0 + flare) * (1.0 + 0.03 * sin(7.0 * th + bclk * 0.8));
        vec2 rim = vec2(r * cos(th), 0.18 * R0 * (1.0 + 0.6 * c) + 0.045 * R0 * sin(5.0 * th + bclk * 1.1) + r * sin(th) * 0.15);
        loc = rot(rim - piv, lean) + piv;                            // anchored to the (tilted) margin
        float len = L * R0 * (1.0 - 0.15 * c);
        // two travelling waves with their own speed/wavelength -> independent, never in lockstep
        float wave = sin(2.6 * (tclk * spd - t * lam) + ph2) + 0.45 * sin(5.1 * (tclk * spd * 1.3 - t * lam * 0.8) + ph2 * 2.1);
        float amp = (0.10 + 0.55 * t) * R0 * sway * (fil ? 1.5 : 1.0);
        vec2 lag = -E.xy * (fil ? 2.2 : 1.4) * t * t / s;           // dragged behind the swimming body
        hang = vec2(wave * amp * pow(t, 1.15) + lag.x + sin(lean) * 0.35 * t * len + rim.x * 0.15 * c * t * (1.0 - t),
                    t * len + lag.y) + vec2((fract(ph * 7.0) - 0.5) * 3.2 * (0.4 + t), (fract(ph * 13.0) - 0.5) * 2.0);
        bright = (fil ? 0.35 : 0.7) * (1.0 - 0.6 * t);
        rigid = false;
    } else if (kind < 7.5) {                                         // motes orbiting the animal
        loc = in_a.zw * R0 + 5.0 * vec2(sin(tclk * 0.3 + ph), cos(tclk * 0.37 + ph * 1.3));
        float tw = 0.5 + 0.5 * sin(tclk * 1.6 + ph * 20.0);
        bright = tw * tw * 1.4;
        rigid = false;
    } else {                                                         // faint halo blobs
        loc = in_a.zw * R0 + 3.0 * vec2(sin(tclk * 0.2 + ph), cos(tclk * 0.25 + ph * 1.3));
        rad = in_c.x * s * 0.7;
        soft = 1.0;
        bright = 0.8 + 0.4 * c;
        rigid = false;
    }
    if (rigid) loc = rot(loc - piv, lean) + piv;
    vec2 p = A.xy + (loc + hang) * s;
    p += u_shake * (0.2 + 0.4 * (1.0 - layer)) + refractEdge(p) * 0.6;

    vec3 col = (in_c.z > 1.5) ? vec3(0.86, 0.95, 1.0) : mix(C.rgb, D.rgb, clamp(in_c.z, 0.0, 1.0));
    col = mix(col, vec3(0.04, 0.16, 0.42), layer * 0.40);            // far animals: bluer, softer
    col *= bright * E.w * (0.85 + 0.30 * c) * (0.88 + 0.24 * sin(bclk * 1.3 + ph * 30.0));
    float a = in_c.y * pres;
    if (kind > 4.5 && kind < 6.5) a *= pow(1.0 - tpar, 0.7);        // tentacles fade out toward their tips
    if (kind > 7.5) a *= 1.0;

    vec2 ext = vec2(rad + 1.0);
    vec2 world = p + in_quad * ext;
    gl_Position = vec4(world.x / u_res.x * 2.0 - 1.0, 1.0 - world.y / u_res.y * 2.0, 0.0, 1.0);
    v_local = in_quad * ext;
    v_rad = rad;
    v_col = vec4(col, a);
    v_world = world;
    v_stretch = 1.0;
    v_soft = soft;
}
'''

CORAL_VERT = """
#version 330
in vec2 in_quad;
in vec4 in_a;   // x, y, radius, alpha
in vec4 in_b;   // r, g, b, glow
in vec4 in_c;   // sway weight, sway amplitude (px), phase, frequency (motes: rise speed)
in vec4 in_d;   // pulse phase, layer (0 near .. 1 far), kind, extra (rays: beam index)
uniform vec2 u_res;
uniform float u_time;
uniform vec2 u_shake;
uniform float u_coral;       // 0..1 how much of the garden has faded in
uniform float u_depth;
uniform vec4 u_beam[3];      // light shafts: x at top, width, strength, phase
out vec2 v_local;
out float v_rad;
out vec4 v_col;
out vec2 v_world;
out float v_stretch;
out float v_soft;

vec2 refractEdge(vec2 p) {
    vec2 c = vec2(640.0, 285.0);
    vec2 q = abs(p - c) - (vec2(420.0, 205.0) - 78.0);
    float sd = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - 78.0;
    float e = smoothstep(-70.0, 0.0, sd);
    return normalize(c - p + 0.001) * e * 3.5 * sin(u_time * 0.9 + p.x * 0.03 + p.y * 0.045);
}
float beamX(vec4 B, float y) { return B.x + 0.36 * (y - 54.0) + 10.0 * sin(u_time * 0.18 + B.w); }
float beamLight(vec2 p) {                  // how much of the light shafts fall on this point
    float acc = 0.0;
    for (int i = 0; i < 3; i++) {
        vec4 B = u_beam[i];
        float d = (p.x - beamX(B, p.y)) / (B.y * 0.85);
        acc += B.z * (0.7 + 0.3 * sin(u_time * 0.5 + B.w * 2.0)) * exp(-d * d);
    }
    return acc * (1.0 - 0.7 * smoothstep(0.75, 1.0, u_depth));
}
void cull() {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    v_local = vec2(0.0); v_rad = 1.0; v_col = vec4(0.0); v_world = vec2(0.0); v_stretch = 1.0; v_soft = 0.0;
}
void main() {
    float t = u_time, kind = in_d.z, layer = in_d.y;
    if (u_coral < 0.003) { cull(); return; }
    vec2 p = in_a.xy;
    vec3 col = in_b.rgb;
    float alpha = in_a.w * u_coral, rad = in_a.z, soft = 0.0;
    float sw = in_c.x, amp = in_c.y, ph = in_c.z, fr = in_c.w;
    float pulse = pow(0.5 + 0.5 * sin(t * 0.42 + in_d.x), 2.0);       // slow breathing glow, never a flash
    if (kind < 2.0 || (kind > 4.5)) {                                  // body / glowing tip / sand: sway with the current
        float cur = 0.6 * sin(t * 0.21 + p.y * 0.004) + 0.4 * sin(t * 0.13 + p.x * 0.003 + 1.7);
        float w = sw * sw;                                              // roots stay put, tips move most
        float sway = amp * w * (0.65 * sin(t * fr + ph + p.y * 0.02) + 0.35 * sin(t * fr * 1.7 + ph * 2.3) + 0.8 * cur);
        p.x += sway;
        p.y += 0.2 * sway * sin(t * fr * 0.7 + ph);
        if (kind < 0.5) col *= 1.0 + in_b.w * 0.45 * pulse;
        else if (kind < 1.5) { alpha *= 0.30 + 0.70 * pulse; col *= 1.0 + 0.2 * pulse; }
        else col *= 0.85 + 0.30 * (0.5 + 0.5 * sin(p.x * 0.05 + t * 0.5 + sin(p.y * 0.08 + t * 0.4) * 2.0));   // caustics on the sand
        float bm = beamLight(p);                                        // selected corals catch the light shafts
        col = col * (1.0 + 0.30 * bm * (kind < 1.5 ? 1.0 : 0.6)) + bm * vec3(0.025, 0.06, 0.08);
    } else if (kind < 2.5) {                                           // motes rising and wandering around the corals
        float range = 110.0;
        float f = mod(t * fr + ph * 20.0, range) / range;
        p.y -= f * range;
        p.x += 8.0 * sin(t * 0.2 + ph) + 4.0 * sin(t * 0.37 + ph * 2.0);
        alpha *= smoothstep(0.0, 0.15, f) * (1.0 - smoothstep(0.7, 1.0, f)) * (0.5 + 0.5 * sin(t * 1.6 + ph * 13.0));
    } else if (kind < 3.5) {                                           // faint halo around a glowing tip
        alpha *= 0.4 + 0.6 * pulse;
        soft = 1.0;
    } else {                                                           // light shaft dots
        vec4 B = u_beam[int(in_d.w + 0.5)];
        p.x = beamX(B, p.y) + in_a.x * B.y * 0.6;
        float along = (p.y - 54.0) / 460.0;
        alpha *= B.z * (0.55 + 0.45 * sin(t * 0.5 + B.w * 2.0 + ph)) * (1.0 - 0.5 * along)
               * (1.0 - 0.7 * smoothstep(0.75, 1.0, u_depth));
    }
    float fogk = (kind > 0.5 && kind < 1.5) ? 0.30 : 0.55;
    if (kind < 1.5 || kind > 2.5 && kind < 3.5 || kind > 4.5) {        // distance: bluer, softer
        col = mix(col, vec3(0.05, 0.08, 0.24), layer * fogk);
        if (kind < 0.5 || kind > 4.5) alpha *= 1.0 - 0.22 * layer;
    }
    col *= 1.0 - 0.25 * smoothstep(0.80, 1.0, u_depth);
    p += u_shake * (0.15 + 0.5 * (1.0 - layer)) + refractEdge(p) * 0.6;
    vec2 ext = vec2(rad + 1.0);
    vec2 world = p + in_quad * ext;
    gl_Position = vec4(world.x / u_res.x * 2.0 - 1.0, 1.0 - world.y / u_res.y * 2.0, 0.0, 1.0);
    v_local = in_quad * ext;
    v_rad = rad;
    v_col = vec4(col, alpha);
    v_world = world;
    v_stretch = 1.0;
    v_soft = soft;
}
"""

FISH_VERT = """
#version 330
#define NF 18
in vec2 in_quad;
in vec4 in_a;   // kind, fish id, u, v
in vec4 in_b;   // radius, alpha, colour mix, phase
uniform vec2 u_res;
uniform float u_time;
uniform vec2 u_shake;
uniform float u_coral;
uniform float u_depth;
uniform vec4 u_fA[NF];   // x, y, half-length px, direction (+1 right / -1 left)
uniform vec4 u_fB[NF];   // tail phase, speed, layer, presence
uniform vec4 u_fC[NF];   // body colour, slope (nose up/down)
uniform vec4 u_fD[NF];   // belly / fin colour
out vec2 v_local;
out float v_rad;
out vec4 v_col;
out vec2 v_world;
out float v_stretch;
out float v_soft;
void cull() {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    v_local = vec2(0.0); v_rad = 1.0; v_col = vec4(0.0); v_world = vec2(0.0); v_stretch = 1.0; v_soft = 0.0;
}
void main() {
    int id = int(in_a.y + 0.5);
    float kind = in_a.x;
    vec4 A = u_fA[id], B = u_fB[id], C = u_fC[id], D = u_fD[id];
    float pres = B.w * u_coral;
    if (pres < 0.003) { cull(); return; }
    float u = in_a.z, v = in_a.w, S = A.z, dir = A.w, tp = B.x;
    float hh = 0.30 * (0.35 + 0.65 * sqrt(max(1.0 - u * u, 0.0)));         // body thickness along its length
    float und = 0.10 * sin(tp - u * 2.4) * (0.5 - 0.5 * u);                   // undulation grows toward the tail
    float xl = u, yl = 0.0, shade = 0.5;
    vec3 col;
    if (kind < 0.5) { xl = u; yl = v * hh + und; shade = 0.5 + 0.5 * v; col = mix(C.rgb, D.rgb, shade); }
    else if (kind < 1.5) {                                                    // tail fan, flapping
        xl = -1.0 - 0.55 * u;
        yl = v * (0.06 + 0.30 * u) + (0.10 + 0.22 * u) * sin(tp + 2.4);
        col = mix(C.rgb, D.rgb, 0.6);
    } else if (kind < 2.5) {                                                  // dorsal fin
        xl = u;
        yl = -0.30 * (0.35 + 0.65 * sqrt(max(1.0 - u * u, 0.0))) - v * 0.24 * (1.0 - pow((u - 0.05) / 0.45, 2.0)) + und;
        col = mix(C.rgb, D.rgb, 0.5);
    } else if (kind < 3.5) {                                                  // pelvic fin
        xl = 0.15 - u * 0.3;
        yl = 0.28 + v * 0.16 + und;
        col = D.rgb;
    } else if (kind < 4.5) { xl = 0.62; yl = -0.07 + und; col = vec3(0.96, 0.98, 1.0); }   // eye
    else { xl = u * 0.8; yl = 0.0 + und; col = mix(C.rgb, vec3(0.8, 1.0, 1.0), 0.6) * 1.2; }  // glowing lateral line
    vec2 p = vec2(A.x + dir * xl * S, A.y + yl * S + C.w * xl * S);
    col = mix(col, vec3(0.05, 0.08, 0.24), B.z * 0.35);
    // soft, natural light: a faint per-dot shimmer, and the deep water slowly (never fully) dims them
    float shimmer = 0.94 + 0.06 * sin(u_time * 1.3 + in_b.w * 6.0 + A.x * 0.02);
    col *= 0.84 * shimmer * (1.0 - 0.20 * smoothstep(0.80, 1.0, u_depth));
    p += u_shake * (0.15 + 0.5 * (1.0 - B.z));
    float rad = in_b.x;
    vec2 ext = vec2(rad + 1.0);
    vec2 world = p + in_quad * ext;
    gl_Position = vec4(world.x / u_res.x * 2.0 - 1.0, 1.0 - world.y / u_res.y * 2.0, 0.0, 1.0);
    v_local = in_quad * ext;
    v_rad = rad;
    v_col = vec4(col, in_b.y * pres);
    v_world = world;
    v_stretch = 1.0;
    v_soft = 0.0;
}
"""

BACKING = (0.025, 0.025, 0.05)


# ----------------------------------------------------------------------------
# DEPTH: one number drives the whole descent
# ----------------------------------------------------------------------------
def waterline_for(depth):
    """pixel height of the waterline in the window for a given depth (0 = above the window)"""
    s = float(smoothstep(0.06, 0.50, depth))
    wl = 540.0 - 640.0 * s
    if depth > 0.50:                       # past the window top the surface keeps receding
        wl -= (depth - 0.50) * 900.0
    return wl


def spill_for(depth):
    return float(smoothstep(0.28, 0.60, depth) * (1.0 - 0.55 * smoothstep(0.70, 1.0, depth)))


class Descent:
    """Autopilot dives slowly after the first fireworks; UP/DOWN take over (A hands it back)."""
    START_DELAY = 12.0
    RATE = 0.0095                          # depth per second under autopilot

    def __init__(self):
        self.reset()

    def reset(self):
        self.depth = 0.0
        self.target = 0.0
        self.auto = True
        self.clock = 0.0
        self.goal = None

    def glide_to(self, goal):
        """G key: float down (or up) to a depth at 0.1/s, then stop"""
        self.goal = goal
        self.auto = False

    def jump_to(self, depth):
        self.depth = self.target = depth
        self.clock = self.START_DELAY          # autopilot carries on deeper from here

    def update(self, dt, up, down, paused):
        if paused:              # frozen: no key input, no glide, no autopilot, no easing -> resumes without a jump
            return
        if up or down:
            self.auto = False
            self.goal = None
            self.target += (down - up) * 0.14 * dt
        elif self.goal is not None:
            step = min(abs(self.goal - self.target), 0.10 * dt)
            self.target += step if self.goal > self.target else -step
            if abs(self.goal - self.target) < 1e-4:
                self.goal = None
        elif self.auto and not paused:
            self.clock += dt
            if self.clock > self.START_DELAY:
                ramp = min(1.0, (self.clock - self.START_DELAY) / 6.0)
                self.target += self.RATE * ramp * dt
        self.target = min(1.0, max(0.0, self.target))
        self.depth += (self.target - self.depth) * (1.0 - math.exp(-dt * 2.5))


class Renderer:
    LAYERS = ("outside", "underwater", "interior", "spill", "reflect")

    def __init__(self, ctx, scene):
        import moderngl
        self.mgl = moderngl
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        self.fw_prog = ctx.program(vertex_shader=FW_VERT, fragment_shader=FW_FRAG)
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], np.float32)
        self.quad = ctx.buffer(quad.tobytes())
        self.layers = {}
        for name in self.LAYERS:
            arr = np.ascontiguousarray(scene[name], np.float32)
            buf = ctx.buffer(arr.tobytes())
            vao = ctx.vertex_array(self.prog, [
                (self.quad, "2f", "in_quad"),
                (buf, "2f 1f 4f 3f/i", "in_pos", "in_radius", "in_col", "in_fx")])
            self.layers[name] = (vao, len(arr), buf)
        fw = np.ascontiguousarray(scene["fireworks"], np.float32)
        self.fw_buf = ctx.buffer(fw.tobytes())
        self.fw_vao = ctx.vertex_array(self.fw_prog, [
            (self.quad, "2f", "in_quad"),
            (self.fw_buf, "2f 2f 2f 3f 4f 4f/i", "in_o", "in_v", "in_t", "in_col", "in_p", "in_k")])
        self.fw_n = len(fw)
        jd = np.ascontiguousarray(scene["jelly_dots"], np.float32)
        self.jelly_prog = ctx.program(vertex_shader=JELLY_VERT, fragment_shader=FW_FRAG)
        self.jelly_buf = ctx.buffer(jd.tobytes())
        self.jelly_vao = ctx.vertex_array(self.jelly_prog, [
            (self.quad, "2f", "in_quad"), (self.jelly_buf, "4f 4f 4f/i", "in_a", "in_b", "in_c")])
        self.jelly_n = len(jd)
        # --- Phase 5: coral garden (4 draw lists) and fish
        self.coral = scene["coral"]
        self._arr(self.prog, "u_kA", self.coral.kA)          # static: glow sources never change
        self._arr(self.prog, "u_kC", self.coral.kC)
        self.coral_prog = ctx.program(vertex_shader=CORAL_VERT, fragment_shader=FW_FRAG)
        self.coral_prog["u_res"].value = (float(W), float(H))
        self._arr(self.coral_prog, "u_beam", self.coral.beams)   # static: the three light shafts
        self.coral_vaos = {}
        for key in ("back_body", "back_glow", "front_body", "front_glow"):
            arr = np.ascontiguousarray(getattr(self.coral, key), np.float32)
            if len(arr):
                buf = ctx.buffer(arr.tobytes())
                vao = ctx.vertex_array(self.coral_prog, [(self.quad, "2f", "in_quad"),
                                                         (buf, "4f 4f 4f 4f/i", "in_a", "in_b", "in_c", "in_d")])
                self.coral_vaos[key] = (vao, len(arr), buf)
        fd = np.ascontiguousarray(scene["fish"].dots, np.float32)
        self.fish_prog = ctx.program(vertex_shader=FISH_VERT, fragment_shader=FW_FRAG)
        self.fish_prog["u_res"].value = (float(W), float(H))
        self.fish_buf = ctx.buffer(fd.tobytes())
        self.fish_vao = ctx.vertex_array(self.fish_prog, [(self.quad, "2f", "in_quad"),
                                                          (self.fish_buf, "4f 4f/i", "in_a", "in_b")])
        self.fish_n = len(fd)
        self.jelly_prog["u_res"].value = (float(W), float(H))
        self.prog["u_res"].value = (float(W), float(H))
        self.fw_prog["u_res"].value = (float(W), float(H))

    def _draw(self, name, mode, clip):
        self.prog["u_mode"].value = mode
        self.prog["u_clip"].value = clip
        vao, n, _ = self.layers[name]
        vao.render(self.mgl.TRIANGLE_STRIP, instances=n)

    @staticmethod
    def _arr(prog, name, a):
        if name in prog:
            prog[name].write(np.ascontiguousarray(a, np.float32).tobytes())

    @staticmethod
    def _u(prog, name, value):
        if name in prog:
            prog[name].value = value

    def _coral_pass(self, which, t, shake, depth, coral, fish=None):
        """one side of the coral garden: solid dots (alpha) then glowing dots (additive); fish swim
        between the back corals and the animals / front corals"""
        ctx, mgl, cp = self.ctx, self.mgl, self.coral_prog
        for k, v in (("u_time", t), ("u_shake", shake), ("u_coral", coral), ("u_depth", depth)):
            self._u(cp, k, v)
        ctx.blend_func = (mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA)
        if which + "_body" in self.coral_vaos:
            vao, n, _ = self.coral_vaos[which + "_body"]
            vao.render(mgl.TRIANGLE_STRIP, instances=n)
        ctx.blend_func = (mgl.SRC_ALPHA, mgl.ONE)
        if which + "_glow" in self.coral_vaos:
            vao, n, _ = self.coral_vaos[which + "_glow"]
            vao.render(mgl.TRIANGLE_STRIP, instances=n)
        ctx.blend_func = (mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA)
        if which == "back" and fish is not None:
            fp = self.fish_prog
            for k, v in (("u_time", t), ("u_shake", shake), ("u_coral", coral), ("u_depth", depth)):
                self._u(fp, k, v)
            for k, a in fish.uniforms().items():
                self._arr(fp, k, a)
            self.fish_vao.render(mgl.TRIANGLE_STRIP, instances=self.fish_n)

    def render(self, t, depth=0.0, fbo=None, jelly=None, fish=None):
        ctx, mgl, prog, fp = self.ctx, self.mgl, self.prog, self.fw_prog
        if fbo is not None:
            fbo.use()
        ctx.viewport = (0, 0, W, H)
        ctx.clear(*BACKING)
        ctx.enable(mgl.BLEND)
        shake = train_shake(t)
        wl = waterline_for(depth)
        prog["u_time"].value = t
        prog["u_shake"].value = shake
        prog["u_lamps"].value = lamp_flicker(t)
        prog["u_wl"].value = wl
        prog["u_depth"].value = depth
        prog["u_spill"].value = spill_for(depth)
        garden = float(smoothstep(0.50, 0.74, depth))     # the garden fades in as the window goes under
        prog["u_garden"].value = garden
        coral = float(smoothstep(CORAL_FADE[0], CORAL_FADE[1], depth))   # the coral garden fades in a little later
        prog["u_coral"].value = coral
        ju = jelly.uniforms() if jelly is not None else None
        if ju is not None:
            for k in ("u_jA", "u_jG"):
                self._arr(prog, k, ju[k])
        ctx.blend_func = (mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA)
        # 1) the lakeside world (it sinks behind the rising water; skipped once fully covered)
        if depth < 0.52:
            self._draw("outside", 1, 1)
        # 2) the underwater world, fading in below the waterline
        if depth > 0.04:
            self._draw("underwater", 1, 1)
        # 2a) coral garden, back half: far + middle corals, light shafts, then the fish between them
        if coral > 0.003:
            self._coral_pass("back", t, shake, depth, coral, fish)
        # 2b) the jellyfish garden: additive dots, each animal animated from its own state
        if garden > 0.003 and ju is not None:
            jp = self.jelly_prog
            for k in ("u_jA", "u_jB", "u_jC", "u_jD", "u_jE"):
                self._arr(jp, k, ju[k])
            jp["u_time"].value = t
            jp["u_shake"].value = shake
            jp["u_garden"].value = garden
            ctx.blend_func = (mgl.SRC_ALPHA, mgl.ONE)
            self.jelly_vao.render(mgl.TRIANGLE_STRIP, instances=self.jelly_n)
            ctx.blend_func = (mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA)
        # 2c) coral garden, front half: the big near corals overlap the jellyfish and fish
        if coral > 0.003:
            self._coral_pass("front", t, shake, depth, coral)
        # 3) fireworks in the sky and in the lake (additive light), until the sea swallows them
        if depth < 0.52:
            ctx.blend_func = (mgl.SRC_ALPHA, mgl.ONE)
            fp["u_tc"].value = t % FW_CYCLE
            fp["u_time"].value = t
            fp["u_shake"].value = shake
            fp["u_wl"].value = wl
            for refl in (0, 1):
                fp["u_refl"].value = refl
                self.fw_vao.render(mgl.TRIANGLE_STRIP, instances=self.fw_n)
        # 4) the interior never moves
        ctx.blend_func = (mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA)
        self._draw("interior", 0, 0)
        # 5) blue light from the sea on the frame and nearby surfaces, then glass reflections
        ctx.blend_func = (mgl.SRC_ALPHA, mgl.ONE)
        if spill_for(depth) > 0.001:
            self._draw("spill", 5, 0)
        self._draw("reflect", 2, 0)



# ----------------------------------------------------------------------------
# 11. RUN MODES
# ----------------------------------------------------------------------------
def vignette(x, y):
    qx, qy = x / W - 0.5, y / H - 0.5
    return 1.0 - 0.5 * np.clip(np.hypot(qx, qy * 1.1) * 1.45, 0, 1.4) ** 2.3


def save_cpu_preview(scene, path, ss=2):
    """draws the very same dots with Pillow - slow, but needs no GPU"""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (W * ss, H * ss), tuple(int(c * 255) for c in BACKING))
    dr = ImageDraw.Draw(img, "RGBA")
    for name in ("outside", "interior"):
        a = scene[name]
        if name == "outside":
            a = a[sdr(a[:, 0], a[:, 1], WIN) < 1.0]
        col = np.clip(a[:, 3:6] * vignette(a[:, 0], a[:, 1])[:, None], 0, 1)
        c8 = (col * 255).astype(int)
        al = (a[:, 6] * 255).astype(int)
        for i in range(len(a)):
            x, y, r = a[i, 0] * ss, a[i, 1] * ss, a[i, 2] * ss
            dr.ellipse([x - r, y - r, x + r, y + r], fill=(c8[i, 0], c8[i, 1], c8[i, 2], al[i]))
    img.resize((W, H), Image.LANCZOS).save(path)


def run_shot(scene, path, t, depth=0.0):
    scene["jelly"].simulate(t)
    scene["fish"].simulate(t)
    import moderngl
    from PIL import Image
    try:
        ctx = moderngl.create_standalone_context(require=330)
    except Exception:
        ctx = moderngl.create_standalone_context(require=330, backend="egl")
    fbo = ctx.simple_framebuffer((W, H))
    r = Renderer(ctx, scene)
    r.render(t, depth, fbo, scene["jelly"], scene["fish"])
    Image.frombytes("RGB", (W, H), fbo.read()).transpose(Image.FLIP_TOP_BOTTOM).save(path)


APP_TITLE = "The Last Train Beneath the Sea \u2013 Phase 5: The Luminous Coral Garden"


def run_window(scene, garden=False):
    import moderngl
    import pygame
    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.set_caption(APP_TITLE)
    pygame.display.set_mode((W, H), pygame.OPENGL | pygame.DOUBLEBUF, vsync=1)
    ctx = moderngl.create_context()
    renderer = Renderer(ctx, scene)
    clock = pygame.time.Clock()
    descent = Descent()
    jelly = scene["jelly"]
    fish = scene["fish"]
    if garden:
        descent.jump_to(GARDEN_DEPTH)
    t, paused, running, shots = 0.0, False, True, 0
    last_title = None
    while running:
        dt = min(clock.tick(60) / 1000.0, 0.1)
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key == pygame.K_r:
                    t = 0.0
                    descent.reset()
                    jelly.reset()
                    fish.reset()
                    if garden:
                        descent.jump_to(GARDEN_DEPTH)
                elif e.key == pygame.K_a:
                    descent.auto = True
                    descent.goal = None
                elif e.key == pygame.K_g:
                    descent.glide_to(GARDEN_DEPTH)
                elif e.key == pygame.K_s:
                    from PIL import Image
                    data = ctx.screen.read(components=3)
                    Image.frombytes("RGB", (W, H), data).transpose(Image.FLIP_TOP_BOTTOM).save(f"shot_{shots:02d}.png")
                    shots += 1
        keys = pygame.key.get_pressed()
        descent.update(dt, keys[pygame.K_UP], keys[pygame.K_DOWN], paused)
        if not paused:
            t += dt
            jelly.update(dt)
            fish.update(dt)
        renderer.render(t, descent.depth, None, jelly, fish)
        pygame.display.flip()
        title = "%s  |  depth %d%% %s%s" % (APP_TITLE, round(descent.depth * 100),
                                            "(auto)" if descent.auto else "(manual: A autopilot, G garden)",
                                            "  |  PAUSED" if paused else "")
        if title != last_title:                      # only touch the window when the text changes
            pygame.display.set_caption(title)
            last_title = title
    pygame.quit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", help="CPU render to PNG (no GPU needed)")
    ap.add_argument("--shot", help="headless GPU render to PNG")
    ap.add_argument("--time", type=float, default=3.0)
    ap.add_argument("--depth", type=float, default=0.0)
    ap.add_argument("--garden", action="store_true", help="start inside the jellyfish garden")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    scene = build_scene(args.seed)
    print({k: len(v) for k, v in scene.items() if hasattr(v, "shape")}, "dots")
    if args.preview:
        save_cpu_preview(scene, args.preview)
    elif args.shot:
        run_shot(scene, args.shot, args.time, args.depth)
    else:
        run_window(scene, args.garden)


if __name__ == "__main__":
    main()