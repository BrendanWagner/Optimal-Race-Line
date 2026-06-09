"""
track.py
--------
Loads Silverstone track geometry and produces the 4-column track table:
    s       - arc length along centerline (m)
    n_min   - right boundary distance from centerline (negative, m)
    n_max   - left boundary distance from centerline (positive, m)
    kappa   - signed curvature of centerline (1/m)

Also stores the centerline (x, y) and heading angle for plotting.

# SWAP THIS block (lines marked ## FASTF1 ##) to load live data:
#
#   import fastf1
#   session = fastf1.get_session(2023, 'Silverstone', 'Q')
#   session.load()
#   lap = session.laps.pick_fastest()
#   pos = lap.get_pos_data()
#   # pos contains X, Y columns in meters
#   # You'll also need circuit_info for track boundaries:
#   circuit_info = session.get_circuit_info()
#   # Then replace the raw_centerline below with the loaded coordinates.
"""

import numpy as np
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d
import fastf1


_RAW = np.array([
    # Start/finish straight
    [0,    0],   [100,   5],  [200,  12],  [300,  18],  [400,  22],
    [500,  24],  [600,  24],  [700,  22],
    # Vale / Club complex
    [780,  18],  [840,   5],  [880, -20],  [900, -50],
    [895, -90],  [870,-130],  [840,-160],
    # Loop toward Abbey
    [800,-185],  [740,-195],  [680,-190],  [620,-175],
    [570,-150],  [540,-120],
    # Abbey corner
    [520, -80],  [510,  -40], [520,   0],  [540,  35],
    # Farm straight / Village
    [570,  65],  [610,  90],  [660, 108],  [720, 118],
    [800, 122],  [880, 120],  [950, 112],
    # Loop / Aintree
    [1010, 95],  [1055, 70],  [1075, 38],  [1070,  0],
    [1050,-35],  [1010,-60],  [960, -72],
    # Hangar straight
    [890, -75],  [810, -72],  [730, -65],  [650, -55],
    [570, -42],  [500, -28],
    # Stowe
    [440, -10],  [400,  15],  [375,  48],  [368,  85],
    [375, 120],  [395, 150],  [425, 172],
    # Vale exit / new section
    [465, 185],  [510, 190],  [555, 185],  [595, 170],
    [625, 148],  [648, 120],  [658,  88],
    # Club
    [655,  55],  [640,  24],  [615,   0],  [580, -18],
    [540, -25],  [490, -20],  [450,  -5],
    # Back toward S/F
    [400,  15],  [330,  20],  [250,  22],  [150,  16],
    [50,    7],  [0,     0],
])

# Alternative _RAW
session = fastf1.get_session(2023, 'Silverstone', 'Q')
session.load(telemetry=True, weather=False, messages=False)

circuit_info = session.get_circuit_info()
corners = circuit_info.corners  # DataFrame with X, Y columns

# Get the track boundary from a fast lap
lap = session.laps.pick_fastest()
# pos = lap.get_pos_data(extra_interpolate_edges=True)
pos = lap.get_pos_data() # I think this might be better than the previous line

_RAW = pos[['X', 'Y']].dropna().values / 10.0

# ---------------------------------------------------------------------------
# Raw Silverstone centerline waypoints (x, y) in metres, roughly to scale.
# Traced from the 2023 F1 circuit layout. The circuit is ~5.891 km.
# Origin near the start/finish straight.
# ---------------------------------------------------------------------------

# session = fastf1.get_session(2023, 'Silverstone', 'Q')
# session.load()
# lap = session.laps.pick_fastest()
# pos = lap.get_pos_data()
# # pos contains X, Y columns in meters
# # You'll also need circuit_info for track boundaries:
# circuit_info = session.get_circuit_info()


def _build_spline(pts, smoothing=None):
    """Fit a closed parametric spline through 2-D waypoints."""
    x, y = pts[:, 0], pts[:, 1]
    tck, _ = splprep([x, y], s=smoothing if smoothing else 0,
                     per=True, k=3)
    return tck


def _resample(tck, N):
    """Evaluate spline at N equally-spaced parameter values."""
    u = np.linspace(0, 1, N, endpoint=False)
    xy = np.array(splev(u, tck)).T          # (N, 2)
    return xy


def _arc_length(xy):
    """Cumulative arc length along a closed polyline."""
    d = np.linalg.norm(np.diff(xy, axis=0, append=xy[:1]), axis=1)
    s = np.concatenate([[0], np.cumsum(d[:-1])])
    return s, d


def _curvature(xy):
    """
    Signed curvature at each point of a closed polyline.
    Uses central finite differences on x(s), y(s).
    Positive = left turn, negative = right turn.
    """
    # First derivatives
    dx  = np.gradient(xy[:, 0])
    dy  = np.gradient(xy[:, 1])
    # Second derivatives
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    # Signed curvature formula: kappa = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)
    num   = dx * ddy - dy * ddx
    denom = (dx**2 + dy**2) ** 1.5
    kappa = np.where(np.abs(denom) > 1e-9, num / denom, 0.0)
    return kappa


def _smooth(arr, sigma=2.0):
    """Gaussian smooth a periodic array."""
    return gaussian_filter1d(arr, sigma=sigma, mode='wrap')


def _boundary_distances(centerline, heading, half_width_left, half_width_right):
    """
    Given a centerline and per-point track half-widths, compute n_min / n_max.

    For our hardcoded track we use a piecewise-constant track width profile
    that mimics Silverstone's real geometry (wide straights, narrow chicanes).

    ## FASTF1 ## Replace this function body with actual boundary projection:
    ##           project FastF1's inner/outer boundary polylines onto the
    ##           normal direction at each centerline station.
    """
    N = len(centerline)
    n_max = np.full(N,  half_width_left)   # left  boundary (positive)
    n_min = np.full(N, -half_width_right)  # right boundary (negative)

    # Width profile: narrower in corners, wider on straights
    # We modulate by |kappa| - high curvature -> narrower track
    return n_min, n_max


def build_track(N=500):
    """
    Main entry point.  Returns a dict with keys:
        s, n_min, n_max, kappa   - the NLP track table (arrays length N)
        x, y, heading            - centerline geometry for plotting
        total_length             - lap distance in metres
    """
    tck = _build_spline(_RAW)
    cl  = _resample(tck, N)                    # (N, 2) centerline

    s, ds = _arc_length(cl)
    total_length = s[-1] + ds[-1]              # close the loop

    # Heading angle at each point
    dx = np.gradient(cl[:, 0])
    dy = np.gradient(cl[:, 1])
    heading = np.arctan2(dy, dx)

    # Raw then smoothed curvature
    kappa_raw = _curvature(cl)
    kappa = _smooth(kappa_raw, sigma=3.0)

    # Track width: base 8 m each side, narrowed in tight corners
    abs_kappa_norm = np.abs(kappa) / (np.abs(kappa).max() + 1e-9)
    half_left  = 8.0 - 2.5 * abs_kappa_norm   # 5.5 - 8 m
    half_right = 8.0 - 2.5 * abs_kappa_norm

    n_min, n_max = _boundary_distances(cl, heading, half_left, half_right)

    # Normalize s to start at 0
    s = s / total_length * total_length        # already starts at 0

    return dict(
        s            = s,
        n_min        = n_min,
        n_max        = n_max,
        kappa        = kappa,
        x            = cl[:, 0],
        y            = cl[:, 1],
        heading      = heading,
        total_length = total_length,
        ds           = np.full(N, total_length / N),
    )


if __name__ == '__main__':
    track = build_track()
    print(f"Total lap length : {track['total_length']:.1f} m")
    print(f"Grid points      : {len(track['s'])}")
    print(f"Max |kappa|      : {np.abs(track['kappa']).max():.4f} 1/m")
    print(f"Track width range: {(track['n_max']-track['n_min']).min():.1f} - "
          f"{(track['n_max']-track['n_min']).max():.1f} m")
