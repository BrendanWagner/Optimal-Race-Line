"""
vehicle.py
----------
Point-mass vehicle model parameters and derived quantities.

The vehicle is fully described by:
    mu      - friction coefficient (dimensionless)
    g       - gravitational acceleration (m/s^2)
    v_max   - maximum speed (m/s)

The friction circle constraint is:
    a_x^2 + a_y^2 <= (mu * g)^2

All methods are pure functions of scalars or numpy arrays.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Vehicle parameters — generic performance / GT car
# ---------------------------------------------------------------------------
MU    = 1.5          # friction coefficient
G     = 9.81         # m/s^2
V_MAX = 80.0         # m/s  (~288 km/h) — realistic for Silverstone GT car
V_MIN = 5.0          # m/s  — numerical floor, avoids 1/v singularity

GRIP  = MU * G       # combined grip limit (m/s^2) ~14.7 m/s^2


def max_cornering_speed(kappa):
    """
    Maximum speed sustainable at a given curvature, from the friction circle
    with a_x = 0 (pure cornering).

        a_y = v^2 * kappa  =>  v_max_corner = sqrt(mu*g / |kappa|)

    Used as the warm-start speed profile.
    """
    abs_k = np.abs(kappa)
    # Avoid division by zero on straights
    safe_k = np.where(abs_k > 1e-6, abs_k, 1e-6)
    v = np.sqrt(GRIP / safe_k)
    return np.clip(v, V_MIN, V_MAX)


def lateral_accel(v, kappa, n, chi=0.0):
    """
    Lateral acceleration in the Frenet frame.

        a_y = v^2 * (kappa + d_chi/ds) / (1 - n*kappa)

    In the quasi-steady approximation (d_chi/ds ~ 0 along the solution),
    this simplifies to:

        a_y = v^2 * kappa / (1 - n*kappa)

    We keep the full expression for the collocation constraints.
    """
    scale = 1.0 - n * kappa
    # Clamp scale away from zero (physically: car can't be at center of curvature)
    scale = np.where(np.abs(scale) > 0.05, scale, np.sign(scale) * 0.05)
    return v**2 * kappa / scale


def friction_residual(a_x, a_y):
    """
    Returns the friction circle utilization in [0, 1].
    Values > 1 violate the constraint.
    """
    return np.sqrt(a_x**2 + a_y**2) / GRIP


def max_ax_given_ay(a_y):
    """
    Maximum |a_x| available given a lateral acceleration a_y.
    From the friction circle boundary:
        |a_x| = sqrt((mu*g)^2 - a_y^2)
    """
    remaining = GRIP**2 - a_y**2
    return np.sqrt(np.clip(remaining, 0, None))


if __name__ == '__main__':
    print(f"Grip limit : {GRIP:.2f} m/s^2  ({GRIP/G:.2f} g)")
    print(f"V_max      : {V_MAX:.1f} m/s  ({V_MAX*3.6:.1f} km/h)")
    # Spot check: at kappa = 0.02 (50 m radius), max cornering speed?
    v = max_cornering_speed(np.array([0.02]))
    print(f"Max speed at kappa=0.02 (R=50m): {v[0]:.1f} m/s  ({v[0]*3.6:.1f} km/h)")
