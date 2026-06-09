"""
optimize.py
-----------
Formulates and solves the racing line optimal control problem as an NLP.

Decision variables (flattened into a single vector z):
    n[i]    - lateral position at station i (m from centerline)
    chi[i]  - heading error at station i (rad)
    v[i]    - speed at station i (m/s)
    ax[i]   - longitudinal acceleration at station i (m/s^2)

    Total: 4 * N variables

Objective:
    min  sum_i  ds / v[i]        (lap time approximation)

Constraints:
    Equality   - trapezoidal collocation of the 3 ODEs (3*N constraints)
    Equality   - periodicity: state at i=N equals state at i=0  (3 constraints)
    Inequality - friction circle at each node                   (N constraints)
    Inequality - track bounds at each node                      (2*N constraints)
    Bounds     - v in [V_MIN, V_MAX], ax in [-GRIP, GRIP], n in [n_min, n_max]
"""

import numpy as np
from scipy.optimize import minimize
import vehicle as V


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------
def _unpack(z, N):
    n   = z[0*N : 1*N]
    chi = z[1*N : 2*N]
    v   = z[2*N : 3*N]
    ax  = z[3*N : 4*N]
    return n, chi, v, ax


def _pack(n, chi, v, ax):
    return np.concatenate([n, chi, v, ax])


# ---------------------------------------------------------------------------
# Dynamics: f(state, control) -> state derivative w.r.t. s
# ---------------------------------------------------------------------------
def _f(n, chi, v, ax, kappa, ds):
    """
    State derivatives w.r.t. arc length s (not time).

    Frenet-frame kinematics:
        dn/ds   = sin(chi)
        dchi/ds = (a_y / v^2) - kappa   [from centripetal balance]
        dv/ds   = ax / v                [chain rule: dv/dt = ax, dt/ds = 1/v]

    a_y is determined by the path geometry:
        a_y = v^2 * (kappa_path)
    where kappa_path = kappa + dchi/ds, but we resolve this self-consistently.
    """
    scale = np.clip(1.0 - n * kappa, 0.05, None)

    # a_y from centripetal balance in curvilinear frame
    # a_y = v^2 * kappa / scale  (quasi-steady, chi small)
    ay = v**2 * kappa / scale

    dn_ds   = np.sin(chi)
    # dchi/ds: heading error changes to follow path curvature
    dchi_ds = ay / np.clip(v**2, 0.1, None) - kappa
    dv_ds   = ax / np.clip(v, V.V_MIN, None)

    return dn_ds, dchi_ds, dv_ds, ay


# ---------------------------------------------------------------------------
# Objective: minimize lap time = sum ds/v
# ---------------------------------------------------------------------------
def _objective(z, N, ds):
    _, _, v, ax = _unpack(z, N)
    v_safe = np.clip(v, V.V_MIN, None)
    smoothness = 1e-3 * np.sum(np.diff(ax)**2)
    return np.sum(ds / v_safe) + smoothness
    # return np.sum(ds / v_safe)


def _obj_grad(z, N, ds):
    _, _, v, _ = _unpack(z, N)
    v_safe = np.clip(v, V.V_MIN, None)
    grad = np.zeros_like(z)
    grad[2*N:3*N] = -ds / v_safe**2
    return grad


# ---------------------------------------------------------------------------
# Collocation constraints (equality)
# Returns vector of residuals; constraint is residual == 0
# ---------------------------------------------------------------------------
def _collocation(z, N, ds, kappa):
    n, chi, v, ax = _unpack(z, N)

    # Next-index (periodic wrap)
    np1 = np.roll(n,   -1)
    cp1 = np.roll(chi, -1)
    vp1 = np.roll(v,   -1)

    # Derivatives at current and next node
    dn_i,   dchi_i,   dv_i,   _ = _f(n,   chi, v,   ax,           kappa,    ds)
    dn_ip1, dchi_ip1, dv_ip1, _ = _f(np1, cp1, vp1, np.roll(ax,-1), np.roll(kappa,-1), ds)

    # Trapezoidal rule: x_{i+1} = x_i + ds/2 * (f_i + f_{i+1})
    res_n   = np1 - n   - 0.5 * ds * (dn_i   + dn_ip1)
    res_chi = cp1 - chi - 0.5 * ds * (dchi_i + dchi_ip1)
    res_v   = vp1 - v   - 0.5 * ds * (dv_i   + dv_ip1)

    return np.concatenate([res_n, res_chi, res_v])


# ---------------------------------------------------------------------------
# Friction circle constraint (inequality: >= 0 means feasible)
# ---------------------------------------------------------------------------
def _friction(z, N, kappa):
    n, chi, v, ax = _unpack(z, N)
    scale = np.clip(1.0 - n * kappa, 0.05, None)
    ay = v**2 * kappa / scale
    utilization = ax**2 + ay**2
    return V.GRIP**2 - utilization          # >= 0


# ---------------------------------------------------------------------------
# Warm start: centerline path, speed from max cornering envelope
# ---------------------------------------------------------------------------
def _warm_start(N, kappa, n_min, n_max, ds):
    n   = np.zeros(N)                           # start on centerline
    chi = np.zeros(N)                           # no heading error
    v   = V.max_cornering_speed(kappa)          # kinematic speed limit

    # Smooth with a forward-backward pass to respect acceleration limits
    # Forward pass: can't accelerate faster than GRIP
    for i in range(1, N):
        v_fwd = np.sqrt(v[i-1]**2 + 2 * V.GRIP * ds[i-1])
        v[i] = min(v[i], v_fwd)
    # Backward pass: can't brake harder than GRIP
    for i in range(N-2, -1, -1):
        v_bwd = np.sqrt(v[i+1]**2 + 2 * V.GRIP * ds[i])
        v[i] = min(v[i], v_bwd)

    v   = np.clip(v, V.V_MIN, V.V_MAX)
    ax  = np.zeros(N)
    return _pack(n, chi, v, ax)


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------
def solve(track, verbose=True):
    """
    Solve the racing line NLP.

    Parameters
    ----------
    track : dict from track.build_track()
    verbose : bool

    Returns
    -------
    dict with keys: n, chi, v, ax, ay, s, lap_time, success
    """
    N     = len(track['s'])
    ds    = track['ds']
    kappa = track['kappa']
    n_min = track['n_min']
    n_max = track['n_max']

    if verbose:
        print(f"Setting up NLP: N={N}, variables={4*N}, "
              f"eq. constraints={3*N}, ineq. constraints={3*N}")

    # --- Warm start ---
    z0 = _warm_start(N, kappa, n_min, n_max, ds)
    t0 = _objective(z0, N, ds)
    if verbose:
        print(f"Warm-start lap time: {t0:.2f} s")

    # --- Variable bounds ---
    n_bounds   = list(zip(n_min,              n_max))
    chi_bounds = [(-np.pi/4, np.pi/4)] * N    # ±45° heading error
    v_bounds   = [(V.V_MIN, V.V_MAX)]   * N
    ax_bounds  = [(-V.GRIP, V.GRIP)]    * N
    bounds = n_bounds + chi_bounds + v_bounds + ax_bounds

    # --- Constraints ---
    constraints = [
        {
            'type': 'eq',
            'fun' : lambda z: _collocation(z, N, ds, kappa),
        },
        {
            'type': 'ineq',
            'fun' : lambda z: _friction(z, N, kappa),
        },
    ]

    # --- Solve ---
    if verbose:
        print("Running SLSQP optimizer...")

    def _callback(z):
        _, _, v, _ = _unpack(z, N)
        t = np.sum(ds / np.clip(v, V.V_MIN, None))
        print(f"  iter {_callback.count:4d} | lap time: {t:.3f} s", flush=True)
        _callback.count += 1
        _callback.count = 0

    result = minimize(
        fun     = _objective,
        x0      = z0,
        args    = (N, ds),
        jac     = _obj_grad,
        method  = 'SLSQP',
        bounds  = bounds,
        constraints = constraints,
        # callback = _callback,
        options = {
            'maxiter' : 500,
            'ftol'    : 1e-6,
            'iprint'  : 1 if verbose else 0,
            'disp'    : verbose,
        },
    )

    n, chi, v, ax = _unpack(result.x, N)

    # Compute derived quantities
    scale = np.clip(1.0 - n * kappa, 0.05, None)
    ay    = v**2 * kappa / scale
    lap_time = np.sum(ds / np.clip(v, V.V_MIN, None))

    if verbose:
        print(f"\nOptimizer status : {result.message}")
        print(f"Optimal lap time : {lap_time:.2f} s  "
              f"({lap_time/60:.0f}m {lap_time%60:.1f}s)")
        print(f"Max speed        : {v.max()*3.6:.1f} km/h")
        print(f"Max lateral G    : {np.abs(ay).max()/V.G:.2f} g")
        # print(f"Max braking G    : {np.abs(ax[ax<0]).max()/V.G:.2f} g")
        braking = ax[ax < 0]
        if len(braking) > 0:
            print(f"Max braking G    : {np.abs(braking).max()/V.G:.2f} g")
        else:
            print(f"Max braking G    : n/a (no braking nodes found)")

    return dict(
        n        = n,
        chi      = chi,
        v        = v,
        ax       = ax,
        ay       = ay,
        s        = track['s'],
        lap_time = lap_time,
        success  = result.success,
        message  = result.message,
    )


if __name__ == '__main__':
    import track as T
    tr  = T.build_track(N=50)
    sol = solve(tr)
