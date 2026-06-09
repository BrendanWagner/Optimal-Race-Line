"""
plot.py
-------
Generates three figures from the racing line solution:

    1. Track map     - Silverstone outline + racing line colored by speed
    2. Speed profile - v(s) vs arc length, with braking/accel zones shaded
    3. G-force profile - ax(s) and ay(s) vs arc length, friction circle overlay
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
import vehicle as V


def _racing_line_xy(track, sol):
    """Convert (s, n) solution back to (x, y) Cartesian coordinates."""
    x_cl = track['x']
    y_cl = track['y']
    heading = track['heading']
    n = sol['n']
    # Offset centerline point by n in the normal direction
    x = x_cl + n * (-np.sin(heading))
    y = y_cl + n * (  np.cos(heading))
    return x, y


def _boundary_xy(track, side='left'):
    """Cartesian coordinates of track boundaries."""
    x_cl = track['x']
    y_cl = track['y']
    heading = track['heading']
    n_vals = track['n_max'] if side == 'left' else track['n_min']
    x = x_cl + n_vals * (-np.sin(heading))
    y = y_cl + n_vals * (  np.cos(heading))
    return x, y


# ---------------------------------------------------------------------------
# Figure 1: Track map
# ---------------------------------------------------------------------------
def plot_track_map(track, sol, ax=None, save_path=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')

    # Rotate coordinates
    theta = np.radians(90)  # adjust this angle until orientation looks right
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    xy = R @ np.array([track['x'], track['y']])
    track = {**track, 'x': xy[0], 'y': xy[1]}

    # Track boundaries
    xl, yl = _boundary_xy(track, 'left')
    xr, yr = _boundary_xy(track, 'right')
    ax.fill(np.concatenate([xl, xr[::-1]]),
            np.concatenate([yl, yr[::-1]]),
            color='#2d2d4e', zorder=1)
    ax.plot(np.append(xl, xl[0]), np.append(yl, yl[0]),
            color='#ffffff', lw=0.8, alpha=0.4, zorder=2)
    ax.plot(np.append(xr, xr[0]), np.append(yr, yr[0]),
            color='#ffffff', lw=0.8, alpha=0.4, zorder=2)

    # Centerline (dashed)
    ax.plot(np.append(track['x'], track['x'][0]),
            np.append(track['y'], track['y'][0]),
            color='#ffffff', lw=0.5, alpha=0.2, linestyle='--', zorder=3)

    # Racing line colored by speed
    rx, ry = _racing_line_xy(track, sol)
    rx = np.append(rx, rx[0])
    ry = np.append(ry, ry[0])
    v  = np.append(sol['v'], sol['v'][0])

    points  = np.array([rx, ry]).T.reshape(-1, 1, 2)
    segs    = np.concatenate([points[:-1], points[1:]], axis=1)
    norm    = mcolors.Normalize(vmin=v.min(), vmax=v.max())
    lc      = LineCollection(segs, cmap='plasma', norm=norm, zorder=4, lw=2.2)
    lc.set_array(v)
    ax.add_collection(lc)

    # Colorbar
    cbar = plt.colorbar(lc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Speed (m/s)', color='white', fontsize=9)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white', fontsize=8)

    # Add km/h secondary ticks
    v_mps_ticks = cbar.get_ticks()
    cbar.set_ticklabels([f'{v:.0f}\n({v*3.6:.0f})' for v in v_mps_ticks])

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Silverstone — Optimal Racing Line',
                 color='white', fontsize=13, pad=12)

    # Start/finish marker
    ax.plot(track['x'][0], track['y'][0], 's',
            color='#ffdd00', ms=8, zorder=5)
    ax.annotate('S/F', xy=(track['x'][0], track['y'][0]),
                xytext=(track['x'][0]+30, track['y'][0]+30),
                color='#ffdd00', fontsize=8,
                arrowprops=dict(arrowstyle='->', color='#ffdd00', lw=0.8))

    if standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
        return fig, ax


# ---------------------------------------------------------------------------
# Figure 2: Speed profile
# ---------------------------------------------------------------------------
def plot_speed_profile(track, sol, ax=None, save_path=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 4))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')

    s = sol['s'] / 1000   # km
    v = sol['v']
    ax_sol = sol['ax']

    # Shade braking zones red, acceleration zones green
    ax.fill_between(s, v, V.V_MAX,
                    where=ax_sol < -0.5, alpha=0.15, color='#ff4444',
                    label='Braking zone')
    ax.fill_between(s, v, V.V_MIN,
                    where=ax_sol >  0.5, alpha=0.15, color='#44ff88',
                    label='Acceleration zone')

    # Speed curve
    ax.plot(s, v,        color='#e040fb', lw=2,   zorder=3, label='Speed (m/s)')
    ax.plot(s, v * 3.6 / (V.V_MAX * 3.6) * V.V_MAX,
            color='#e040fb', lw=0, alpha=0)   # invisible, just for scale ref

    # Second y-axis in km/h
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim()[0] * 3.6, ax.get_ylim()[1] * 3.6)
    ax2.set_ylabel('Speed (km/h)', color='#aaaacc', fontsize=9)
    ax2.tick_params(colors='#aaaacc')
    ax2.spines['right'].set_color('#aaaacc')

    ax.set_xlabel('Arc length (km)', color='white', fontsize=10)
    ax.set_ylabel('Speed (m/s)',     color='white', fontsize=10)
    ax.set_title('Speed Profile',    color='white', fontsize=12)
    ax.set_xlim(s[0], s[-1])
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#555577')
    ax.spines['left'].set_color('#555577')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=8, facecolor='#2d2d4e', labelcolor='white',
              framealpha=0.8, loc='upper right')

    # Lap time annotation
    ax.text(0.02, 0.95,
            f"Lap time: {sol['lap_time']:.1f}s  "
            f"({int(sol['lap_time']//60)}m {sol['lap_time']%60:.1f}s)",
            transform=ax.transAxes, color='#ffdd00',
            fontsize=9, va='top')

    if standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
        return fig, ax


# ---------------------------------------------------------------------------
# Figure 3: G-force profile
# ---------------------------------------------------------------------------
def plot_gforce_profile(track, sol, ax=None, save_path=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 4))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')

    s  = sol['s'] / 1000
    ax_g = sol['ax'] / V.G
    ay_g = sol['ay'] / V.G
    total_g = np.sqrt(sol['ax']**2 + sol['ay']**2) / V.G

    ax.plot(s, ay_g,    color='#40c4ff', lw=1.5, label='Lateral G (a_y)')
    ax.plot(s, ax_g,    color='#ff6e40', lw=1.5, label='Long. G (a_x)')
    ax.plot(s, total_g, color='#ffffff', lw=1.0, alpha=0.5,
            linestyle='--', label='Total G')
    ax.axhline( V.MU,  color='#ff4444', lw=0.8, linestyle=':',
                label=f'Grip limit ±{V.MU}g')
    ax.axhline(-V.MU,  color='#ff4444', lw=0.8, linestyle=':')
    ax.axhline(0,       color='#555577', lw=0.5)

    ax.set_xlabel('Arc length (km)', color='white', fontsize=10)
    ax.set_ylabel('Acceleration (g)', color='white', fontsize=10)
    ax.set_title('G-Force Profile',   color='white', fontsize=12)
    ax.set_xlim(s[0], s[-1])
    ax.tick_params(colors='white')
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#555577')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=8, facecolor='#2d2d4e', labelcolor='white',
              framealpha=0.8, loc='upper right')

    if standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
        return fig, ax


# ---------------------------------------------------------------------------
# Combined figure (all three stacked)
# ---------------------------------------------------------------------------
def plot_all(track, sol, save_path=None):
    fig = plt.figure(figsize=(14, 14))
    fig.patch.set_facecolor('#1a1a2e')

    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    for a in [ax1, ax2, ax3]:
        a.set_facecolor('#1a1a2e')

    plot_track_map(track, sol, ax=ax1)
    plot_speed_profile(track, sol, ax=ax2)
    plot_gforce_profile(track, sol, ax=ax3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"Saved to {save_path}")
    return fig


if __name__ == '__main__':
    import track as T
    import optimize as O

    tr  = T.build_track(N=100) # Edit this for resolution
    sol = O.solve(tr, verbose=True)
    plot_all(tr, sol, save_path='C:/Users/bmwag/Documents/Optimal Race Line/results.png')
    plt.show()
