"""
Plotting helpers for the new PPO experiment.

Two charts:

  * speed_profile.png — speed (with limit overlay) on top, grade band in the
    middle, action timeline at the bottom; station markers as vertical lines.
    Lets you tell at a glance whether the policy is coasting downhill and
    powering uphill, or doing something silly.

  * training_curves.png — return / episode length / energy from the in-memory
    logger captured during training.
"""

import os
import numpy as np


def plot_speed_profile(
    out_path,
    segs, vels, acts, ens,
    grades, limits, station_segs,
    title,
    dx=100.0,
):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    segs = np.asarray(segs)
    vels = np.asarray(vels)
    acts = np.asarray(acts)
    ens = np.asarray(ens)
    pos_km = segs * dx / 1000.0

    fig, axes = plt.subplots(
        4, 1, figsize=(14, 12),
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1.5, 1.5, 1.2]},
    )

    # ---- 1. speed + limit overlay ----
    ax1 = axes[0]
    ax1.plot(pos_km, vels * 3.6, 'b-', lw=1.5, label='Train speed')
    # speed-limit step plot, masking station entries (limit==1)
    limit_kmh = np.where(limits <= 1.0, np.nan, limits * 3.6)
    seg_pos_km = np.arange(len(limits)) * dx / 1000.0
    ax1.step(seg_pos_km, limit_kmh, where='post', color='r', lw=1.0,
             alpha=0.7, label='Speed limit')
    for s in station_segs:
        ax1.axvline(s * dx / 1000.0, color='k', lw=0.6, alpha=0.4, ls=':')
    ax1.set_ylabel('Speed (km/h)')
    ax1.set_title(title)
    ax1.grid(alpha=0.3)
    ax1.legend(loc='upper right', fontsize=9)

    # ---- 2. grade band ----
    ax2 = axes[1]
    ax2.fill_between(seg_pos_km, 0, grades, where=grades >= 0,
                     color='tab:red', alpha=0.4, label='Uphill')
    ax2.fill_between(seg_pos_km, 0, grades, where=grades < 0,
                     color='tab:green', alpha=0.4, label='Downhill')
    ax2.axhline(0, color='k', lw=0.5)
    ax2.set_ylabel('Grade (%)')
    ax2.grid(alpha=0.3)
    ax2.legend(loc='upper right', fontsize=9)

    # ---- 3. cumulative energy ----
    ax3 = axes[2]
    ax3.plot(pos_km, ens, 'r-', lw=1.5)
    ax3.set_ylabel('Energy (kWh)')
    ax3.grid(alpha=0.3)

    # ---- 4. action strip ----
    ax4 = axes[3]
    colors = ['#d73027', '#fc8d59', '#4575b4', '#1a9850']  # Brake, Coast, Cruise, Power
    names = ['Brake', 'Coast', 'Cruise', 'Power']
    for i in range(4):
        m = acts == i
        if m.any():
            ax4.scatter(pos_km[m], np.full(m.sum(), i), c=colors[i],
                        s=4, label=names[i], alpha=0.7)
    ax4.set_yticks([0, 1, 2, 3])
    ax4.set_yticklabels(names)
    ax4.set_xlabel('Position (km)')
    ax4.set_ylabel('Action')
    ax4.legend(loc='upper right', fontsize=9, markerscale=3, ncol=4)
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_training_curves(out_path, logger):
    """
    Reads InMemoryLogger.test_log / update_log / training_log and produces
    a multi-panel chart of return, episode length, and policy/value losses.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    def _plot_series(ax, log, key, title, color):
        if not log:
            ax.set_title(f'{title}  (no data)')
            return
        xs, ys = [], []
        for row in log:
            if key in row:
                xs.append(row['step'])
                ys.append(float(row[key]))
        if not xs:
            ax.set_title(f'{title}  (key "{key}" missing)')
            return
        ax.plot(xs, ys, color=color, lw=1.4)
        if len(ys) > 25:
            w = max(1, len(ys) // 25)
            sm = np.convolve(ys, np.ones(w) / w, mode='valid')
            ax.plot(xs[w - 1:], sm, color=color, lw=2.4, alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel('env step')
        ax.grid(alpha=0.3)

    _plot_series(axes[0, 0], logger.test_log,    'test/returns_stat/mean',
                 'Test return (mean)', 'tab:blue')
    _plot_series(axes[0, 1], logger.test_log,    'test/lens_stat/mean',
                 'Test episode length (mean)', 'tab:orange')
    _plot_series(axes[1, 0], logger.update_log,  'update/loss',
                 'Update loss', 'tab:red')
    _plot_series(axes[1, 1], logger.training_log, 'train/returns_stat/mean',
                 'Train return (mean)', 'tab:green')

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
