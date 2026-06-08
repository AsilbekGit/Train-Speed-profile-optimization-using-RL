"""
Plotting / logging helpers for the PPO experiment.

Charts and files produced:

  * speed_profile.png — speed (with limit overlay, low-limit zones shaded, and
    over-limit steps marked) on top, grade band, cumulative energy, then the
    action timeline — all vs DISTANCE. Tells you at a glance whether the policy
    coasts downhill, powers uphill, and respects the tight speed-limit zones.

  * speed_vs_time.png — speed + active speed limit + action strip vs TIME. Makes
    acceleration/braking phases and station dwell easy to read on the clock.

  * speed_log.csv — per-timestep table (t_s, segment, position_km, speed_ms,
    speed_kmh, speed_limit_ms, over_limit_ms, action, action_name, energy_kwh)
    so the train's current speed at every second is directly inspectable.

  * training_curves.png — return / episode length / loss from the in-memory
    logger captured during training.

`save_rollout_outputs()` writes the first three from one call so the forward and
return trainers stay in sync.
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
    # speed-limit step plot, masking station entries (limit==1)
    limit_kmh = np.where(limits <= 1.0, np.nan, limits * 3.6)
    seg_pos_km = np.arange(len(limits)) * dx / 1000.0
    # set y-range first so the zone shading spans the full panel height
    ymax = float(np.nanmax([np.nanmax(vels * 3.6), np.nanmax(limit_kmh)])) * 1.08
    ax1.set_ylim(0, ymax)
    # shade tight non-station low-limit zones — this is where braking matters,
    # and where the return trip used to fail (the seg-~718 3.0 m/s zone)
    ax1.fill_between(seg_pos_km, 0, ymax, where=(limits > 1.0) & (limits <= 5.0),
                     step='post', color='orange', alpha=0.15,
                     label='Low-limit zone (<=5 m/s)')
    ax1.step(seg_pos_km, limit_kmh, where='post', color='r', lw=1.0,
             alpha=0.7, label='Speed limit')
    ax1.plot(pos_km, vels * 3.6, 'b-', lw=1.5, label='Train speed')
    # mark steps where the train exceeds its segment's (non-station) limit
    seg_i = np.clip(segs.astype(int), 0, len(limits) - 1)
    active_lim = np.asarray(limits)[seg_i]
    viol = (active_lim > 1.0) & (vels > active_lim + 0.1)
    if np.any(viol):
        ax1.scatter(pos_km[viol], (vels * 3.6)[viol], c='red', s=14, zorder=5,
                    label=f'Over limit ({int(viol.sum())} steps)')
    for s in station_segs:
        ax1.axvline(s * dx / 1000.0, color='k', lw=0.6, alpha=0.4, ls=':')
    ax1.set_ylabel('Speed (km/h)')
    ax1.set_title(title)
    ax1.grid(alpha=0.3)
    ax1.legend(loc='upper right', fontsize=8, ncol=2)

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
            val = row.get(key, '')
            # CSV-loaded rows carry every column (blank when that stat was not
            # logged on that step); skip blanks / non-numeric cells.
            if val in ('', None):
                continue
            try:
                y = float(val)
                x = float(row['step'])
            except (ValueError, TypeError):
                continue
            xs.append(x)
            ys.append(y)
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

    # NOTE: the logger flattens keys WITHOUT a test/ train/ update/ prefix
    # (the step_type prefix only selects the bucket, it is not prepended to the
    # column names). So the keys here must be the bare flattened names that
    # actually appear in test.csv / train.csv / update.csv.
    _plot_series(axes[0, 0], logger.test_log,    'returns_stat/mean',
                 'Test return (mean)', 'tab:blue')
    _plot_series(axes[0, 1], logger.test_log,    'lens_stat/mean',
                 'Test episode length (mean)', 'tab:orange')
    _plot_series(axes[1, 0], logger.update_log,  'loss/mean',
                 'Update loss', 'tab:red')
    _plot_series(axes[1, 1], logger.training_log, 'returns_stat/mean',
                 'Train return (mean)', 'tab:green')

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


ACTION_NAMES = ['Brake', 'Coast', 'Cruise', 'Power']
ACTION_COLORS = ['#d73027', '#fc8d59', '#4575b4', '#1a9850']


def write_speed_log_csv(out_path, segs, vels, acts, ens, limits, dx=100.0, dt=1.0):
    """
    Per-timestep log of the greedy rollout so the train's current speed at every
    second is inspectable as a flat table (one row per env step / DT seconds):

        t_s, segment, position_km, speed_ms, speed_kmh,
        speed_limit_ms, over_limit_ms, action, action_name, energy_kwh
    """
    import csv
    segs = np.asarray(segs); vels = np.asarray(vels)
    acts = np.asarray(acts); ens = np.asarray(ens)
    limits = np.asarray(limits)
    n_lim = len(limits)
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t_s', 'segment', 'position_km', 'speed_ms', 'speed_kmh',
                    'speed_limit_ms', 'over_limit_ms', 'action', 'action_name',
                    'energy_kwh'])
        for i in range(len(vels)):
            s = int(segs[i]) if i < len(segs) else 0
            s = max(0, min(s, n_lim - 1))
            v = float(vels[i])
            lim = float(limits[s])
            over = max(0.0, v - lim) if lim > 1.0 else 0.0
            a = int(acts[i]) if i < len(acts) else 0
            w.writerow([f'{i * dt:.1f}', s, f'{s * dx / 1000.0:.3f}',
                        f'{v:.3f}', f'{v * 3.6:.2f}',
                        f'{lim:.2f}', f'{over:.2f}',
                        a, ACTION_NAMES[a], f'{float(ens[i]):.4f}'])


def plot_speed_vs_time(out_path, segs, vels, acts, limits, title, dt=1.0):
    """
    Speed vs TIME (one point per env step). The position-based speed_profile.png
    shows speed over distance; this shows it over the clock, which makes
    acceleration/braking phases and dwell time at stations easy to read. The
    active speed limit (the limit of whatever segment the train is on at that
    instant) is overlaid, and limit violations are marked in red.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    segs = np.asarray(segs); vels = np.asarray(vels); acts = np.asarray(acts)
    limits = np.asarray(limits)
    n_lim = len(limits)
    t = np.arange(len(vels)) * dt
    seg_i = np.clip(segs.astype(int), 0, n_lim - 1)
    seg_lim = limits[seg_i]
    active_limit_kmh = np.where(seg_lim <= 1.0, np.nan, seg_lim * 3.6)

    fig, (ax, axa) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]},
    )

    ax.plot(t, vels * 3.6, 'b-', lw=1.4, label='Train speed')
    ax.step(t, active_limit_kmh, where='post', color='r', lw=1.0, alpha=0.7,
            label='Active speed limit')
    viol = (seg_lim > 1.0) & (vels > seg_lim + 0.1)
    if np.any(viol):
        ax.scatter(t[viol], (vels * 3.6)[viol], c='red', s=12, zorder=5,
                   label=f'Over limit ({int(viol.sum())} steps)')
    ax.set_ylabel('Speed (km/h)')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)

    for i in range(4):
        m = acts == i
        if m.any():
            axa.scatter(t[m], np.full(m.sum(), i), c=ACTION_COLORS[i], s=4,
                        label=ACTION_NAMES[i], alpha=0.7)
    axa.set_yticks([0, 1, 2, 3])
    axa.set_yticklabels(ACTION_NAMES)
    axa.set_xlabel('Time (s)')
    axa.set_ylabel('Action')
    axa.grid(alpha=0.3)
    axa.legend(loc='upper right', fontsize=9, markerscale=3, ncol=4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def save_rollout_outputs(out_dir, segs, vels, acts, ens, grades, limits,
                         title, dx=100.0, dt=1.0):
    """
    Single entry point for all greedy-rollout artifacts, shared by the forward
    (ppo/train.py) and return (return_trip/train_ppo.py) trainers so they stay
    in sync:
        - speed_profile.npz    raw arrays
        - speed_log.csv        per-timestep speed table (NEW)
        - speed_profile.png    speed/grade/energy/action vs DISTANCE
        - speed_vs_time.png    speed + action vs TIME (NEW)
    """
    segs = np.asarray(segs); vels = np.asarray(vels)
    acts = np.asarray(acts); ens = np.asarray(ens)
    grades = np.asarray(grades); limits = np.asarray(limits)

    np.savez(os.path.join(out_dir, 'speed_profile.npz'),
             segments=segs, velocities=vels, actions=acts, energies=ens)
    write_speed_log_csv(os.path.join(out_dir, 'speed_log.csv'),
                        segs, vels, acts, ens, limits, dx=dx, dt=dt)
    station_segs = np.where(limits <= 1.0)[0]
    plot_speed_profile(os.path.join(out_dir, 'speed_profile.png'),
                       segs, vels, acts, ens, grades, limits, station_segs,
                       title=title, dx=dx)
    plot_speed_vs_time(os.path.join(out_dir, 'speed_vs_time.png'),
                       segs, vels, acts, limits, title=title, dt=dt)
