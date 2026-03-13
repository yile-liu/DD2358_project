import os
import sys
import csv
import time
import importlib.util

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── module loader ──────────────────────────────────────────────────────────────

def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── locate source files ────────────────────────────────────────────────────────

_HERE     = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(_HERE)

BASELINE_MOD = _load(
    os.path.join(_BASE_DIR, 'profile_code',   'DLA2DAggregate_timer.py'),     'baseline')
V1_MOD       = _load(
    os.path.join(_BASE_DIR, 'optimized_code', 'DLA2DAggregate_optimized.py'), 'v1')
V2_MOD       = _load(
    os.path.join(_BASE_DIR, 'optimized_code', 'DLA2DAggregate_optimized_v2.py'), 'v2')

V3_MOD       = _load(
    os.path.join(_BASE_DIR, 'optimized_code', 'DLA2DAggregate_optimized_v3.py'), 'v3')

# ── parameter grid ─────────────────────────────────────────────────────────────

PARTICLECAP_VALUES = [25, 50, 100, 250, 500, 750, 1000]
STEPS_VALUES       = [100, 1000]

OUTPUT_DIR = _HERE


# ── timed runners ──────────────────────────────────────────────────────────────

def _run_baseline(particleCap, steps):
    """baseline exposes run_timed() which returns a timing dict."""
    r = BASELINE_MOD.run_timed(particleCap, steps)
    return {'total': r['total_time'], 'aggr': r['while_time'], 'radial': r['for_time']}


def _run_v1(particleCap, steps):
    """
    v1 has no run_timed() wrapper, so we instrument it inline.
    The aggregation loop uses Opt-1/2/3; radial density is still the original
    Python nested loops (Student 1 did not touch that phase).
    """
    mod       = V1_MOD
    length    = mod.length
    movements = mod.movements
    radiusDensity = mod.radiusDensity

    t0 = time.time()
    lattice, stickyLattice = mod.initialise(length, movements)
    center = int(length / 2)
    occupiedPositions = {(center, center)}
    latticeRadiusData = []
    particleNumber = 1
    kill = 1;  radius = 15;  killRadius = 30;  moving = 1
    maxIdx = length

    t_a0 = time.time()
    while particleNumber < particleCap:
        if kill or not moving:
            startingPosition = mod.pick_starting_position(radius)
        moving = 1;  kill = 0
        while moving:
            path = mod.generate_path(steps, startingPosition)
            rows  = path[:, 0].astype(int)
            cols  = path[:, 1].astype(int)
            valid = (rows >= 0) & (rows <= maxIdx) & (cols >= 0) & (cols <= maxIdx)
            safeRows = np.clip(rows, 0, maxIdx)
            safeCols = np.clip(cols, 0, maxIdx)
            lc = np.zeros(steps, dtype=bool)
            lc[valid] = stickyLattice[safeRows[valid], safeCols[valid]] == 1
            hit = np.nonzero(lc)[0]
            if len(hit) > 0:
                lattice, stickyLattice = mod.particle_collision(
                    hit[0], path, lattice, stickyLattice,
                    particleNumber, occupiedPositions)
                lrMax, radius, killRadius = mod.lattice_radius_check(
                    occupiedPositions, length, radius, killRadius)
                latticeRadiusData.append([particleNumber, lrMax])
                moving = 0;  particleNumber += 1
            else:
                kill, moving, startingPosition = mod.kill_check(
                    steps, path, length, startingPosition, moving, killRadius)
    t_a1 = time.time()

    # radial density: original Python loops
    particlePositions = np.nonzero(lattice)
    particleCoords = np.zeros((particleCap, 2))
    particleCoords[:, 0] = particlePositions[0][:]
    particleCoords[:, 1] = particlePositions[1][:]
    filledDensityStore = np.zeros((10, particleCap))

    t_r0 = time.time()
    for particle in range(particleCap - 1):
        rr = particleCoords[particle, 0]
        rc_p = particleCoords[particle, 1]
        filledDensity = np.zeros(10)
        for width in range(10):
            radiusCircle = int(radiusDensity[width])
            correlationLattice = np.zeros(
                ((2 * radiusCircle) + 1, (2 * radiusCircle) + 1))
            for row in range((2 * radiusCircle) + 1):
                correlationLattice[row] = lattice[
                    int(rr + radiusCircle - row),
                    int(rc_p - radiusCircle):int(rc_p + radiusCircle + 1)]
            circleX, circleY = np.ogrid[
                -radiusCircle:radiusCircle + 1,
                -radiusCircle:radiusCircle + 1]
            mask = circleX ** 2 + circleY ** 2 <= radiusCircle ** 2
            values = correlationLattice[mask]
            filled = int(np.count_nonzero(values))
            total  = len(values)
            filledDensity[width] = filled / total if total > 0 else 0.0
        filledDensityStore[:, particle] = filledDensity
    t_r1 = time.time()

    t1 = time.time()
    return {'total': t1 - t0, 'aggr': t_a1 - t_a0, 'radial': t_r1 - t_r0}


def _run_v2(particleCap, steps):
    """v2 uses the same aggregation loop as v1 plus the vectorised radial density."""
    mod       = V2_MOD
    length    = mod.length
    movements = mod.movements

    t0 = time.time()
    lattice, stickyLattice = mod.initialise(length, movements)
    center = int(length / 2)
    occupiedPositions = {(center, center)}
    latticeRadiusData = []
    particleNumber = 1
    kill = 1;  radius = 15;  killRadius = 30;  moving = 1
    maxIdx = length

    t_a0 = time.time()
    while particleNumber < particleCap:
        if kill or not moving:
            startingPosition = mod.pick_starting_position(radius)
        moving = 1;  kill = 0
        while moving:
            path = mod.generate_path(steps, startingPosition)
            rows  = path[:, 0].astype(int)
            cols  = path[:, 1].astype(int)
            valid = (rows >= 0) & (rows <= maxIdx) & (cols >= 0) & (cols <= maxIdx)
            safeRows = np.clip(rows, 0, maxIdx)
            safeCols = np.clip(cols, 0, maxIdx)
            lc = np.zeros(steps, dtype=bool)
            lc[valid] = stickyLattice[safeRows[valid], safeCols[valid]] == 1
            hit = np.nonzero(lc)[0]
            if len(hit) > 0:
                lattice, stickyLattice = mod.particle_collision(
                    hit[0], path, lattice, stickyLattice,
                    particleNumber, occupiedPositions)
                lrMax, radius, killRadius = mod.lattice_radius_check(
                    occupiedPositions, length, radius, killRadius)
                latticeRadiusData.append([particleNumber, lrMax])
                moving = 0;  particleNumber += 1
            else:
                kill, moving, startingPosition = mod.kill_check(
                    steps, path, length, startingPosition, moving, killRadius)
    t_a1 = time.time()

    particlePositions = np.nonzero(lattice)
    particleCoords = np.zeros((particleCap, 2))
    particleCoords[:, 0] = particlePositions[0][:]
    particleCoords[:, 1] = particlePositions[1][:]

    t_r0 = time.time()
    mod._radial_density_vectorized(lattice, particleCoords, particleCap)
    t_r1 = time.time()

    t1 = time.time()
    return {'total': t1 - t0, 'aggr': t_a1 - t_a0, 'radial': t_r1 - t_r0}


def _run_v3(particleCap, steps):
    """v3 uses the same aggregation loop as v1 plus the vectorised radial density."""
    mod       = V2_MOD
    length    = mod.length
    movements = mod.movements

    t0 = time.time()
    lattice, stickyLattice = mod.initialise(length, movements)
    center = int(length / 2)
    occupiedPositions = {(center, center)}
    latticeRadiusData = []
    particleNumber = 1
    kill = 1;  radius = 15;  killRadius = 30;  moving = 1
    maxIdx = length

    t_a0 = time.time()
    while particleNumber < particleCap:
        if kill or not moving:
            startingPosition = mod.pick_starting_position(radius)
        moving = 1;  kill = 0
        while moving:
            path = mod.generate_path(steps, startingPosition)
            rows  = path[:, 0].astype(int)
            cols  = path[:, 1].astype(int)
            valid = (rows >= 0) & (rows <= maxIdx) & (cols >= 0) & (cols <= maxIdx)
            safeRows = np.clip(rows, 0, maxIdx)
            safeCols = np.clip(cols, 0, maxIdx)
            lc = np.zeros(steps, dtype=bool)
            lc[valid] = stickyLattice[safeRows[valid], safeCols[valid]] == 1
            hit = np.nonzero(lc)[0]
            if len(hit) > 0:
                lattice, stickyLattice = mod.particle_collision(
                    hit[0], path, lattice, stickyLattice,
                    particleNumber, occupiedPositions)
                lrMax, radius, killRadius = mod.lattice_radius_check(
                    occupiedPositions, length, radius, killRadius)
                latticeRadiusData.append([particleNumber, lrMax])
                moving = 0;  particleNumber += 1
            else:
                kill, moving, startingPosition = mod.kill_check(
                    steps, path, length, startingPosition, moving, killRadius)
    t_a1 = time.time()

    particlePositions = np.nonzero(lattice)
    particleCoords = np.zeros((particleCap, 2))
    particleCoords[:, 0] = particlePositions[0][:]
    particleCoords[:, 1] = particlePositions[1][:]

    t_r0 = time.time()
    mod._radial_density_vectorized(lattice, particleCoords, particleCap)
    t_r1 = time.time()

    t1 = time.time()
    return {'total': t1 - t0, 'aggr': t_a1 - t_a0, 'radial': t_r1 - t_r0}


# ── sweep ──────────────────────────────────────────────────────────────────────

def run_sweep():
    records = []
    for particleCap in PARTICLECAP_VALUES:
        for steps in STEPS_VALUES:
            print(f'\n{"="*60}')
            print(f'  particleCap={particleCap}  steps={steps}')
            print(f'{"="*60}')

            row = {'particleCap': particleCap, 'steps': steps}

            print('  [baseline] ', end='', flush=True)
            t = _run_baseline(particleCap, steps)
            row.update({k + '_baseline': v for k, v in t.items()})
            print(f"{t['total']:.2f}s  (aggr {t['aggr']:.2f}s  radial {t['radial']:.2f}s)")

            print('  [v1]       ', end='', flush=True)
            t = _run_v1(particleCap, steps)
            row.update({k + '_v1': v for k, v in t.items()})
            print(f"{t['total']:.2f}s  (aggr {t['aggr']:.2f}s  radial {t['radial']:.2f}s)")

            print('  [v2]       ', end='', flush=True)
            t = _run_v2(particleCap, steps)
            row.update({k + '_v2': v for k, v in t.items()})
            print(f"{t['total']:.2f}s  (aggr {t['aggr']:.2f}s  radial {t['radial']:.2f}s)")

            print('  [v3]       ', end='', flush=True)
            t = _run_v3(particleCap, steps)
            row.update({k + '_v3': v for k, v in t.items()})
            print(f"{t['total']:.2f}s  (aggr {t['aggr']:.2f}s  radial {t['radial']:.2f}s)")

            records.append(row)
    return records


# ── CSV ────────────────────────────────────────────────────────────────────────

def save_csv(records):
    path = os.path.join(OUTPUT_DIR, 'benchmark_v3_results.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f'\nSaved {path}')
    return path


# ── plots ──────────────────────────────────────────────────────────────────────

def _sub(records, steps):
    """Filter records to a given steps value and sort by particleCap."""
    rows = [r for r in records if r['steps'] == steps]
    return sorted(rows, key=lambda r: r['particleCap'])

def plot_speedup(records):
    """Speedup of v1, v2, and v3 over baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, steps in zip(axes, STEPS_VALUES):
        rows = _sub(records, steps)
        caps = [r['particleCap'] for r in rows]

        for label, key, marker in [
            ('v1 (aggr opt only)',         'total_v1', 'o'),
            ('v2 (+vec radial density)',   'total_v2', 's'),
            ('v3 (+GPU radial density)',   'total_v3', 'D'),
        ]:
            speedup = [r['total_baseline'] / r[key] for r in rows]
            ax.plot(caps, speedup, marker + '-', label=label)

        ax.axhline(1.0, color='grey', linestyle='--', linewidth=0.8,
                   label='baseline (1×)')
        ax.set_xlabel('particleCap')
        ax.set_ylabel('Speedup over baseline  (×)')
        ax.set_title(f'steps = {steps}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('Speedup vs Baseline', fontsize=13)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'benchmark_v3_speedup.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved {out}')
    plt.close(fig)


def plot_absolute(records):
    """Absolute wall-clock time for all versions."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, steps in zip(axes, STEPS_VALUES):
        rows = _sub(records, steps)
        caps = [r['particleCap'] for r in rows]

        for label, key, marker in [
            ('baseline',                   'total_baseline', '^'),
            ('v1 (aggr opt only)',         'total_v1',       'o'),
            ('v2 (+vec radial density)',   'total_v2',       's'),
            ('v3 (+GPU radial density)',   'total_v3',       'D'),
        ]:
            ax.plot(caps, [r[key] for r in rows], marker + '-', label=label)

        ax.set_xlabel('particleCap')
        ax.set_ylabel('Total time (s)')
        ax.set_title(f'steps = {steps}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('Absolute Runtime', fontsize=13)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'benchmark_v3_absolute.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved {out}')
    plt.close(fig)


def plot_breakdown(records):
    """
    Stacked bar chart: aggregation (dark) vs radial density (light) for each
    version at each particleCap value.  One subplot per steps value.
    """
    versions = ['baseline', 'v1', 'v2', 'v3']
    labels   = ['Baseline', 'v1 (aggr opt)', 'v2 (+vec radial)', 'v3 (+GPU radial)']
    c_aggr   = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
    c_rad    = ['#aec7e8', '#ffbb78', '#98df8a', '#c5b0d5']

    fig, axes = plt.subplots(1, 2, figsize=(17, 5))
    for ax, steps in zip(axes, STEPS_VALUES):
        rows  = _sub(records, steps)
        caps  = [r['particleCap'] for r in rows]
        n_cap = len(caps)
        n_ver = len(versions)
        width = 0.18
        x     = np.arange(n_cap)

        for i, (ver, lbl) in enumerate(zip(versions, labels)):
            offset = (i - n_ver / 2 + 0.5) * width
            agg_t  = [r[f'aggr_{ver}']   for r in rows]
            rad_t  = [r[f'radial_{ver}'] for r in rows]
            ax.bar(x + offset, agg_t, width, color=c_aggr[i],
                   label=f'{lbl} – aggr')
            ax.bar(x + offset, rad_t, width, bottom=agg_t,
                   color=c_rad[i], label=f'{lbl} – radial', alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(caps)
        ax.set_xlabel('particleCap')
        ax.set_ylabel('Time (s)')
        ax.set_title(f'steps = {steps}')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Aggregation vs Radial Density Breakdown', fontsize=13)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'benchmark_v3_breakdown.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved {out}')
    plt.close(fig)


def plot_radial_comparison(records):
    """
    Dedicated plot comparing only the radial density phase across all versions.
    Useful for isolating the GPU speedup effect.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, steps in zip(axes, STEPS_VALUES):
        rows = _sub(records, steps)
        caps = [r['particleCap'] for r in rows]

        for label, key, marker in [
            ('baseline',                   'radial_baseline', '^'),
            ('v1 (original radial)',       'radial_v1',       'o'),
            ('v2 (vec radial)',            'radial_v2',       's'),
            ('v3 (GPU radial)',            'radial_v3',       'D'),
        ]:
            ax.plot(caps, [r[key] for r in rows], marker + '-', label=label)

        ax.set_xlabel('particleCap')
        ax.set_ylabel('Radial density time (s)')
        ax.set_title(f'steps = {steps}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('Radial Density Phase — All Versions', fontsize=13)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'benchmark_v3_radial_only.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved {out}')
    plt.close(fig)


if __name__ == '__main__':
    records = run_sweep()
    save_csv(records)
    plot_speedup(records)
    plot_absolute(records)
    plot_breakdown(records)
    plot_radial_comparison(records)
    print('\nDone.')