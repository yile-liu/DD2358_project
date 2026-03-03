"""
Profiler script for DLA2DAggregate_modified.py
Runs different combinations of particleCap and steps,
collects: total time, per-function time, top 20 hottest lines.
"""
import time
import io
import re
import csv
from line_profiler import LineProfiler

# Import target functions from the modified script
from DLA2DAggregate_line_profiler import (
    main, initialise, pick_starting_position, generate_path,
    particle_collision, lattice_radius_check, kill_check,
)

# Parameter grid
particleCap_values = [25, 50, 100, 250, 500, 750, 1000]
steps_values = [100, 1000]

# All functions to profile
PROFILED_FUNCS = [
    main, initialise, pick_starting_position, generate_path,
    particle_collision, lattice_radius_check, kill_check,
]


def run_profiled(particleCap, steps):
    """Run main() under LineProfiler and return (total_time, profiler)."""
    lp = LineProfiler()
    for func in PROFILED_FUNCS:
        lp.add_function(func)
    wrapped = lp(main)

    t0 = time.time()
    wrapped(particleCap, steps)
    total_time = time.time() - t0

    return total_time, lp


def extract_function_times(lp):
    """Extract per-function total time from profiler stats."""
    func_times = {}
    for (filename, lineno, fname), timings in lp.get_stats().timings.items():
        if timings:
            # timings: list of (lineno, nhits, time_us)
            total_us = sum(t[2] for t in timings)
            func_times[fname] = total_us * 1e-6  # convert to seconds
    return func_times


def extract_top_lines(lp, top_n=20):
    """Extract top N most time-consuming lines across all functions."""
    lines = []
    for (filename, start_lineno, fname), timings in lp.get_stats().timings.items():
        for lineno, nhits, time_us in timings:
            if time_us > 0:
                lines.append({
                    'function': fname,
                    'file': filename,
                    'lineno': lineno,
                    'hits': nhits,
                    'time_s': time_us * 1e-6,
                })
    lines.sort(key=lambda x: x['time_s'], reverse=True)
    return lines[:top_n]


def read_source_line(filepath, lineno):
    """Read a single source line from file."""
    try:
        with open(filepath, 'r') as f:
            for i, line in enumerate(f, 1):
                if i == lineno:
                    return line.rstrip()
    except Exception:
        pass
    return '<unavailable>'


def main_profiler():
    results = []

    for particleCap in particleCap_values:
        for steps in steps_values:
            print('=' * 70)
            print(f'Running: particleCap={particleCap}, steps={steps}')
            print('=' * 70)

            total_time, lp = run_profiled(particleCap, steps)
            func_times = extract_function_times(lp)
            top_lines = extract_top_lines(lp, top_n=20)

            # Print summary
            print(f'\nTotal time: {total_time:.2f}s')
            print(f'\n--- Per-function time ---')
            for fname, ftime in sorted(func_times.items(), key=lambda x: x[1], reverse=True):
                pct = ftime / total_time * 100 if total_time > 0 else 0
                print(f'  {fname:30s}  {ftime:10.4f}s  ({pct:5.1f}%)')

            print(f'\n--- Top 20 lines by time ---')
            print(f'  {"Rank":>4s}  {"Function":25s}  {"Line":>5s}  {"Hits":>10s}  {"Time(s)":>10s}  Code')
            for i, entry in enumerate(top_lines, 1):
                src = read_source_line(entry['file'], entry['lineno'])
                print(f"  {i:4d}  {entry['function']:25s}  {entry['lineno']:5d}  {entry['hits']:10d}  {entry['time_s']:10.4f}  {src}")

            results.append({
                'particleCap': particleCap,
                'steps': steps,
                'total_time': total_time,
                'func_times': func_times,
                'top_lines': top_lines,
            })
            print()

    # Write CSV summary
    csv_path = 'profile_results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header: particleCap, steps, total_time, then one column per function
        all_funcs = sorted({fn for r in results for fn in r['func_times']})
        header = ['particleCap', 'steps', 'total_time'] + [f'time_{fn}' for fn in all_funcs]
        writer.writerow(header)
        for r in results:
            row = [r['particleCap'], r['steps'], f"{r['total_time']:.4f}"]
            for fn in all_funcs:
                row.append(f"{r['func_times'].get(fn, 0):.4f}")
            writer.writerow(row)
    print(f'CSV summary saved to {csv_path}')

    # Write top-lines detail CSV
    lines_csv_path = 'profile_top_lines.csv'
    with open(lines_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['particleCap', 'steps', 'rank', 'function', 'lineno', 'hits', 'time_s', 'code'])
        for r in results:
            for i, entry in enumerate(r['top_lines'], 1):
                src = read_source_line(entry['file'], entry['lineno'])
                writer.writerow([
                    r['particleCap'], r['steps'], i,
                    entry['function'], entry['lineno'], entry['hits'],
                    f"{entry['time_s']:.4f}", src,
                ])
    print(f'Top-lines detail saved to {lines_csv_path}')


if __name__ == '__main__':
    main_profiler()
