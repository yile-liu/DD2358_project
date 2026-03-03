"""
Timer runner for DLA 2D Aggregate.
Runs different combinations of particleCap and steps,
collects: total time, while-loop time, for-loop time, and their percentages.
Saves CSV and generates plots.
"""
import csv
import os

import matplotlib.pyplot as plt
import pandas as pd

from DLA2DAggregate_timer import run_timed

# Parameter grid
particleCap_values = [25, 50, 100, 250, 500, 750, 1000]
steps_values = [100, 1000]

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(OUTPUT_DIR, 'timer_results.csv')


def run_sweep():
    """Run parameter sweep and return list of result dicts."""
    results = []

    for particleCap in particleCap_values:
        for steps in steps_values:
            print('=' * 70)
            print(f'Running: particleCap={particleCap}, steps={steps}')
            print('=' * 70)

            timing = run_timed(particleCap, steps)

            print(f'\n--- Timing Summary ---')
            print(f'While loop (particle aggregation): {timing["while_time"]:.2f}s ({timing["while_pct"]:.1f}%)')
            print(f'For loop (radial distribution):    {timing["for_time"]:.2f}s ({timing["for_pct"]:.1f}%)')
            print(f'Total time:                        {timing["total_time"]:.2f}s')

            results.append({
                'particleCap': particleCap,
                'steps': steps,
                **timing,
            })
            print()

    return results


def save_csv(results):
    """Write results to CSV."""
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['particleCap', 'steps', 'total_time', 'while_time', 'while_pct', 'for_time', 'for_pct']
        writer.writerow(header)
        for r in results:
            writer.writerow([
                r['particleCap'], r['steps'],
                f"{r['total_time']:.4f}",
                f"{r['while_time']:.4f}", f"{r['while_pct']:.1f}",
                f"{r['for_time']:.4f}", f"{r['for_pct']:.1f}",
            ])
    print(f'CSV summary saved to {CSV_PATH}')


def plot_absolute_times(df):
    """Plot while-loop time, for-loop time, and total time vs particleCap."""
    steps_vals = sorted(df['steps'].unique())
    fig, axes = plt.subplots(1, len(steps_vals), figsize=(7 * len(steps_vals), 5),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax, steps in zip(axes, steps_vals):
        sub = df[df['steps'] == steps].sort_values('particleCap')
        x = sub['particleCap']

        ax.plot(x, sub['while_time'], 'o-', label='While loop (aggregation)')
        ax.plot(x, sub['for_time'],   's-', label='For loop (radial density)')
        ax.plot(x, sub['total_time'], '^-', label='Total')

        ax.set_xlabel('particleCap')
        ax.set_ylabel('Time (s)')
        ax.set_title(f'steps = {steps}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('Execution Time vs particleCap', fontsize=14, y=1.02)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'timer_absolute.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved {out}')


def plot_proportions(df):
    """Plot while-loop % and for-loop % of total time vs particleCap."""
    steps_vals = sorted(df['steps'].unique())
    fig, axes = plt.subplots(1, len(steps_vals), figsize=(7 * len(steps_vals), 5),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax, steps in zip(axes, steps_vals):
        sub = df[df['steps'] == steps].sort_values('particleCap')
        x = sub['particleCap']

        ax.plot(x, sub['while_pct'], 'o-', label='While loop %')
        ax.plot(x, sub['for_pct'],   's-', label='For loop %')

        ax.set_xlabel('particleCap')
        ax.set_ylabel('Proportion of total time (%)')
        ax.set_title(f'steps = {steps}')
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('Time Proportion vs particleCap', fontsize=14, y=1.02)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'timer_proportions.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved {out}')


def main_timer():
    results = run_sweep()
    save_csv(results)

    df = pd.read_csv(CSV_PATH)
    plot_absolute_times(df)
    plot_proportions(df)
    plt.show()


if __name__ == '__main__':
    main_timer()
