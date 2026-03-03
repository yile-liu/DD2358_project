# Performance Optimization of DLA-python

## Introduction

### What is Diffusion Limited Aggregation?

Diffusion Limited Aggregation (DLA) is a growth model in which particles perform random walks and attach to a growing cluster upon contact, forming branching fractal structures. The simulation proceeds as follows. A seed particle is placed at the center of a lattice. New particles are released one at a time. Each particle performs a random walk until it either reaches a cluster and becomes attached, or move beyond the border and is discarded. This process repeats until the cluster reaches a target size. Theoretically, in two dimensions, the resulting aggregates have a fractal dimension of approximately 1.71.

### The DLA-python Repository

The repository [DLA-python](https://github.com/georich/DLA-python), authored by George Richards, provides a Python/NumPy implementation of DLA simulations. Among all the files, `DLA2DAggregate.py` is the most complete implementation, containing the full simulation loop, fractal dimension calculation, and radial density analysis. The remaining files are variants that modify the seed geometry, boundary conditions, or introduce additional features such as animation and sticking probability. However, they all share the same core algorithmic structure: random walk generation, collision detection, sticky-site rebuilding after each particle attachment, and same post-simulation analysis.

For this reason, we focus our profiling and optimization exclusively on `DLA2DAggregate.py`. The optimization strategies developed here are directly applicable to all other files in the repository.

## Baseline Code Analysis and Profiling

### Code Structure

The code consists of two phases. 

The **aggregation phase** grows the cluster using two 1251 x 1251 arrays: `lattice` (storing deposited particles) and `stickyLattice` (marking sites adjacent to the cluster). For each new particle, the code randomly decides a starting position on a circle around the cluster, generates a random path, and checks collision against `stickyLattice` step by step on the path. After each collision and attachment, `particle_collision` rebuilds the entire `stickyLattice` by scanning all occupied sites via `np.nonzero(lattice)`, and `lattice_radius_check` performs a third `np.nonzero` scan to update the maximum radius.

The **radial density phase** computes C(r) after the aggregate is complete. For each particle, for each of 10 radius values, it extracts a sub-lattice, applies a circular mask, and counts filled/empty sites.

Two key parameters control the simulation: `particleCap` defines the target number of particles in the aggregate, and `steps` defines the maximum number of random walk steps per path segment. If a particle does not collide or leave the boundary within `steps` steps, it continues walking from its last position with a new path segment.

### Profiling

We used two profiling approaches, setting `particleCap` over {25, 50, 100, 250, 500, 750, 1000} and `steps` over {100, 1000}.:

- **Timer profiling**: We separately measure the execution time of the aggregation and the radial density computation.

- **Line profiling**: Using Python's `line_profiler`, we decorated every function with `@profile` and recorded per-line hit counts and execution times.

#### Coarse-Grained Timing

<image src="./profile_code/timer_absolute.png">

| particleCap | steps | Total (s) | Aggregation (s) | Aggr. % | Radial Density (s) | R.D. % |
|-------------|-------|-----------|-----------------|---------|---------------------|--------|
| 25          | 100   | 1.62      | 0.42            | 25.9    | 1.19                | 73.3   |
| 25          | 1000  | 1.65      | 0.41            | 24.5    | 1.24                | 74.8   |
| 100         | 100   | 6.64      | 1.52            | 22.9    | 5.11                | 76.9   |
| 100         | 1000  | 6.93      | 1.77            | 25.6    | 5.15                | 74.2   |
| 250         | 100   | 16.73     | 4.01            | 24.0    | 12.71               | 75.9   |
| 250         | 1000  | 16.72     | 4.23            | 25.3    | 12.47               | 74.6   |
| 500         | 100   | 33.29     | 8.35            | 25.1    | 24.92               | 74.9   |
| 500         | 1000  | 35.30     | 9.70            | 27.5    | 25.59               | 72.5   |
| 750         | 100   | 58.76     | 14.60           | 24.8    | 44.15               | 75.1   |
| 750         | 1000  | 62.09     | 16.08           | 25.9    | 46.00               | 74.1   |
| 1000        | 100   | 80.70     | 22.64           | 28.0    | 58.05               | 71.9   |
| 1000        | 1000  | 85.06     | 22.68           | 26.7    | 62.37               | 73.3   |

The radial density computation takes 72–77% of total runtime, while the aggregation takes 23–28%.



<image src="./profile_code/timer_proportions.png">


Both two parts grow roughly linearly with `particleCap`, but not perfectly. The complexity of radial density computation part is linear because it iterates over all particles with fixed per-particle work. The aggregation art grows slightly faster than linear: as the cluster grows, the generation radius increases, particles must walk longer distances to reach the cluster, and the `particle_collision` function scans more occupied sites per call (O(N) per particle, O(N^2) cumulative). This explains why the aggregation percentage rises slightly from ~23% to ~28% as `particleCap` increases.

The `steps` parameter has minimal impact on total runtime: for example, at `particleCap=1000`, increasing `steps` from 100 to 1000 only changes the total time from 80.70s to 85.06s. This is because `steps` only controls the length of each path segment, not the number of steps a particle takes before collision or removal. A smaller `steps` causes more internal loop iterations (more path segments with less steps in each), while a larger `steps` wastes computation on unused path tails. In either case, the actual work per particle remains.

#### Line Profiling Results

Representative results for `particleCap=25, steps=100` (wall-clock time: 6.84s):

**Per-function breakdown:**

| Function                  | Time (s) | Description |
|---------------------------|----------|-------------|
| `main` (radial density)   | 49.85    | Radial density nested loops |
| `particle_collision`      | 2.15     | Sticky-site full rebuild |
| `lattice_radius_check`    | 1.05     | Max radius computation |
| `generate_path`           | 0.44     | Random walk generation |
| `initialise`              | 0.06     | Lattice setup |
| `kill_check`              | 0.03     | Out-of-bounds check |
| `pick_starting_position`  | 0.01     | Starting point generation |

Note: `line_profiler` reports cumulative per-line time, so the `main` total exceeds wall-clock time due to profiling overhead in the deeply nested loops.

**Top hotspot lines — radial density calculation:**

| Rank | Line | Hits      | Time (s) | Code |
|------|------|-----------|----------|------|
| 1    | 190  | 2,161,392 | 8.42     | `coords[n,0], coords[n,1] = coordsToCheck[0][n], coordsToCheck[1][n]` |
| 2    | 195  | 2,161,392 | 6.77     | `row = int(coords[n,0])` |
| 3    | 196  | 2,161,392 | 6.62     | `column = int(coords[n,1])` |
| 4    | 197  | 2,161,392 | 5.39     | `check = correlationLattice[row, column]` |
| 5    | 198  | 2,161,392 | 5.35     | `if check == 0:` |
| 6    | 199  | 2,155,837 | 5.00     | `empty += 1` |
| 7    | 194  | 2,161,632 | 3.67     | `for n in range(len(coords)):` |
| 8    | 189  | 2,161,632 | 3.45     | `for n in range(len(coordsToCheck[0])):` |

These 8 lines account for 44.7s out of 49.9s (89.6%) of the radial density section. Even with only 25 particles, the innermost loops execute over 2 million iterations.

**Top hotspot lines — particle aggregation:**

| Rank | Line | Hits   | Time (s) | Code |
|------|------|--------|----------|------|
| 9    | 144  | 24     | 2.16     | `particle_collision(...)` call |
| 10   | 58   | 24     | 1.07     | `np.nonzero(lattice)` in `particle_collision` |
| 11   | 68   | 24     | 1.05     | `np.nonzero(lattice)` in `particle_collision` (2nd call) |
| 13   | 78   | 24     | 1.05     | `np.nonzero(lattice)` in `lattice_radius_check` |
| 14   | 130  | 345    | 0.56     | `generate_path(...)` call |
| 17   | 135  | 33,518 | 0.23     | Collision check: `stickyLattice[int(path[n,0]), ...]` |

The dominant cost in aggregation is `particle_collision`, which calls `np.nonzero` on the full 1251 x 1251 lattice three times per particle (twice in `particle_collision`, once in `lattice_radius_check`).

