# Performance Optimization of DLA-python

## Introduction

### What is Diffusion Limited Aggregation?

Diffusion Limited Aggregation (DLA) is a growth model in which particles perform random walks and attach to a growing cluster upon contact, forming branching fractal structures. The simulation proceeds as follows. A seed particle is placed at the center of a lattice. New particles are released one at a time. Each particle performs a random walk until it either reaches the cluster and becomes attached, or moves beyond the border and is discarded. This process repeats until the cluster reaches a target size. Theoretically, in two dimensions, the resulting aggregates have a fractal dimension of approximately 1.71.

### The DLA-python Repository

The repository [DLA-python](https://github.com/georich/DLA-python), authored by George Richards, provides a Python/NumPy implementation of DLA simulations. Among all the files, `DLA2DAggregate.py` is the most complete implementation, containing the full particle aggregation simulation and radical density calculation. The remaining files are variants that modify the seed geometry, boundary conditions, or introduce additional features such as animation and sticking probability. They all share the same core algorithmic structure: random walk generation, collision detection, sticky-site rebuilding after each particle attachment, and the same radical density calculation.

For this reason, we focus our profiling and optimization exclusively on `DLA2DAggregate.py`. The optimization strategies developed here are directly applicable to all other files in the repository.

## Baseline Code Analysis and Profiling

### Code Structure

The code consists of two phases. 

The **aggregation phase** grows the cluster using two 1251 x 1251 arrays: `lattice` (storing deposited particles) and `stickyLattice` (marking sites adjacent to the cluster). For each new particle, the code picks a random starting position on a circle around the cluster, generates a random walk path, and checks for collision against `stickyLattice` step by step. After each collision and attachment, `particle_collision` rebuilds the entire `stickyLattice` by scanning all occupied sites via `np.nonzero(lattice)`, and `lattice_radius_check` performs a third `np.nonzero` scan to update the maximum radius.

The **radial density phase** computes C(r) after the aggregate is complete. For each particle, for each of 10 radius values, it extracts a sub-lattice, applies a circular mask, and counts filled/empty sites.

Two key parameters control the simulation: `particleCap` defines the target number of particles in the aggregate, and `steps` defines the number of random steps per loop. If a particle does not collide or leave the boundary within `steps` steps, it continues walking from its last position with a new path segment.

### Profiling

We used two profiling approaches, setting `particleCap` over {25, 50, 100, 250, 500, 750, 1000} and `steps` over {100, 1000}:

- **Timer profiling**: We separately measured the execution time of the aggregation phase and the radial density phase.

- **Line profiling**: Using Python's `line_profiler`, we decorated every function with `@profile` and recorded per-line hit counts and execution times.

#### Time Profiling

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

The radial density phase takes 72–77% of total runtime, while the aggregation phase takes 23–28%.

<image src="./profile_code/timer_proportions.png">

Both phases grow roughly linearly with `particleCap`, but not perfectly. The radial density phase is linear because it iterates over all particles with fixed work per particle. The aggregation phase grows slightly faster than linear: as the cluster grows, the generation radius increases, particles must walk longer to reach the cluster, and `particle_collision` scans more occupied sites per call (O(N) per particle, O(N^2) total). This explains why the aggregation percentage rises slightly from ~23% to ~28% as `particleCap` increases.

The `steps` parameter has minimal impact on total runtime: for example, at `particleCap=1000`, increasing `steps` from 100 to 1000 only changes the total time from 80.70s to 85.06s. This is because `steps` only controls the number of steps generated at a time, not the actual number of steps a particle takes before collision or removal. A smaller `steps` causes more loop iterations (each with fewer steps), while a larger `steps` wastes computation on unused path tails. In either case, the actual work per particle stays about the same.

#### Line Profiling

Representative results for `particleCap=25, steps=100` (wall-clock time: 6.84s):

**Per-function breakdown:**

| Function                  | Time (s) | Description |
|---------------------------|----------|-------------|
| `main` (radial density)   | 49.85    | Radial density phase nested loops |
| `particle_collision`      | 2.15     | Sticky-site full rebuild |
| `lattice_radius_check`    | 1.05     | Max radius computation |
| `generate_path`           | 0.44     | Random walk generation |
| `initialise`              | 0.06     | Lattice setup |
| `kill_check`              | 0.03     | Out-of-bounds check |
| `pick_starting_position`  | 0.01     | Starting point generation |

Note: the `main` total exceeds wall-clock time because of profiling overhead.

**Top hotspot lines — radial density phase:**

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

These 8 lines account for 44.7s out of 49.9s (89.6%) of the radial density phase. Even with only 25 particles, the innermost loops run over 2 million iterations.

**Top hotspot lines — aggregation phase:**

| Rank | Function | Line | Hits | Time (s) | Code |
|------|----------|------|------|----------|------|
| 9  | `main`              | 144 | 24     | 2.16 | `particle_collision(...)` call |
| 10 | `particle_collision`| 58  | 24     | 1.07 | `np.nonzero(lattice)` |
| 11 | `main`              | 145 | 24     | 1.05 | `lattice_radius_check(...)` call |
| 12 | `particle_collision`| 68  | 24     | 1.05 | `np.nonzero(lattice)` (2nd call) |
| 13 | `lattice_radius_check` | 78 | 24 | 1.05 | `lattice_radius_check` call |
| 14 | `main`              | 130 | 345    | 0.56 | `generate_path(...)` call |
| 15 | `main`              | 184 | 22,128 | 0.31 | `correlationLattice[row] = lattice[...]` |
| 16 | `generate_path`     | 47  | 34,155 | 0.28 | `path[n+1] = particlePosition + movements[direction[n]]` |
| 17 | `main`              | 135 | 33,518 | 0.23 | `stickyLattice[int(path[n,0]), ...]` collision check |
| 20 | `main`              | 141 | 33,518 | 0.09 | `if logicCheck[n] == 1:` |

Apart from the radial density phase, the most expensive part is the **sticky-site rebuild** (`particle_collision` + `lattice_radius_check`). These functions are called only 24 times (once per deposited particle), yet they take over 4 s combined. The reason is that each call runs `np.nonzero` on the full 1251 × 1251 lattice: twice inside `particle_collision` (lines 58, 68) and once inside `lattice_radius_check` (line 78), scanning over 1.5 million elements each time.

Another bottleneck is the **path generation** and **collision detection** loops. Both use Python-level `for` loops with no vectorization:

```python
# generate_path (line 47): one step at a time
for n in range(steps - 1):
    path[n+1] = particlePosition + movements[direction[n]]

# collision check (line 135): one element at a time
for n in range(steps):
    try:
        logicCheck[n] = stickyLattice[int(path[n,0]), int(path[n,1])] == 1
    except IndexError:
        continue
    ...
    if logicChech[n] == 1:
    ...
```

Both loops access array elements one by one in Python instead of using vectorized NumPy operations. The collision detection loop also wraps every access in a `try/except IndexError` block, which adds extra overhead. Line 135 is hit 33,518 times even in this small 25-particle run.

## Optimization Methodology

### Particle Aggregation

Based on the profiling results, we found three main bottlenecks in the aggregation phase and applied a different optimization for each.

#### Incremental Position Tracking

In the original code, every time a particle is deposited, `particle_collision` calls `np.nonzero(lattice)` twice and `lattice_radius_check` calls it once more to find deposited particle and rebuild aggregation from scratch. Each `np.nonzero` goes through the whole 1251 × 1251 lattice to find occupied sites. For 25 particles, these three scans already take over 4 seconds combined.

But we only add one new particle each time, so scanning the entire lattice is unnecessary. Instead, we keep a set that stores all deposited particle coordinates. When a new particle is deposited, we just add its position to the set and update its 4 neighbors in `stickyLattice`. This changes the cost from O(N_lattice) to O(1) per particle.

**Before:**
```python
def particle_collision(logicCheck, path, lattice, stickyLattice, particleNumber):
    logicCheck2 = np.nonzero(logicCheck)
    collision = path[logicCheck2[0]]
    lattice[int(collision[0,0]), int(collision[0,1])] = particleNumber + 1
    stickyInitialTest = np.nonzero(lattice)          # scan whole lattice
    stickySitesTest = []
    for n in range(4):
        toDo = len(stickyInitialTest[0])
        for o in range(toDo):
            stickySitesTest.append(...)              # rebuild ALL sticky sites
    for n in range(4*toDo):
        stickyLattice[stickySitesTest[n][0], stickySitesTest[n][1]] = 1
    latticeCheck = np.nonzero(lattice)               # scan whole lattice again
    for n in range(latticeCheckLength):
        stickyLattice[latticeCheck[0][n], latticeCheck[1][n]] = 0
```

**After:**
```python
def particle_collision(collisionIdx, path, lattice, stickyLattice,
                       particleNumber, occupiedPositions):
    cr, cc = int(path[collisionIdx, 0]), int(path[collisionIdx, 1])
    lattice[cr, cc] = particleNumber + 1
    occupiedPositions.add((cr, cc))                  # O(1) set insertion
    for dr, dc in movements:                         # only check 4 neighbors
        nr, nc = cr + dr, cc + dc
        if 0 <= nr <= maxIdx and 0 <= nc <= maxIdx \
           and (nr, nc) not in occupiedPositions:
            stickyLattice[nr, nc] = 1
    stickyLattice[cr, cc] = 0
```

We apply the same idea to `lattice_radius_check`. Instead of calling `np.nonzero(lattice)` to find all particles, we directly use the `occupiedPositions` set:

```python
def lattice_radius_check(occupiedPositions, length, radius, killRadius):
    center = length / 2
    positions = np.array(list(occupiedPositions))
    distances = np.sqrt((positions[:, 0] - center)**2
                      + (positions[:, 1] - center)**2)
    latticeRadiusMax = distances.max()
    if latticeRadiusMax > radius:
        radius = latticeRadiusMax * 1.5
        killRadius = latticeRadiusMax * 2
    return latticeRadiusMax, radius, killRadius
```

#### Vectorized Path Generation

The original `generate_path` builds the random walk path one step at a time in a Python `for` loop (34,155 iterations for just 25 particles). Since each step is an independent random direction, the whole path is just a cumulative sum of step vectors. We can replace the loop with a single `np.cumsum` call, which lets NumPy do the work in C instead of Python.

**Before:**
```python
def generate_path(steps, startingPosition):
    particlePosition = startingPosition
    direction = np.random.choice(4, steps)
    path = np.zeros((steps, 2))
    path[0] = startingPosition
    for n in range(steps - 1):                       # Python loop, 999 iterations
        path[n+1] = particlePosition + movements[direction[n]]
        particlePosition = path[n+1]
```

**After:**
```python
def generate_path(steps, startingPosition):
    direction = np.random.choice(4, steps - 1)
    stepVectors = movements[direction]
    path = np.empty((steps, 2))
    path[0] = startingPosition
    path[1:] = startingPosition + np.cumsum(stepVectors, axis=0)  # one call
```

#### Vectorized Collision Detection

The original collision detection checks each step one by one in a Python loop, and uses `try/except IndexError` for boundary handling. This loop runs 33,518 times even for 25 particles. Both the per-element access and the exception handling add overhead.

We replace this with NumPy array operations: first compute a boolean mask for valid indices (within lattice bounds). Out-of-bounds positions are handled by the mask instead of exceptions. Then use indexing to look up all valid positions in `stickyLattice` at once. 

**Before:**
```python
logicCheck = np.zeros(steps)
for n in range(steps):
    try:
        logicCheck[n] = stickyLattice[int(path[n,0]), int(path[n,1])] == 1
    except IndexError:
        continue
    if logicCheck[n] == 1:
        lattice, stickyLattice = particle_collision(logicCheck, path, ...)
        ...
        break
```

**After:**
```python
rows = path[:, 0].astype(int)
cols = path[:, 1].astype(int)
valid = (rows >= 0) & (rows <= maxIdx) & (cols >= 0) & (cols <= maxIdx)
safeRows = np.clip(rows, 0, maxIdx)
safeCols = np.clip(cols, 0, maxIdx)
logicCheck = np.zeros(steps, dtype=bool)
logicCheck[valid] = stickyLattice[safeRows[valid], safeCols[valid]] == 1
hitIndices = np.nonzero(logicCheck)[0]

if len(hitIndices) > 0:
    collisionIdx = hitIndices[0]
    lattice, stickyLattice = particle_collision(collisionIdx, path, ...)
```

The vectorized version checks all steps at once rather than stopping at the first hit. We then take `hitIndices[0]` to get the earliest collision, which gives the same result as the original `break`. The `np.nonzero` here works on a small 1000-element boolean array (path length), which is very different from the old full-lattice scans on a 1.56M-element array.

