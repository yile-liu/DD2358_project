import numpy as np
import time
import optimization_pybind11 as opt_cpp

length = 1250
movements = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
radiusDensity = np.linspace(2, 90, 10)

_CIRCLE_MASKS = []
for _r in radiusDensity:
    _rc = int(_r)
    _cx, _cy = np.ogrid[-_rc:_rc + 1, -_rc:_rc + 1]
    _CIRCLE_MASKS.append(_cx * _cx + _cy * _cy <= _rc * _rc)

_MAX_RADIUS = int(radiusDensity[-1])  # = 90; used as pad width


def initialise(length, movements):
    lattice = np.zeros((length + 1, length + 1))
    initialSeed = [length / 2, length / 2]
    lattice[int(initialSeed[0]), int(initialSeed[1])] = 1
    stickyInitial = np.nonzero(lattice)
    stickySites = []
    for n in range(4):
        stickySites.append(
            (stickyInitial[0][0], stickyInitial[1][0]) + movements[n]
        )
    stickyLattice = np.zeros((length + 1, length + 1))
    for n in range(4):
        stickyLattice[stickySites[n][0], stickySites[n][1]] = 1
    return lattice, stickyLattice

def pick_starting_position(radius):
    startingPosition = np.zeros(2)
    randomAngle = np.random.randint(0, 359)
    angleRadians = np.deg2rad(randomAngle)
    x = radius * np.cos(angleRadians)
    y = radius * np.sin(angleRadians)
    startingPosition[0] = int((length / 2) + x)
    startingPosition[1] = int((length / 2) + y)
    return startingPosition


def generate_paths_gpu(n_walkers, steps, startingPositions):
    directions = np.random.randint(0, 4, (n_walkers, steps - 1))
    step_vectors = np.asarray(movements)[directions]  # (n_walkers, steps-1, 2)
    paths = np.empty((n_walkers, steps, 2), dtype=np.int32)
    paths[:, 0, :] = np.asarray(startingPositions)
    paths[:, 1:, :] = startingPositions[:, None, :] + np.cumsum(step_vectors, axis=1)
    return paths  # stays on GPU

def generate_path(steps, startingPosition):
    direction = np.random.choice(4, steps - 1)
    stepVectors = movements[direction]
    path = np.empty((steps, 2))
    path[0] = startingPosition
    path[1:] = startingPosition + np.cumsum(stepVectors, axis=0)
    return path


def particle_collision(collisionIdx, path, lattice, stickyLattice,
                       particleNumber, occupiedPositions):
    cr, cc = int(path[collisionIdx, 0]), int(path[collisionIdx, 1])
    lattice[cr, cc] = particleNumber + 1
    occupiedPositions.add((cr, cc))
    maxIdx = length
    for dr, dc in movements:
        nr, nc = cr + dr, cc + dc
        if (0 <= nr <= maxIdx and 0 <= nc <= maxIdx
                and (nr, nc) not in occupiedPositions):
            stickyLattice[nr, nc] = 1
    stickyLattice[cr, cc] = 0
    return lattice, stickyLattice

def check_collisions_gpu(paths_gpu, stickyLattice_gpu, maxIdx):
    rows = np.clip(paths_gpu[:, :, 0], 0, maxIdx)
    cols = np.clip(paths_gpu[:, :, 1], 0, maxIdx)
    hits = stickyLattice_gpu[rows, cols] == 1  # (n_walkers, steps)
    first_hits = np.argmax(hits, axis=1)        # first collision per walker
    hit_mask = hits.any(axis=1)                 # which walkers actually hit
    return first_hits, hit_mask


def lattice_radius_check(occupiedPositions, length, radius, killRadius):
    center = length / 2
    positions = np.array(list(occupiedPositions))
    distances = np.sqrt(
        (positions[:, 0] - center) ** 2 + (positions[:, 1] - center) ** 2
    )
    latticeRadiusMax = float(distances.max())
    if latticeRadiusMax > radius:
        killRadius = float(latticeRadiusMax * 2)
        radius = float(latticeRadiusMax * 1.5)
    return latticeRadiusMax, radius, killRadius


def kill_check(steps, path, length, startingPosition, moving, killRadius):
    killCheck = np.sqrt(
        ((path[:, 0] - length / 2) ** 2) + ((path[:, 1] - length / 2) ** 2)
    )
    toKill = np.any(killCheck > killRadius)
    if toKill:
        return 1, 0, startingPosition
    return 0, moving, path[-1]


_CIRCLE_MASKS = []
for _r in radiusDensity:
    _rc = int(_r)
    _cx, _cy = np.ogrid[-_rc:_rc + 1, -_rc:_rc + 1]
    _CIRCLE_MASKS.append(_cx * _cx + _cy * _cy <= _rc * _rc)

_MAX_RADIUS = int(radiusDensity[-1])  # = 90; used as pad width

def _radial_density_vectorized(lattice, particleCoords, particleCap):
    """
    Opt-4 + Opt-5: fully vectorized radial density C(r).

    Opt-4 – Sub-lattice extraction
        Pad the lattice once by _MAX_RADIUS so every 2-D slice is always
        in-bounds. Inside the loop, extract each sub-lattice with a single
        2-D NumPy slice instead of a per-row Python loop.

    Opt-5 – Circle counting
        Apply the pre-computed boolean mask directly to the sub-array and
        count non-zero elements with np.count_nonzero, eliminating both the
        coordinate-copy loop and the per-element counting loop.
    """

    padded = np.pad(lattice, _MAX_RADIUS, mode='constant', constant_values=0)

    filledDensityStore = np.zeros((10, particleCap))

    for particle in range(particleCap - 1):
        pr = int(particleCoords[particle, 0]) + _MAX_RADIUS
        pc = int(particleCoords[particle, 1]) + _MAX_RADIUS

        filledDensity = np.zeros(10)
        for width in range(10):
            rc   = int(radiusDensity[width])
            mask = _CIRCLE_MASKS[width]

            sub = padded[pr - rc: pr + rc + 1,
                         pc - rc: pc + rc + 1]

            values = sub[mask]
            filled = int(np.count_nonzero(values))
            total  = len(values)
            filledDensity[width] = filled / total if total > 0 else 0.0

        filledDensityStore[:, particle] = filledDensity

    filledDensityMean = filledDensityStore.mean(axis=1)
    return filledDensityMean, filledDensityStore



def main(particleCap, steps):
    t0 = time.time()
    lattice, stickyLattice = initialise(length, movements)
    center = int(length / 2)
    occupiedPositions = {(center, center)}
    latticeRadiusData = []
    particleNumber = 1
    kill       = 1
    radius     = 15.0
    killRadius = radius * 2
    moving     = 1
    maxIdx     = length

    while particleNumber < particleCap:
        if kill or not moving:
            startingPosition = pick_starting_position(radius)
        moving = 1
        kill   = 0

        while moving:
            path = generate_path(steps, startingPosition)

            rows  = path[:, 0].astype(int)
            cols  = path[:, 1].astype(int)
            valid = ((rows >= 0) & (rows <= maxIdx) &
                     (cols >= 0) & (cols <= maxIdx))
            safeRows = np.clip(rows, 0, maxIdx)
            safeCols = np.clip(cols, 0, maxIdx)
            logicCheck = np.zeros(steps, dtype=bool)
            logicCheck[valid] = (
                stickyLattice[safeRows[valid], safeCols[valid]] == 1
            )
            hitIndices = np.nonzero(logicCheck)[0]

            if len(hitIndices) > 0:
                collisionIdx = hitIndices[0]
                lattice, stickyLattice = particle_collision(
                    collisionIdx, path, lattice, stickyLattice,
                    particleNumber, occupiedPositions
                )
                latticeRadiusMax, radius, killRadius = lattice_radius_check(
                    occupiedPositions, length, radius, killRadius
                )
                latticeRadiusData.append([particleNumber, latticeRadiusMax])
                moving = 0
                particleNumber += 1
            else:
                kill, moving, startingPosition = kill_check(
                    steps, path, length, startingPosition, moving, killRadius
                )

    print('The radius of the fractal is', latticeRadiusData[-1][1])
    t1 = time.time()
    print(f'Aggregation finished in {round((t1 - t0) / 60, 2)} minutes.')

    tCorrelation1 = time.time()
    particlePositions = np.nonzero(lattice)
    particleCoords    = np.zeros((particleCap, 2))
    particleCoords[:, 0] = particlePositions[0][:]
    particleCoords[:, 1] = particlePositions[1][:]

    filledDensityMean, _ = _radial_density_vectorized(
        lattice, particleCoords, particleCap
    )

    tCorrelation2 = time.time()
    print(f'Radial density vectorized through numpy finished in '
          f'{round(tCorrelation2 - tCorrelation1, 2)} seconds.')
    print('Finished')

    # now also use pybind11
    t1 = time.time()
    filledDensityStore = opt_cpp.for_substitute(particleCap, particleCoords, lattice)
    t2 = time.time()

    print(f'time for radial density for pybind11: ', t2 - t1)


if __name__ == '__main__':
    main(particleCap=10001, steps=1000)
