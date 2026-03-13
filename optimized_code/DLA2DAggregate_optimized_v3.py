import numpy as np
import time
import cupy as cp

length = 1250
movements = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
movements_cp = cp.array(movements)
radiusDensity = np.linspace(2, 90, 10)



_CIRCLE_MASKS = []
for _r in radiusDensity:
    _rc = int(_r)
    _cx, _cy = np.ogrid[-_rc:_rc + 1, -_rc:_rc + 1]
    _CIRCLE_MASKS.append(_cx * _cx + _cy * _cy <= _rc * _rc)

_MAX_RADIUS = int(radiusDensity[-1])  # = 90; used as pad width


def initialise(length, movements):
    movements_cp = cp.array(movements)
    
    lattice = cp.zeros((length + 1, length + 1))
    initialSeed = [length // 2, length // 2]
    lattice[initialSeed[0], initialSeed[1]] = 1
    
    stickyInitial = cp.nonzero(lattice)
    r0 = int(stickyInitial[0][0].get())
    c0 = int(stickyInitial[1][0].get())
    
    stickyLattice = cp.zeros((length + 1, length + 1))
    for n in range(4):
        dr, dc = int(movements[n][0]), int(movements[n][1])
        nr, nc = r0 + dr, c0 + dc
        stickyLattice[nr, nc] = 1
    
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


def generate_path_gpu(steps, startingPosition):
    direction = cp.random.randint(0, 4, steps - 1)
    stepVectors = movements_cp[direction]
    start = cp.array(startingPosition)
    path = cp.empty((steps, 2))
    path[0] = start
    path[1:] = start + cp.cumsum(stepVectors, axis=0)
    return path


def particle_collision(collisionIdx, path, lattice, stickyLattice,
                       particleNumber, occupiedPositions):
    cr = int(path[collisionIdx, 0].get())
    cc = int(path[collisionIdx, 1].get())
    lattice[cr, cc] = particleNumber + 1
    occupiedPositions.add((cr, cc))
    for dr, dc in movements:
        nr, nc = cr + int(dr), cc + int(dc)
        if 0 <= nr <= length and 0 <= nc <= length and (nr, nc) not in occupiedPositions:
            stickyLattice[nr, nc] = 1
    stickyLattice[cr, cc] = 0
    return lattice, stickyLattice


def lattice_radius_check(occupiedPositions, length, radius, killRadius):
    center = length / 2
    positions = np.array(list(occupiedPositions))
    distances = np.sqrt(
        (positions[:, 0] - center) ** 2 + (positions[:, 1] - center) ** 2
    )
    latticeRadiusMax = distances.max()
    if latticeRadiusMax > radius:
        radius = latticeRadiusMax * 1.5
        killRadius = latticeRadiusMax * 2
    return latticeRadiusMax, radius, killRadius


def kill_check(steps, path, length, startingPosition, moving, killRadius):
    killCheck = cp.sqrt(
        ((path[:, 0] - length / 2) ** 2) + ((path[:, 1] - length / 2) ** 2)
    )
    toKill = bool(cp.any(killCheck > killRadius))
    if toKill:
        return 1, 0, startingPosition
    return 0, moving, path[-1].get()

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
            total = len(values)
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
    radius     = 15
    killRadius = radius * 2
    moving     = 1
    maxIdx     = length

    while particleNumber < particleCap:
        if kill or not moving:
            startingPosition = pick_starting_position(radius)
        moving = 1
        kill   = 0

        while moving:
            path = generate_path_gpu(steps, startingPosition)

            rows= cp.array(path[:, 0].astype(int))
            cols= cp.array(path[:, 1].astype(int))
            valid= ((rows >= 0) & (rows <= maxIdx) &
                        (cols >= 0) & (cols <= maxIdx))

            safeRows = cp.clip(rows, 0, maxIdx)
            safeCols = cp.clip(cols, 0, maxIdx)

            logicCheck= cp.zeros(steps, dtype=bool)
            logicCheck[valid]= (stickyLattice[safeRows[valid], safeCols[valid]] == 1)

            hitIndices = cp.nonzero(logicCheck)[0]

            if len(hitIndices) > 0:
                collisionIdx = int(hitIndices[0].get())
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

    particlePositions = cp.nonzero(lattice)
    particleCoords = np.zeros((particleCap, 2))
    particleCoords[:, 0] = particlePositions[0][:].get()
    particleCoords[:, 1] = particlePositions[1][:].get()

    filledDensityMean, _ = _radial_density_vectorized(
        lattice.get(), particleCoords, particleCap
    )

    tCorrelation2 = time.time()
    print(f'Radial density finished in '
          f'{round(tCorrelation2 - tCorrelation1, 2)} seconds.')
    print('Finished')


if __name__ == '__main__':
    main(particleCap=1001, steps=10)
