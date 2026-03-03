"""
DLA 2D Aggregate with timing instrumentation.
Exports run_timed(particleCap, steps) which returns a dict of timing results.
"""
# DLA 2D Aggregate, George Richards 4228068
import numpy as np
import time

# Initial values and movements possible in 2D (a plane)
length = 1250
movements = np.array([[1,0], [-1,0], [0,1], [0,-1]])
radiusDensity = np.linspace(2, 90, 10)


def initialise(length, movements):
    lattice = np.zeros((length+1, length+1))
    initialSeed = [(length/2), (length/2)]
    lattice[int(initialSeed[0]), int(initialSeed[1])] = 1
    stickyInitial = np.nonzero(lattice)
    stickySites = []
    for n in range(4):
        stickySites.append((stickyInitial[0][0], stickyInitial[1][0]) + movements[n])
    stickyLattice = np.zeros((length+1, length+1))
    for n in range(4):
        stickyLattice[stickySites[n][0], stickySites[n][1]] = 1
    return lattice, stickyLattice


def pick_starting_position(radius):
    startingPosition = np.zeros(2)
    randomAngle = np.random.randint(0, 359)
    angleRadians = np.deg2rad(randomAngle)
    x = radius*np.cos(angleRadians)
    y = radius*np.sin(angleRadians)
    startingPosition[0], startingPosition[1] = int((length/2) + x), int((length/2) + y)
    return startingPosition


def generate_path(steps, startingPosition):
    particlePosition = startingPosition
    direction = np.random.choice(4, steps)
    path = np.zeros((steps, 2))
    path[0] = startingPosition
    for n in range(steps - 1):
        path[n+1] = particlePosition + movements[direction[n]]
        particlePosition = path[n+1]
    return path


def particle_collision(logicCheck, path, lattice, stickyLattice, particleNumber):
    logicCheck2 = np.nonzero(logicCheck)
    collision = path[logicCheck2[0]]
    lattice[int(collision[0,0]), int(collision[0,1])] = particleNumber + 1
    stickyInitialTest = np.nonzero(lattice)
    stickySitesTest = []
    for n in range(4):
        toDo = len(stickyInitialTest[0])
        for o in range(toDo):
            stickySitesTest.append((stickyInitialTest[0][o], stickyInitialTest[1][o]) + movements[n])
    for n in range(4*toDo):
        stickyLattice[stickySitesTest[n][0], stickySitesTest[n][1]] = 1
    latticeCheck = np.nonzero(lattice)
    latticeCheckLength = len(latticeCheck[0])
    for n in range(latticeCheckLength):
        stickyLattice[latticeCheck[0][n], latticeCheck[1][n]] = 0
    return lattice, stickyLattice


def lattice_radius_check(lattice, length, radius, killRadius):
    latticeRadiusInitial = np.nonzero(lattice)
    latticeRadiusY = length/2 - latticeRadiusInitial[0]
    latticeRadiusX = latticeRadiusInitial[1] - length/2
    latticeRadiusValues = np.sqrt(latticeRadiusX**2 + latticeRadiusY**2)
    latticeRadiusMax = max(latticeRadiusValues)
    if latticeRadiusMax > radius:
        radius = latticeRadiusMax*1.5
        killRadius = latticeRadiusMax*2
        return latticeRadiusMax, radius, killRadius
    else:
        return latticeRadiusMax, radius, killRadius


def kill_check(steps, path, length, startingPosition, moving, killRadius):
    killCheck = np.zeros(steps)
    killCheck = np.sqrt(((path[:,0]-length/2)**2) + ((path[:,1]-length/2)**2))
    toKill = np.any(killCheck > killRadius)
    if toKill:
        kill = 1
        moving = 0
        return kill, moving, startingPosition
    else:
        kill = 0
        startingPosition = path[-1]
        return kill, moving, startingPosition


def run_timed(particleCap, steps):
    """Run the DLA simulation with timing for while-loop and for-loop.

    Returns dict with keys:
        total_time, while_time, for_time, while_pct, for_pct
    """
    t_total_start = time.time()
    lattice, stickyLattice = initialise(length, movements)
    latticeRadiusData = []
    particleNumber = 1
    kill = 1
    radius = 15
    killRadius = radius*2
    moving = 1

    t_while_start = time.time()
    while particleNumber < particleCap:
        if kill or not moving:
            startingPosition = pick_starting_position(radius)
        moving = 1
        kill = 0
        while moving:
            path = generate_path(steps, startingPosition)
            logicCheck = np.zeros(steps)
            for n in range(steps):
                try:
                    logicCheck[n] = stickyLattice[int(path[n,0]), int(path[n,1])] == 1
                except IndexError:
                    continue
                if logicCheck[n] == 1:
                    lattice, stickyLattice = particle_collision(logicCheck, path, lattice, stickyLattice, particleNumber)
                    latticeRadiusMax, radius, killRadius = lattice_radius_check(lattice, length, radius, killRadius)
                    latticeRadiusData.append([particleNumber, latticeRadiusMax])
                    moving = 0
                    particleNumber += 1
                    break
            if moving:
                kill, moving, startingPosition = kill_check(steps, path, length, startingPosition, moving, killRadius)
    t_while_end = time.time()

    print('The radius of the fractal is', latticeRadiusData[-1][1])
    N, r = zip(*latticeRadiusData)

    particlePositions = np.nonzero(lattice)
    particlePositionsRows = particlePositions[0][:]
    particlePositionsColumns = particlePositions[1][:]
    particleCoords = np.zeros(shape=(particleCap, 2))
    particleCoords[:,0], particleCoords[:,1] = particlePositionsRows, particlePositionsColumns
    filledDensityStore = np.zeros(shape=(10, particleCap))

    t_for_start = time.time()
    for particle in range(particleCap - 1):
        referenceParticleRow = particleCoords[particle,0]
        referenceParticleColumn = particleCoords[particle,1]
        filledDensity = np.zeros(10)
        for width in range(10):
            radiusCircle = int(radiusDensity[width])
            repeats = 1
            emptyStore = np.zeros(repeats)
            filledStore = np.zeros(repeats)
            correlationLattice = np.zeros(shape=((2*radiusCircle)+1,(2*radiusCircle)+1))
            for row in range((2*radiusCircle)+1):
                correlationLattice[row] = lattice[int(referenceParticleRow+radiusCircle-row),int(referenceParticleColumn-radiusCircle):int(referenceParticleColumn+radiusCircle+1)]
            circleX, circleY = np.ogrid[-radiusCircle:radiusCircle+1, -radiusCircle:radiusCircle+1]
            circleSelection = circleX*circleX + circleY*circleY <= radiusCircle*radiusCircle
            coordsToCheck = np.where(circleSelection == True)
            coords = np.zeros(shape=(len(coordsToCheck[0]),2))
            for n in range(len(coordsToCheck[0])):
                coords[n,0], coords[n,1] = coordsToCheck[0][n], coordsToCheck[1][n]
            filled = 0
            empty = 0
            for n in range(len(coords)):
                row = int(coords[n,0])
                column = int(coords[n,1])
                check = correlationLattice[row, column]
                if check == 0:
                    empty += 1
                else:
                    filled += 1
            emptyStore = empty
            filledStore = filled
            filledDensity[width] = filled / (empty + filled)
        filledDensityStore[:,particle] = filledDensity
    t_for_end = time.time()

    filledDensityMean = np.zeros(10)
    for n in range(10):
        filledDensityMean[n] = np.mean(filledDensityStore[n,:])

    print('Finished')
    t_total_end = time.time()

    while_time = t_while_end - t_while_start
    for_time = t_for_end - t_for_start
    total_time = t_total_end - t_total_start

    return {
        'total_time': total_time,
        'while_time': while_time,
        'for_time': for_time,
        'while_pct': while_time / total_time * 100 if total_time > 0 else 0,
        'for_pct': for_time / total_time * 100 if total_time > 0 else 0,
    }
