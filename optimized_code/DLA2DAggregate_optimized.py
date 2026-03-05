# DLA 2D Aggregate, George Richards 4228068
import numpy as np
import time
from line_profiler import profile

# Initial values and movements possible in 2D (a plane)
length = 1250
movements = np.array([[1,0], [-1,0], [0,1], [0,-1]])
radiusDensity = np.linspace(2, 90, 10)

@profile
def initialise(length, movements):
    # Seed particle needs to be marked in the middle of the array/lattice
    # 'Sticky' site are the ones adjacent to the seed, marked in a seperate
    # array to not interfere with colouring of the later plot
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

@profile
def pick_starting_position(radius):
    # Starting position generated via parametric equations and a random angle
    startingPosition = np.zeros(2)
    randomAngle = np.random.randint(0, 359)
    angleRadians = np.deg2rad(randomAngle)
    x = radius*np.cos(angleRadians)
    y = radius*np.sin(angleRadians)
    startingPosition[0], startingPosition[1] = int((length/2) + x), int((length/2) + y)
    return startingPosition

@profile
def generate_path(steps, startingPosition):
    # Path of 1000 steps generated via randomly picking one of four directions
    direction = np.random.choice(4, steps - 1)
    stepVectors = movements[direction]
    path = np.empty((steps, 2))
    path[0] = startingPosition
    path[1:] = startingPosition + np.cumsum(stepVectors, axis=0)
    return path

@profile
def particle_collision(collisionIdx, path, lattice, stickyLattice, particleNumber, occupiedPositions):
    # Mark in the lattice which site the particle collided at
    cr, cc = int(path[collisionIdx, 0]), int(path[collisionIdx, 1])
    lattice[cr, cc] = particleNumber + 1
    occupiedPositions.add((cr, cc))
    # Add sticky neighbors for the new particle only
    maxIdx = length
    for dr, dc in movements:
        nr, nc = cr + dr, cc + dc
        if 0 <= nr <= maxIdx and 0 <= nc <= maxIdx and (nr, nc) not in occupiedPositions:
            stickyLattice[nr, nc] = 1
    # Remove the collision site from sticky lattice
    stickyLattice[cr, cc] = 0
    return lattice, stickyLattice

@profile
def lattice_radius_check(occupiedPositions, length, radius, killRadius):
    # Calculate max distance from center using tracked positions
    center = length / 2
    positions = np.array(list(occupiedPositions))
    distances = np.sqrt((positions[:, 0] - center)**2 + (positions[:, 1] - center)**2)
    latticeRadiusMax = distances.max()
    if latticeRadiusMax > radius:
        radius = latticeRadiusMax * 1.5
        killRadius = latticeRadiusMax * 2
    return latticeRadiusMax, radius, killRadius

@profile
def kill_check(steps, path, length, startingPosition, moving, killRadius):
    # Since particle did not collide with another, remove it if it walked
    # outside of the kill radius, otherwise continue with same particle
    killCheck = np.zeros(steps)
    killCheck = np.sqrt(((path[:,0]-length/2)**2) + ((path[:,1]-length/2)**2))
    toKill = np.any(killCheck > killRadius)
    if toKill: # Kill particle
        kill = 1
        moving = 0
        return kill, moving, startingPosition
    else: # Keep going with current particle
        kill = 0
        startingPosition = path[-1] # Last position as new starting position
        return kill, moving, startingPosition

@profile
def main(particleCap, steps):
    # More initial values
    t0 = time.time()
    lattice, stickyLattice = initialise(length, movements)
    # Track occupied positions incrementally to avoid repeated np.nonzero scans
    center = int(length / 2)
    occupiedPositions = {(center, center)}
    latticeRadiusData = []
    particleNumber = 1
    kill = 1 # Set as 1 to generate a starting position on first iteration
    radius = 15
    killRadius = radius*2
    moving = 1
    maxIdx = length

    while particleNumber < particleCap:

        # Generate starting position again if needed
        if kill or not moving:
            startingPosition = pick_starting_position(radius)

        moving = 1
        kill = 0

        while moving:
            path = generate_path(steps, startingPosition)
            # Vectorized collision check: clip to valid indices, then bulk lookup
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
                lattice, stickyLattice = particle_collision(collisionIdx, path, lattice, stickyLattice, particleNumber, occupiedPositions)
                latticeRadiusMax, radius, killRadius = lattice_radius_check(occupiedPositions, length, radius, killRadius)

                latticeRadiusData.append([particleNumber, latticeRadiusMax])
                moving = 0
                particleNumber += 1
            else:
                # Kill particle or carry on with same one
                kill, moving, startingPosition = kill_check(steps, path, length, startingPosition, moving, killRadius)

    print('The radius of the fractal is', latticeRadiusData[-1][1])
    N, r = zip(*latticeRadiusData)

    t1 = time.time()
    t2 = round((t1-t0)/60,1)
    print('Finished run in', t2, 'minutes.')

    # Finding positions of all partiles in lattice for radial distribution function
    tCorrelation1 = time.time()
    particlePositions = np.nonzero(lattice)
    particlePositionsRows = particlePositions[0][:]
    particlePositionsColumns = particlePositions[1][:]
    particleCoords = np.zeros(shape=(particleCap, 2))
    particleCoords[:,0], particleCoords[:,1] = particlePositionsRows, particlePositionsColumns
    filledDensityStore = np.zeros(shape=(10, particleCap))
    for particle in range(particleCap - 1):
        # Setting reference particle
        referenceParticleRow = particleCoords[particle,0]
        referenceParticleColumn = particleCoords[particle,1]
        filledDensity = np.zeros(10)
        for width in range(10):
            # Defining circle to be used for radial distribution function
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
                coords[n,0], coords[n,1] =  coordsToCheck[0][n], coordsToCheck[1][n]
            # Counting which sites are empty or filled
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
    # Averaging over all runs
    filledDensityMean = np.zeros(10)
    for n in range(10):
        filledDensityMean[n] = np.mean(filledDensityStore[n,:])

    tCorrelation2 = time.time()
    tCorrelationFinish = round(tCorrelation2 - tCorrelation1,2)
    print('Finished correlation in', tCorrelationFinish, 'seconds')
    print('Finished')

if __name__ == '__main__':
    main(particleCap=10001, steps=1000)
