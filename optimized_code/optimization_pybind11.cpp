#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h> 
#include <vector>

using namespace std;
namespace py = pybind11;

// radiusDensity = np.linspace(2, 90, 10)
vector<double> radiusDensity = {2, 11, 20, 29, 38, 47, 56, 65, 74, 83};

// we assume that the paprticleCoords and the lattice have dimensions 2
vector<vector<double>> for_substitute(const int particleCap, vector<vector<double>> particleCoords, vector<vector<double>> lattice) {
    // here we predefine the size as it is expensive to allocate memory, hence we allocate everything at the start
    vector<vector<double>> filledDensityStore(10, vector<double>(particleCap, 0));

    for (int particle = 0; particle < particleCap - 1; particle++) {
        // setting reference particle
        int referenceParticleRow = particleCoords[particle][0];
        int referenceParticleColumn = particleCoords[particle][1];
        vector<double> filledDensity(10, 0);
        for (int width = 0; width < 10; width++) {
            // defining circle to be used for radial distribution function
            int radiusCircle = int(radiusDensity[width]);
            int repeats = 1;
            vector<double> emptyStore(repeats, 0);
            vector<double> filledStore(repeats, 0);
            vector<vector<double>> correlationLattice((2*radiusCircle)+1, vector<double>((2*radiusCircle)+1, 0));

            int diameter = (2 * radiusCircle) + 1;
            int latticeHeight = lattice.size();
            int latticeWidth = lattice[0].size(); // Assuming a rectangular grid

            for (int row = 0; row < diameter; row++) {
                int targetRow = referenceParticleRow + radiusCircle - row;

                for (int colOffset = 0; colOffset < diameter; colOffset++) {
                    int targetCol = referenceParticleColumn - radiusCircle + colOffset;

                    // If out of bounds, leave as 0 (empty)
                    if (targetRow >= 0 && targetRow < latticeHeight && 
                        targetCol >= 0 && targetCol < latticeWidth) {
                        correlationLattice[row][colOffset] = lattice[targetRow][targetCol];
                    }
                }
            }

            
            // defining circle selection
            vector<vector<int>> circleSelection((2*radiusCircle)+1, vector<int>((2*radiusCircle)+1, 0));
            for (int row = 0; row < (2*radiusCircle)+1; row++) {
                for (int column = 0; column < (2*radiusCircle)+1; column++) {
                    int y = row - radiusCircle;
                    int x = column - radiusCircle;
                    if (x*x + y*y <= radiusCircle*radiusCircle)
                        circleSelection[row][column] = 1;
                }
            }
            // coordsToCheck = np.where(circleSelection == True)
            // we get all the coordinates where the circleSelection is true
            vector<vector<int>> coordsToCheck;
            for (int row = 0; row < (2*radiusCircle)+1; row++) {
                for (int column = 0; column < (2*radiusCircle)+1; column++) {
                    if (circleSelection[row][column] == 1) {
                        coordsToCheck.push_back({row, column});
                    }
                }
            }

            vector<vector<double>> coords(coordsToCheck.size(), vector<double>(2, 0));
            for (int n = 0; n < coordsToCheck.size(); n++) {
                coords[n][0] = coordsToCheck[n][0];
                coords[n][1] = coordsToCheck[n][1];
            }

            // counting which sites are empty or filled
            int filled = 0;
            int empty = 0;
            for (int n = 0; n < coords.size(); n++) {
                int row = int(coords[n][0]);
                int column = int(coords[n][1]);
                if (correlationLattice[row][column] == 0) {
                    empty += 1;
                } else {
                    filled += 1;
                }
            }

            filledDensity[width] = double(filled) / double(empty + filled);
        }

        for (int width = 0; width < 10; width++) {
            filledDensityStore[width][particle] = filledDensity[width];
        }
    }
    return filledDensityStore;
}



    PYBIND11_MODULE(optimization_pybind11, m) {
        m.doc() = "pybind11 example plugin"; // optional module docstring
        m.def("for_substitute", &for_substitute, py::return_value_policy::automatic);
    }