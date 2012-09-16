/**
 * @file radical_main.cpp
 * @author Nishant Mehta
 *
 * Executable for RADICAL.
 *
 * This file is part of MLPACK 1.0.3.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <mlpack/core.hpp>
#include "radical.hpp"

PROGRAM_INFO("RADICAL", "An implementation of RADICAL, a method for independent"
    "component analysis (ICA).  Assuming that we have an input matrix X, the"
    "goal is to find a square unmixing matrix W such that Y = W * X and the "
    "dimensions of Y are independent components.");

PARAM_STRING_REQ("input_file", "Input dataset filename for ICA.", "i");

PARAM_STRING("output_ic", "File to save independent components to.", "o",
    "output_ic.csv");
PARAM_STRING("output_unmixing", "File to save unmixing matrix to.", "u",
    "output_unmixing.csv");

PARAM_DOUBLE("noise_std_dev", "Standard deviation of Gaussian noise.", "n",
    0.175);
PARAM_INT("replicates", "Number of Gaussian-perturbed replicates to use "
    "(per point) in Radical2D.", "r", 30);
PARAM_INT("angles", "Number of angles to consider in brute-force search "
    "during Radical2D.", "a", 150);
PARAM_INT("sweeps", "Number of sweeps (each sweep calls Radical2D once for "
    "each pair of dimensions", "S", 0);
PARAM_INT("seed", "Random seed.  If 0, 'std::time(NULL)' is used.", "s", 0);

using namespace mlpack;
using namespace mlpack::radical;
using namespace mlpack::math;
using namespace std;
using namespace arma;

int main(int argc, char* argv[])
{
  // Handle parameters.
  CLI::ParseCommandLine(argc, argv);

  // Set random seed.
  if (CLI::GetParam<int>("seed") != 0)
    RandomSeed((size_t) CLI::GetParam<int>("seed"));
  else
    RandomSeed((size_t) std::time(NULL));

  // Load the data.
  const string matXFilename = CLI::GetParam<string>("input_file");
  mat matX;
  data::Load(matXFilename, matX);

  // Load parameters.
  double noiseStdDev = CLI::GetParam<double>("noise_std_dev");
  size_t nReplicates = CLI::GetParam<int>("replicates");
  size_t nAngles = CLI::GetParam<int>("angles");
  size_t nSweeps = CLI::GetParam<int>("sweeps");

  if (nSweeps == 0)
  {
    nSweeps = matX.n_rows - 1;
  }

  // Run RADICAL.
  Radical rad(noiseStdDev, nReplicates, nAngles, nSweeps);
  mat matY;
  mat matW;
  rad.DoRadical(matX, matY, matW);

  // save results
  const string matYFilename = CLI::GetParam<string>("output_ic");
  data::Save(matYFilename, matY);

  const string matWFilename = CLI::GetParam<string>("output_unmixing");
  data::Save(matWFilename, matW);

  /*
  // compute and print objective
  mat matYT = trans(matY);
  double valEst = 0;
  for(size_t i = 0; i < matYT.n_cols; i++)
  {
    vec Yi = vec(matYT.col(i));
    valEst += rad.Vasicek(Yi);
  }
  Log::Info << "Objective (estimate): " << valEst << "." << endl;
  */
}
