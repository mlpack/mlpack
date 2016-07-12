/**
 * @file radical_main.cpp
 * @author Nishant Mehta
 *
 * Executable for RADICAL.
 */

#include <mlpack/core.hpp>
#include "radical.hpp"

PROGRAM_INFO("RADICAL", "An implementation of RADICAL, a method for independent"
    "component analysis (ICA).  Assuming that we have an input matrix X, the"
    "goal is to find a square unmixing matrix W such that Y = W * X and the "
    "dimensions of Y are independent components.  If the algorithm is running"
    "particularly slowly, try reducing the number of replicates.");

PARAM_STRING_IN_REQ("input_file", "Input dataset filename for ICA.", "i");

PARAM_STRING_OUT("output_ic", "File to save independent components to.", "o");
PARAM_STRING_OUT("output_unmixing", "File to save unmixing matrix to.", "u");

PARAM_DOUBLE_IN("noise_std_dev", "Standard deviation of Gaussian noise.", "n",
    0.175);
PARAM_INT_IN("replicates", "Number of Gaussian-perturbed replicates to use "
    "(per point) in Radical2D.", "r", 30);
PARAM_INT_IN("angles", "Number of angles to consider in brute-force search "
    "during Radical2D.", "a", 150);
PARAM_INT_IN("sweeps", "Number of sweeps; each sweep calls Radical2D once for "
    "each pair of dimensions.", "S", 0);
PARAM_INT_IN("seed", "Random seed.  If 0, 'std::time(NULL)' is used.", "s", 0);
PARAM_FLAG("objective", "If set, an estimate of the final objective function "
    "is printed.", "O");

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

  if (!CLI::HasParam("output_ic") && !CLI::HasParam("output_unmixing"))
    Log::Warn << "Neither --output_ic nor --output_unmixing were specified; "
        << "no output will be saved!" << endl;

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

  // Save results.
  const string matYFilename = CLI::GetParam<string>("output_ic");
  if (matYFilename != "")
    data::Save(matYFilename, matY);

  const string matWFilename = CLI::GetParam<string>("output_unmixing");
  if (matWFilename != "")
    data::Save(matWFilename, matW);

  if (CLI::HasParam("objective"))
  {
    // Compute and print objective.
    mat matYT = trans(matY);
    double valEst = 0;
    for (size_t i = 0; i < matYT.n_cols; i++)
    {
      vec y = vec(matYT.col(i));
      valEst += rad.Vasicek(y);
    }

    // Force output even if --verbose is not given.
    const bool ignoring = Log::Info.ignoreInput;
    Log::Info.ignoreInput = false;
    Log::Info << "Objective (estimate): " << valEst << "." << endl;
    Log::Info.ignoreInput = ignoring;
  }
}
