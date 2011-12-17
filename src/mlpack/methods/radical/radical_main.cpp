/**
 * @file radical_main.cpp
 * @author Nishant Mehta
 *
 * Executable for RADICAL.
 */

#include <mlpack/core.hpp>
#include <armadillo>
#include "radical.hpp"

PROGRAM_INFO("RADICAL", "An implementation of RADICAL, a method for independent"
    "component analysis (ICA).  Assuming that we have an input matrix X, the"
    "goal is to find a square unmixing matrix W such that Y = W * X and the "
    "dimensions of Y are independent components.\n"
    "\n"
    "For more details, see the following paper:\n"
    "\n"
    "@article{\n"
    "  title = {ICA Using Spacings Estimates of Entropy},\n"
    "  author = {Learned-Miller, E.G. and Fisher III, J.W.},\n"
    "  journal = {Journal of Machine Learning Research},\n"
    "  volume = {4},\n"
    "  pages = {1271--1295},\n"
    "  year = {2003}\n"
    "}");

PARAM_STRING_REQ("input_file", "Input dataset filename for ICA.", "");

PARAM_STRING("output_ic", "File to save independent components to.", "o",
    "output_ic.csv");
PARAM_STRING("output_unmixing", "File to save unmixing matrix to.", "u",
    "output_unmixing.csv");

PARAM_DOUBLE("noise_std_dev", "Standard deviation of Gaussian noise", "",
    0.175);
PARAM_INT("replicates", "Number of Gaussian-perturbed replicates to use "
    "(per point) in Radical2D.", "", 30);
PARAM_INT("angles", "Number of angles to consider in brute-force search "
    "during Radical2D.", "", 150);
PARAM_INT("sweeps", "Number of sweeps (each sweep calls Radical2D once for "
    "each pair of dimensions", "", 0);

using namespace mlpack;
using namespace mlpack::radical;
using namespace std;
using namespace arma;

int main(int argc, char* argv[])
{
  // Handle parameters.
  CLI::ParseCommandLine(argc, argv);

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
