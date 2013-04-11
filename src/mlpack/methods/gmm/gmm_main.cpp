/**
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @file gmm_main.cpp
 *
 * This program trains a mixture of Gaussians on a given data matrix.
 */
#include "gmm.hpp"

PROGRAM_INFO("Gaussian Mixture Model (GMM) Training",
    "This program takes a parametric estimate of a Gaussian mixture model (GMM)"
    " using the EM algorithm to find the maximum likelihood estimate.  The "
    "model is saved to an XML file, which contains information about each "
    "Gaussian."
    "\n\n"
    "If GMM training fails with an error indicating that a covariance matrix "
    "could not be inverted, this is probably remedied either via a larger "
    "option to the perturbation parameter, or alternately, adding a small "
    "amount of Gaussian noise to the entire dataset.  This helps prevent "
    "Gaussians with zero variance in a particular dimension, which is usually "
    "the cause of non-invertible covariance matrices.");

PARAM_STRING_REQ("input_file", "File containing the data on which the model "
    "will be fit.", "i");
PARAM_INT("gaussians", "Number of Gaussians in the GMM.", "g", 1);
PARAM_STRING("output_file", "The file to write the trained GMM parameters into "
    "(as XML).", "o", "gmm.xml");
PARAM_INT("seed", "Random seed.  If 0, 'std::time(NULL)' is used.", "s", 0);
PARAM_INT("trials", "Number of trials to perform in training GMM.", "t", 10);

// Parameters for EM algorithm.
PARAM_DOUBLE("tolerance", "Tolerance for convergence of EM.", "T", 1e-10);
PARAM_DOUBLE("perturbation", "Perturbation to add to zero-valued diagonal "
    "covariance entries.", "p", 1e-30);
PARAM_INT("max_iterations", "Maximum number of iterations of EM algorithm "
    "(passing 0 will run until convergence).", "n", 250);

// Parameters for dataset modification.
PARAM_DOUBLE("noise", "Variance of zero-mean Gaussian noise to add to data.",
    "N", 0);

using namespace mlpack;
using namespace mlpack::gmm;
using namespace mlpack::util;

int main(int argc, char* argv[])
{
  CLI::ParseCommandLine(argc, argv);

  // Check parameters and load data.
  if (CLI::GetParam<int>("seed") != 0)
    math::RandomSeed((size_t) CLI::GetParam<int>("seed"));
  else
    math::RandomSeed((size_t) std::time(NULL));

  arma::mat dataPoints;
  data::Load(CLI::GetParam<std::string>("input_file").c_str(), dataPoints,
      true);

  const int gaussians = CLI::GetParam<int>("gaussians");
  if (gaussians <= 0)
  {
    Log::Fatal << "Invalid number of Gaussians (" << gaussians << "); must "
        "be greater than or equal to 1." << std::endl;
  }

  // Do we need to add noise to the dataset?
  if (CLI::HasParam("noise"))
  {
    Timer::Start("noise_addition");
    const double noise = CLI::GetParam<double>("noise");
    dataPoints += noise * arma::randn(dataPoints.n_rows, dataPoints.n_cols);
    Log::Info << "Added zero-mean Gaussian noise with variance " << noise
        << " to dataset." << std::endl;
    Timer::Stop("noise_addition");
  }

  // Gather parameters for EMFit object.
  const size_t maxIterations = (size_t) CLI::GetParam<int>("max_iterations");
  const double tolerance = CLI::GetParam<double>("tolerance");
  const double perturbation = CLI::GetParam<double>("perturbation");
  EMFit<> em(maxIterations, tolerance, perturbation);

  // Calculate mixture of Gaussians.
  GMM<> gmm(size_t(gaussians), dataPoints.n_rows, em);

  // Compute the parameters of the model using the EM algorithm.
  Timer::Start("em");
  double likelihood = gmm.Estimate(dataPoints, CLI::GetParam<int>("trials"));
  Timer::Stop("em");

  Log::Info << "Log-likelihood of estimate: " << likelihood << ".\n";

  // Save results.
  gmm.Save(CLI::GetParam<std::string>("output_file"));
}
