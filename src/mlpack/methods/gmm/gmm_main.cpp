/**
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @file gmm_main.cpp
 *
 * This program trains a mixture of Gaussians on a given data matrix.
 *
 * This file is part of MLPACK 1.0.7.
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

#include "gmm.hpp"
#include "no_constraint.hpp"

#include <mlpack/methods/kmeans/refined_start.hpp>

using namespace mlpack;
using namespace mlpack::gmm;
using namespace mlpack::util;
using namespace mlpack::kmeans;
using namespace std;

PROGRAM_INFO("Gaussian Mixture Model (GMM) Training",
    "This program takes a parametric estimate of a Gaussian mixture model (GMM)"
    " using the EM algorithm to find the maximum likelihood estimate.  The "
    "model is saved to an XML file, which contains information about each "
    "Gaussian."
    "\n\n"
    "If GMM training fails with an error indicating that a covariance matrix "
    "could not be inverted, be sure that the 'no_force_positive' flag was not "
    "specified.  Alternately, adding a small amount of Gaussian noise to the "
    "entire dataset may help prevent Gaussians with zero variance in a "
    "particular dimension, which is usually the cause of non-invertible "
    "covariance matrices."
    "\n\n"
    "The 'no_force_positive' flag, if set, will avoid the checks after each "
    "iteration of the EM algorithm which ensure that the covariance matrices "
    "are positive definite.  Specifying the flag can cause faster runtime, "
    "but may also cause non-positive definite covariance matrices, which will "
    "cause the program to crash.");

PARAM_STRING_REQ("input_file", "File containing the data on which the model "
    "will be fit.", "i");
PARAM_INT("gaussians", "Number of Gaussians in the GMM.", "g", 1);
PARAM_STRING("output_file", "The file to write the trained GMM parameters into "
    "(as XML).", "o", "gmm.xml");
PARAM_INT("seed", "Random seed.  If 0, 'std::time(NULL)' is used.", "s", 0);
PARAM_INT("trials", "Number of trials to perform in training GMM.", "t", 10);

// Parameters for EM algorithm.
PARAM_DOUBLE("tolerance", "Tolerance for convergence of EM.", "T", 1e-10);
PARAM_FLAG("no_force_positive", "Do not force the covariance matrices to be "
    "positive definite.", "P");
PARAM_INT("max_iterations", "Maximum number of iterations of EM algorithm "
    "(passing 0 will run until convergence).", "n", 250);

// Parameters for dataset modification.
PARAM_DOUBLE("noise", "Variance of zero-mean Gaussian noise to add to data.",
    "N", 0);

// Parameters for k-means initialization.
PARAM_FLAG("refined_start", "During the initialization, use refined initial "
    "positions for k-means clustering (Bradley and Fayyad, 1998).", "r");
PARAM_INT("samplings", "If using --refined_start, specify the number of "
    "samplings used for initial points.", "S", 100);
PARAM_DOUBLE("percentage", "If using --refined_start, specify the percentage of"
    " the dataset used for each sampling (should be between 0.0 and 1.0).",
    "p", 0.02);

int main(int argc, char* argv[])
{
  CLI::ParseCommandLine(argc, argv);

  // Check parameters and load data.
  if (CLI::GetParam<int>("seed") != 0)
    math::RandomSeed((size_t) CLI::GetParam<int>("seed"));
  else
    math::RandomSeed((size_t) std::time(NULL));

  arma::mat dataPoints;
  data::Load(CLI::GetParam<string>("input_file"), dataPoints,
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
  const bool forcePositive = !CLI::HasParam("no_force_positive");

  // This gets a bit weird because we need different types depending on whether
  // --refined_start is specified.
  double likelihood;
  if (CLI::HasParam("refined_start"))
  {
    const int samplings = CLI::GetParam<int>("samplings");
    const double percentage = CLI::GetParam<double>("percentage");

    if (samplings <= 0)
      Log::Fatal << "Number of samplings (" << samplings << ") must be greater"
          << " than 0!" << std::endl;

    if (percentage <= 0.0 || percentage > 1.0)
      Log::Fatal << "Percentage for sampling (" << percentage << ") must be "
          << "greater than 0.0 and less than or equal to 1.0!" << std::endl;

    typedef KMeans<metric::SquaredEuclideanDistance, RefinedStart> KMeansType;

    // These are default parameters.
    KMeansType k(1000, 1.0, metric::SquaredEuclideanDistance(),
        RefinedStart(samplings, percentage));

    // Depending on the value of 'forcePositive', we have to use different
    // types.
    if (forcePositive)
    {
      EMFit<KMeansType> em(maxIterations, tolerance, k);

      GMM<EMFit<KMeansType> > gmm(size_t(gaussians), dataPoints.n_rows, em);

      // Compute the parameters of the model using the EM algorithm.
      Timer::Start("em");
      likelihood = gmm.Estimate(dataPoints, CLI::GetParam<int>("trials"));
      Timer::Stop("em");

      // Save results.
      gmm.Save(CLI::GetParam<string>("output_file"));
    }
    else
    {
      EMFit<KMeansType, NoConstraint> em(maxIterations, tolerance, k);

      GMM<EMFit<KMeansType, NoConstraint> > gmm(size_t(gaussians),
          dataPoints.n_rows, em);

      // Compute the parameters of the model using the EM algorithm.
      Timer::Start("em");
      likelihood = gmm.Estimate(dataPoints, CLI::GetParam<int>("trials"));
      Timer::Stop("em");

      // Save results.
      gmm.Save(CLI::GetParam<string>("output_file"));
    }
  }
  else
  {
    // Depending on the value of forcePositive, we have to use different types.
    if (forcePositive)
    {
      EMFit<> em(maxIterations, tolerance);

      // Calculate mixture of Gaussians.
      GMM<> gmm(size_t(gaussians), dataPoints.n_rows, em);

      // Compute the parameters of the model using the EM algorithm.
      Timer::Start("em");
      likelihood = gmm.Estimate(dataPoints, CLI::GetParam<int>("trials"));
      Timer::Stop("em");

      // Save results.
      gmm.Save(CLI::GetParam<string>("output_file"));
    }
    else
    {
      // Use no constraints on the covariance matrix.
      EMFit<KMeans<>, NoConstraint> em(maxIterations, tolerance);

      // Calculate mixture of Gaussians.
      GMM<EMFit<KMeans<>, NoConstraint> > gmm(size_t(gaussians),
          dataPoints.n_rows, em);

      // Compute the parameters of the model using the EM algorithm.
      Timer::Start("em");
      likelihood = gmm.Estimate(dataPoints, CLI::GetParam<int>("trials"));
      Timer::Stop("em");

      // Save results.
      gmm.Save(CLI::GetParam<string>("output_file"));
    }
  }

  Log::Info << "Log-likelihood of estimate: " << likelihood << ".\n";
}
