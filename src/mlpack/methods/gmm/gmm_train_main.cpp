/**
 * @author Parikshit Ram
 * @file methods/gmm/gmm_train_main.cpp
 *
 * This program trains a mixture of Gaussians on a given data matrix.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME gmm_train

#include <mlpack/core/util/mlpack_main.hpp>

#include "gmm.hpp"
#include "diagonal_gmm.hpp"
#include "no_constraint.hpp"
#include "diagonal_constraint.hpp"

#include <mlpack/methods/kmeans/refined_start.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

// Program Name.
BINDING_USER_NAME("Gaussian Mixture Model (GMM) Training");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of the EM algorithm for training Gaussian mixture "
    "models (GMMs).  Given a dataset, this can train a GMM for future use "
    "with other tools.");

// Long description.
BINDING_LONG_DESC(
    "This program takes a parametric estimate of a Gaussian mixture model (GMM)"
    " using the EM algorithm to find the maximum likelihood estimate.  The "
    "model may be saved and reused by other mlpack GMM tools."
    "\n\n"
    "The input data to train on must be specified with the " +
    PRINT_PARAM_STRING("input") + " parameter, and the number of Gaussians in "
    "the model must be specified with the " + PRINT_PARAM_STRING("gaussians") +
    " parameter.  Optionally, many trials with different random "
    "initializations may be run, and the result with highest log-likelihood on "
    "the training data will be taken.  The number of trials to run is specified"
    " with the " + PRINT_PARAM_STRING("trials") + " parameter.  By default, "
    "only one trial is run."
    "\n\n"
    "The tolerance for convergence and maximum number of iterations of the EM "
    "algorithm are specified with the " + PRINT_PARAM_STRING("tolerance") +
    " and " + PRINT_PARAM_STRING("max_iterations") + " parameters, "
    "respectively.  The GMM may be initialized for training with another model,"
    " specified with the " + PRINT_PARAM_STRING("input_model") + " parameter."
    " Otherwise, the model is initialized by running k-means on the data.  The "
    "k-means clustering initialization can be controlled with the " +
    PRINT_PARAM_STRING("kmeans_max_iterations") + ", " +
    PRINT_PARAM_STRING("refined_start") + ", " +
    PRINT_PARAM_STRING("samplings") + ", and " +
    PRINT_PARAM_STRING("percentage") + " parameters.  If " +
    PRINT_PARAM_STRING("refined_start") + " is specified, then the "
    "Bradley-Fayyad refined start initialization will be used.  This can often "
    "lead to better clustering results."
    "\n\n"
    "The 'diagonal_covariance' flag will cause the learned covariances to be "
    "diagonal matrices.  This significantly simplifies the model itself and "
    "causes training to be faster, but restricts the ability to fit more "
    "complex GMMs."
    "\n\n"
    "If GMM training fails with an error indicating that a covariance matrix "
    "could not be inverted, make sure that the " +
    PRINT_PARAM_STRING("no_force_positive") + " parameter is not "
    "specified.  Alternately, adding a small amount of Gaussian noise (using "
    "the " + PRINT_PARAM_STRING("noise") + " parameter) to the entire dataset"
    " may help prevent Gaussians with zero variance in a particular dimension, "
    "which is usually the cause of non-invertible covariance matrices."
    "\n\n"
    "The " + PRINT_PARAM_STRING("no_force_positive") + " parameter, if set, "
    "will avoid the checks after each iteration of the EM algorithm which "
    "ensure that the covariance matrices are positive definite.  Specifying "
    "the flag can cause faster runtime, but may also cause non-positive "
    "definite covariance matrices, which will cause the program to crash.");

// Example.
BINDING_EXAMPLE(
    "As an example, to train a 6-Gaussian GMM on the data in " +
    PRINT_DATASET("data") + " with a maximum of 100 iterations of EM and 3 "
    "trials, saving the trained GMM to " + PRINT_MODEL("gmm") + ", the "
    "following command can be used:"
    "\n\n" +
    PRINT_CALL("gmm_train", "input", "data", "gaussians", 6, "trials", 3,
        "output_model", "gmm") +
    "\n\n"
    "To re-train that GMM on another set of data " + PRINT_DATASET("data2") +
    ", the following command may be used: "
    "\n\n" +
    PRINT_CALL("gmm_train", "input_model", "gmm", "input", "data2",
        "gaussians", 6, "output_model", "new_gmm"));

// See also...
BINDING_SEE_ALSO("@gmm_generate", "#gmm_generate");
BINDING_SEE_ALSO("@gmm_probability", "#gmm_probability");
BINDING_SEE_ALSO("Gaussian Mixture Models on Wikipedia",
    "https://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model");
BINDING_SEE_ALSO("GMM class documentation", "@src/mlpack/methods/gmm/gmm.hpp");

// Parameters for training.
PARAM_MATRIX_IN_REQ("input", "The training data on which the model will be "
    "fit.", "i");
PARAM_INT_IN_REQ("gaussians", "Number of Gaussians in the GMM.", "g");

PARAM_INT_IN("seed", "Random seed.  If 0, 'std::time(NULL)' is used.", "s", 0);
PARAM_INT_IN("trials", "Number of trials to perform in training GMM.", "t", 1);

// Parameters for EM algorithm.
PARAM_DOUBLE_IN("tolerance", "Tolerance for convergence of EM.", "T", 1e-10);
PARAM_FLAG("no_force_positive", "Do not force the covariance matrices to be "
    "positive definite.", "P");
PARAM_INT_IN("max_iterations", "Maximum number of iterations of EM algorithm "
    "(passing 0 will run until convergence).", "n", 250);
PARAM_FLAG("diagonal_covariance", "Force the covariance of the Gaussians to "
    "be diagonal.  This can accelerate training time significantly.", "d");

// Parameters for dataset modification.
PARAM_DOUBLE_IN("noise", "Variance of zero-mean Gaussian noise to add to data.",
    "N", 0);

// Parameters for k-means initialization.
PARAM_INT_IN("kmeans_max_iterations", "Maximum number of iterations for the "
    "k-means algorithm (used to initialize EM).", "k", 1000);
PARAM_FLAG("refined_start", "During the initialization, use refined initial "
    "positions for k-means clustering (Bradley and Fayyad, 1998).", "r");
PARAM_INT_IN("samplings", "If using --refined_start, specify the number of "
    "samplings used for initial points.", "S", 100);
PARAM_DOUBLE_IN("percentage", "If using --refined_start, specify the percentage"
    " of the dataset used for each sampling (should be between 0.0 and 1.0).",
    "p", 0.02);

// Parameters for model saving/loading.
PARAM_MODEL_IN(GMM, "input_model", "Initial input GMM model to start training "
    "with.", "m");
PARAM_MODEL_OUT(GMM, "output_model", "Output for trained GMM model.", "M");

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  // Check parameters and load data.
  if (params.Get<int>("seed") != 0)
    RandomSeed((size_t) params.Get<int>("seed"));
  else
    RandomSeed((size_t) std::time(NULL));

  RequireParamValue<int>(params, "gaussians", [](int x) { return x > 0; }, true,
      "number of Gaussians must be positive");
  const int gaussians = params.Get<int>("gaussians");

  RequireParamValue<int>(params, "trials", [](int x) { return x > 0; }, true,
      "trials must be greater than 0");

  ReportIgnoredParam(params, {{ "diagonal_covariance", true }},
      "no_force_positive");
  RequireAtLeastOnePassed(params, { "output_model" }, false,
      "no model will be saved");

  RequireParamValue<double>(params, "noise", [](double x) { return x >= 0.0; },
      true, "variance of noise must be greater than or equal to 0");

  RequireParamValue<int>(params, "max_iterations", [](int x) { return x >= 0; },
      true, "max_iterations must be greater than or equal to 0");
  RequireParamValue<int>(params, "kmeans_max_iterations",
      [](int x) { return x >= 0; }, true,
      "kmeans_max_iterations must be greater than or equal to 0");

  arma::mat dataPoints = std::move(params.Get<arma::mat>("input"));

  // Do we need to add noise to the dataset?
  if (params.Has("noise"))
  {
    timers.Start("noise_addition");
    const double noise = params.Get<double>("noise");
    dataPoints += noise * arma::randn(dataPoints.n_rows, dataPoints.n_cols);
    Log::Info << "Added zero-mean Gaussian noise with variance " << noise
        << " to dataset." << std::endl;
    timers.Stop("noise_addition");
  }

  // Initialize GMM.
  GMM* gmm = NULL;

  if (params.Has("input_model"))
  {
    gmm = params.Get<GMM*>("input_model");

    if (gmm->Dimensionality() != dataPoints.n_rows)
      Log::Fatal << "Given input data (with " << PRINT_PARAM_STRING("input")
          << ") has dimensionality " << dataPoints.n_rows << ", but the initial"
          << " model (given with " << PRINT_PARAM_STRING("input_model")
          << " has dimensionality " << gmm->Dimensionality() << "!" << endl;
  }

  // Gather parameters for EMFit object.
  const size_t maxIterations = (size_t) params.Get<int>("max_iterations");
  const double tolerance = params.Get<double>("tolerance");
  const bool forcePositive = !params.Has("no_force_positive");
  const bool diagonalCovariance = params.Has("diagonal_covariance");
  const size_t kmeansMaxIterations =
      (size_t) params.Get<int>("kmeans_max_iterations");

  // This gets a bit weird because we need different types depending on whether
  // --refined_start is specified.
  double likelihood;
  if (params.Has("refined_start"))
  {
    RequireParamValue<int>(params, "samplings", [](int x) { return x > 0; },
        true, "number of samplings must be positive");
    RequireParamValue<double>(params, "percentage", [](double x) {
        return x > 0.0 && x <= 1.0; }, true, "percentage to sample must be "
        "be greater than 0.0 and less than or equal to 1.0");

    // Initialize the GMM if needed.  (We didn't do this earlier, because
    // RequireParamValue() would leak the memory if the check failed.)
    if (!params.Has("input_model"))
      gmm = new GMM(size_t(gaussians), dataPoints.n_rows);

    const int samplings = params.Get<int>("samplings");
    const double percentage = params.Get<double>("percentage");

    using KMeansType = KMeans<SquaredEuclideanDistance, RefinedStart>;

    KMeansType k(kmeansMaxIterations, SquaredEuclideanDistance(),
        RefinedStart(samplings, percentage));

    // Depending on the value of forcePositive and diagonalCovariance, we have
    // to use different types.
    if (diagonalCovariance)
    {
      // Convert GMMs into DiagonalGMMs.
      DiagonalGMM dgmm(gmm->Gaussians(), gmm->Dimensionality());
      for (size_t i = 0; i < size_t(gaussians); ++i)
      {
        dgmm.Component(i).Mean() = gmm->Component(i).Mean();
        dgmm.Component(i).Covariance(
            std::move(arma::diagvec(gmm->Component(i).Covariance())));
      }
      dgmm.Weights() = gmm->Weights();

      // Compute the parameters of the model using the EM algorithm.
      timers.Start("em");
      EMFit<KMeansType, PositiveDefiniteConstraint,
          DiagonalGaussianDistribution<>> em(maxIterations, tolerance, k);

      likelihood = dgmm.Train(dataPoints, params.Get<int>("trials"), false,
          em);
      timers.Stop("em");

      // Convert DiagonalGMMs into GMMs.
      for (size_t i = 0; i < size_t(gaussians); ++i)
      {
        gmm->Component(i).Mean() = dgmm.Component(i).Mean();
        gmm->Component(i).Covariance(
            arma::diagmat(dgmm.Component(i).Covariance()));
      }
      gmm->Weights() = dgmm.Weights();
    }
    else if (forcePositive)
    {
      // Compute the parameters of the model using the EM algorithm.
      timers.Start("em");
      EMFit<KMeansType> em(maxIterations, tolerance, k);
      likelihood = gmm->Train(dataPoints, params.Get<int>("trials"), false,
          em);
      timers.Stop("em");
    }
    else
    {
      // Compute the parameters of the model using the EM algorithm.
      timers.Start("em");
      EMFit<KMeansType, NoConstraint> em(maxIterations, tolerance, k);
      likelihood = gmm->Train(dataPoints, params.Get<int>("trials"), false,
          em);
      timers.Stop("em");
    }
  }
  else
  {
    // Initialize the GMM if needed.
    if (!params.Has("input_model"))
      gmm = new GMM(size_t(gaussians), dataPoints.n_rows);

    // Depending on the value of forcePositive and diagonalCovariance, we have
    // to use different types.
    if (diagonalCovariance)
    {
      // Convert GMMs into DiagonalGMMs.
      DiagonalGMM dgmm(gmm->Gaussians(), gmm->Dimensionality());
      for (size_t i = 0; i < size_t(gaussians); ++i)
      {
        dgmm.Component(i).Mean() = gmm->Component(i).Mean();
        dgmm.Component(i).Covariance(
            std::move(arma::diagvec(gmm->Component(i).Covariance())));
      }
      dgmm.Weights() = gmm->Weights();

      // Compute the parameters of the model using the EM algorithm.
      timers.Start("em");
      EMFit<KMeans<>, PositiveDefiniteConstraint,
          DiagonalGaussianDistribution<>> em(maxIterations, tolerance,
          KMeans<>(kmeansMaxIterations));

      likelihood = dgmm.Train(dataPoints, params.Get<int>("trials"), false,
          em);
      timers.Stop("em");

      // Convert DiagonalGMMs into GMMs.
      for (size_t i = 0; i < size_t(gaussians); ++i)
      {
        gmm->Component(i).Mean() = dgmm.Component(i).Mean();
        gmm->Component(i).Covariance(
            arma::diagmat(dgmm.Component(i).Covariance()));
      }
      gmm->Weights() = dgmm.Weights();
    }
    else if (forcePositive)
    {
      // Compute the parameters of the model using the EM algorithm.
      timers.Start("em");
      EMFit<> em(maxIterations, tolerance, KMeans<>(kmeansMaxIterations));
      likelihood = gmm->Train(dataPoints, params.Get<int>("trials"), false,
          em);
      timers.Stop("em");
    }
    else
    {
      // Compute the parameters of the model using the EM algorithm.
      timers.Start("em");
      KMeans<> k(kmeansMaxIterations);
      EMFit<KMeans<>, NoConstraint> em(maxIterations, tolerance, k);
      likelihood = gmm->Train(dataPoints, params.Get<int>("trials"), false,
          em);
      timers.Stop("em");
    }
  }

  Log::Info << "Log-likelihood of estimate: " << likelihood << "." << endl;

  params.Get<GMM*>("output_model") = gmm;
}
