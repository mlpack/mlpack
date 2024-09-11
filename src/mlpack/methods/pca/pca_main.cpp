/**
 * @file methods/pca/pca_main.cpp
 * @author Ryan Curtin
 * @author Marcus Edel
 *
 * Main executable to run PCA.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME pca

#include <mlpack/core/util/mlpack_main.hpp>

#include "pca.hpp"

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

// Program Name.
BINDING_USER_NAME("Principal Components Analysis");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of several strategies for principal components analysis "
    "(PCA), a common preprocessing step.  Given a dataset and a desired new "
    "dimensionality, this can reduce the dimensionality of the data using the "
    "linear transformation determined by PCA.");

// Long description.
BINDING_LONG_DESC(
    "This program performs principal components analysis on the given dataset "
    "using the exact, randomized, randomized block Krylov, or QUIC SVD method. "
    "It will transform the data onto its principal components, optionally "
    "performing dimensionality reduction by ignoring the principal components "
    "with the smallest eigenvalues."
    "\n\n"
    "Use the " + PRINT_PARAM_STRING("input") + " parameter to specify the "
    "dataset to perform PCA on.  A desired new dimensionality can be specified "
    "with the " + PRINT_PARAM_STRING("new_dimensionality") + " parameter, or "
    "the desired variance to retain can be specified with the " +
    PRINT_PARAM_STRING("var_to_retain") + " parameter.  If desired, the "
    "dataset can be scaled before running PCA with the " +
    PRINT_PARAM_STRING("scale") + " parameter."
    "\n\n"
    "Multiple different decomposition techniques can be used.  The method to "
    "use can be specified with the " +
    PRINT_PARAM_STRING("decomposition_method") + " parameter, and it may take "
    "the values 'exact', 'randomized', or 'quic'.");

// Example.
BINDING_EXAMPLE(
    "For example, to reduce the dimensionality of the matrix " +
    PRINT_DATASET("data") + " to 5 dimensions using randomized SVD for the "
    "decomposition, storing the output matrix to " +
    PRINT_DATASET("data_mod") + ", the following command can be used:"
    "\n\n" +
    PRINT_CALL("pca", "input", "data", "new_dimensionality", 5,
        "decomposition_method", "randomized", "output", "data_mod"));

// See also...
BINDING_SEE_ALSO("Principal component analysis on Wikipedia",
    "https://en.wikipedia.org/wiki/Principal_component_analysis");
BINDING_SEE_ALSO("PCA C++ class documentation",
    "@doc/user/methods/pca.md");

// Parameters for program.
PARAM_MATRIX_IN_REQ("input", "Input dataset to perform PCA on.", "i");
PARAM_MATRIX_OUT("output", "Matrix to save modified dataset to.", "o");
PARAM_INT_IN("new_dimensionality", "Desired dimensionality of output dataset. "
    "If 0, no dimensionality reduction is performed.", "d", 0);
PARAM_DOUBLE_IN("var_to_retain", "Amount of variance to retain; should be "
    "between 0 and 1.  If 1, all variance is retained.  Overrides -d.", "r", 0);

PARAM_FLAG("scale", "If set, the data will be scaled before running PCA, such "
    "that the variance of each feature is 1.", "s");

PARAM_STRING_IN("decomposition_method", "Method used for the principal "
    "components analysis: 'exact', 'randomized', 'randomized-block-krylov', "
    "'quic'.", "c", "exact");


//! Run RunPCA on the specified dataset with the given decomposition method.
template<typename DecompositionPolicy>
void RunPCA(util::Params& params,
            util::Timers& timers,
            arma::mat& dataset,
            const size_t newDimension,
            const bool scale,
            const double varToRetain)
{
  PCA<DecompositionPolicy> p(scale);

  Log::Info << "Performing PCA on dataset..." << endl;
  double varRetained;

  timers.Start("pca");
  if (params.Has("var_to_retain"))
  {
    if (params.Has("new_dimensionality"))
      Log::Warn << "New dimensionality (-d) ignored because --var_to_retain "
          << "(-r) was specified." << endl;

    varRetained = p.Apply(dataset, varToRetain);
  }
  else
  {
    varRetained = p.Apply(dataset, newDimension);
  }
  timers.Stop("pca");

  Log::Info << (varRetained * 100) << "% of variance retained (" <<
      dataset.n_rows << " dimensions)." << endl;
}

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  // Load input dataset.
  arma::mat& dataset = params.Get<arma::mat>("input");

  // Issue a warning if the user did not specify an output file.
  RequireAtLeastOnePassed(params, { "output" }, false,
      "no output will be saved");

  // Check decomposition method validity.
  RequireParamInSet<string>(params, "decomposition_method",
      { "exact", "randomized", "randomized-block-krylov", "quic" }, true,
      "unknown decomposition method");

  // Find out what dimension we want.
  RequireParamValue<int>(params, "new_dimensionality",
      [](int x) { return x >= 0; },
      true, "new dimensionality must be non-negative");
  std::ostringstream error;
  error << "cannot be greater than existing dimensionality (" << dataset.n_rows
      << ")";
  RequireParamValue<int>(params, "new_dimensionality",
      [dataset](int x) { return x <= (int) dataset.n_rows; }, true,
      error.str());

  RequireParamValue<double>(params, "var_to_retain",
      [](double x) { return x >= 0.0 && x <= 1.0; }, true,
      "variance retained must be between 0 and 1");
  size_t newDimension = (params.Get<int>("new_dimensionality") == 0) ?
      dataset.n_rows : params.Get<int>("new_dimensionality");

  // Get the options for running PCA.
  const bool scale = params.Has("scale");
  const double varToRetain = params.Get<double>("var_to_retain");
  const string decompositionMethod = params.Get<string>(
      "decomposition_method");

  // Perform PCA.
  if (decompositionMethod == "exact")
  {
    RunPCA<ExactSVDPolicy>(params, timers, dataset, newDimension, scale,
        varToRetain);
  }
  else if (decompositionMethod == "randomized")
  {
    RunPCA<RandomizedSVDPCAPolicy>(params, timers, dataset, newDimension, scale,
        varToRetain);
  }
  else if (decompositionMethod == "randomized-block-krylov")
  {
    RunPCA<RandomizedBlockKrylovSVDPolicy>(params, timers, dataset,
        newDimension, scale, varToRetain);
  }
  else if (decompositionMethod == "quic")
  {
    RunPCA<QUICSVDPolicy>(params, timers, dataset, newDimension, scale,
        varToRetain);
  }

  // Now save the results.
  if (params.Has("output"))
    params.Get<arma::mat>("output") = std::move(dataset);
}
