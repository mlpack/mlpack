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
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/io.hpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "pca.hpp"
#include <mlpack/methods/pca/decomposition_policies/exact_svd_method.hpp>
#include <mlpack/methods/pca/decomposition_policies/quic_svd_method.hpp>
#include <mlpack/methods/pca/decomposition_policies/randomized_svd_method.hpp>
#include <mlpack/methods/pca/decomposition_policies/randomized_block_krylov_method.hpp>

using namespace mlpack;
using namespace mlpack::pca;
using namespace mlpack::util;
using namespace std;

// Program Name.
BINDING_NAME("Principal Components Analysis");

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
BINDING_SEE_ALSO("mlpack::pca::PCA C++ class documentation",
        "@doxygen/classmlpack_1_1pca_1_1PCA.html");

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
void RunPCA(arma::mat& dataset,
            const size_t newDimension,
            const bool scale,
            const double varToRetain)
{
  PCA<DecompositionPolicy> p(scale);

  Log::Info << "Performing PCA on dataset..." << endl;
  double varRetained;

  if (IO::HasParam("var_to_retain"))
  {
    if (IO::HasParam("new_dimensionality"))
      Log::Warn << "New dimensionality (-d) ignored because --var_to_retain "
          << "(-r) was specified." << endl;

    varRetained = p.Apply(dataset, varToRetain);
  }
  else
  {
    varRetained = p.Apply(dataset, newDimension);
  }

  Log::Info << (varRetained * 100) << "% of variance retained (" <<
      dataset.n_rows << " dimensions)." << endl;
}

static void mlpackMain()
{
  // Load input dataset.
  arma::mat& dataset = IO::GetParam<arma::mat>("input");

  // Issue a warning if the user did not specify an output file.
  RequireAtLeastOnePassed({ "output" }, false, "no output will be saved");

  // Check decomposition method validity.
  RequireParamInSet<string>("decomposition_method", { "exact", "randomized",
      "randomized-block-krylov", "quic" }, true,
      "unknown decomposition method");

  // Find out what dimension we want.
  RequireParamValue<int>("new_dimensionality", [](int x) { return x >= 0; },
      true, "new dimensionality must be non-negative");
  std::ostringstream error;
  error << "cannot be greater than existing dimensionality (" << dataset.n_rows
      << ")";
  RequireParamValue<int>("new_dimensionality",
      [dataset](int x) { return x <= (int) dataset.n_rows; }, true,
      error.str());

  RequireParamValue<double>("var_to_retain",
      [](double x) { return x >= 0.0 && x <= 1.0; }, true,
      "variance retained must be between 0 and 1");
  size_t newDimension = (IO::GetParam<int>("new_dimensionality") == 0) ?
      dataset.n_rows : IO::GetParam<int>("new_dimensionality");

  // Get the options for running PCA.
  const bool scale = IO::HasParam("scale");
  const double varToRetain = IO::GetParam<double>("var_to_retain");
  const string decompositionMethod = IO::GetParam<string>(
      "decomposition_method");

  // Perform PCA.
  if (decompositionMethod == "exact")
  {
    RunPCA<ExactSVDPolicy>(dataset, newDimension, scale, varToRetain);
  }
  else if (decompositionMethod == "randomized")
  {
    RunPCA<RandomizedSVDPolicy>(dataset, newDimension, scale, varToRetain);
  }
  else if (decompositionMethod == "randomized-block-krylov")
  {
    RunPCA<RandomizedBlockKrylovSVDPolicy>(dataset, newDimension, scale,
        varToRetain);
  }
  else if (decompositionMethod == "quic")
  {
    RunPCA<QUICSVDPolicy>(dataset, newDimension, scale, varToRetain);
  }

  // Now save the results.
  if (IO::HasParam("output"))
    IO::GetParam<arma::mat>("output") = std::move(dataset);
}
