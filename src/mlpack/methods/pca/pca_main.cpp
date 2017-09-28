/**
 * @file pca_main.cpp
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
#include <mlpack/core/util/cli.hpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "pca.hpp"
#include <mlpack/methods/pca/decomposition_policies/exact_svd_method.hpp>
#include <mlpack/methods/pca/decomposition_policies/quic_svd_method.hpp>
#include <mlpack/methods/pca/decomposition_policies/randomized_svd_method.hpp>
#include <mlpack/methods/pca/decomposition_policies/randomized_block_krylov_method.hpp>

using namespace mlpack;
using namespace mlpack::pca;
using namespace std;

// Document program.
PROGRAM_INFO("Principal Components Analysis", "This program performs principal "
    "components analysis on the given dataset using the exact, randomized, "
    "randomized block Krylov, or QUIC SVD method. It will transform the data "
    "onto its principal components, optionally performing dimensionality "
    "reduction by ignoring the principal components with the smallest "
    "eigenvalues."
    "\n\n"
    "To specify the dataset to perform PCA on, the " + PRINT_DATASET("input") +
    " parameter may be used.  A desired new dimensionality may be specified "
    "with the " + PRINT_PARAM_STRING("new_dimensionality") + " parameter, or "
    "the desired variance to retain may be specified with the " +
    PRINT_PARAM_STRING("var_to_retain") + " parameter.  If desired, the "
    "dataset may be scaled before running PCA with the " +
    PRINT_PARAM_STRING("scale") + " parameter."
    "\n\n"
    "Multiple different decomposition techniques may be used.  The method to "
    "use may be specified with the " +
    PRINT_PARAM_STRING("decomposition_method") + " parameter, and it may take "
    "the values 'exact', 'randomized', or 'quic'."
    "\n\n"
    "For example, to reduce the dimensionality of the matrix " +
    PRINT_DATASET("data") + " to 5 dimensions using randomized SVD for the "
    "decomposition, storing the output matrix to " +
    PRINT_DATASET("data_mod") + ", the following command may be used:"
    "\n\n" +
    PRINT_CALL("pca", "input", "data", "new_dimensionality", 5,
        "decomposition_method", "randomized", "output", "data_mod"));

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
  PCAType<DecompositionPolicy> p(scale);

  Log::Info << "Performing PCA on dataset..." << endl;
  double varRetained;

  if (CLI::HasParam("var_to_retain"))
  {
    if (CLI::HasParam("new_dimensionality"))
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

void mlpackMain()
{
  // Load input dataset.
  arma::mat& dataset = CLI::GetParam<arma::mat>("input");

  // Issue a warning if the user did not specify an output file.
  if (!CLI::HasParam("output"))
    Log::Warn << "--output_file is not specified; no output will be "
        << "saved." << endl;

  // Find out what dimension we want.
  size_t newDimension = dataset.n_rows; // No reduction, by default.
  if (CLI::GetParam<int>("new_dimensionality") != 0)
  {
    // Validate the parameter.
    newDimension = (size_t) CLI::GetParam<int>("new_dimensionality");
    if (newDimension > dataset.n_rows)
    {
      Log::Fatal << "New dimensionality (" << newDimension
          << ") cannot be greater than existing dimensionality ("
          << dataset.n_rows << ")!" << endl;
    }
  }

  // Get the options for running PCA.
  const bool scale = CLI::HasParam("scale");
  const double varToRetain = CLI::GetParam<double>("var_to_retain");
  const string decompositionMethod = CLI::GetParam<string>(
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
  else
  {
    // Invalid decomposition method.
    Log::Fatal << "Invalid decomposition method ('" << decompositionMethod
        << "'); valid choices are 'exact', 'randomized', "
        << "'randomized-block-krylov', 'quic'." << endl;
  }

  // Now save the results.
  if (CLI::HasParam("output"))
    CLI::GetParam<arma::mat>("output") = std::move(dataset);
}
