/**
 * @file det_main.cpp
 * @author Parikshit Ram (pram@cc.gatech.edu)
 *
 * This file runs density estimation trees.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include "dt_utils.hpp"

using namespace mlpack;
using namespace mlpack::det;
using namespace std;

PROGRAM_INFO("Density Estimation With Density Estimation Trees",
    "This program performs a number of functions related to Density Estimation "
    "Trees.  The optimal Density Estimation Tree (DET) can be trained on a set "
    "of data (specified by --training_file or -t) using cross-validation (with "
    "number of folds specified by --folds).  This trained density estimation "
    "tree may then be saved to a model file with the --output_model_file (-M) "
    "option."
    "\n\n"
    "The variable importances of each dimension may be saved with the "
    "--vi_file (-i) option, and the density estimates on each training point "
    "may be saved to the file specified with the --training_set_estimates_file "
    "(-e) option."
    "\n\n"
    "This program also can provide density estimates for a set of test points, "
    "specified in the --test_file (-T) file.  The density estimation tree used "
    "for this task will be the tree that was trained on the given training "
    "points, or a tree stored in the file given with the --input_model_file "
    "(-m) parameter.  The density estimates for the test points may be saved "
    "into the file specified with the --test_set_estimates_file (-E) option.");

// Input data files.
PARAM_MATRIX_IN("training", "The data set on which to build a density "
    "estimation tree.", "t");

// Input or output model.
PARAM_STRING_IN("input_model_file", "File containing already trained density "
    "estimation tree.", "m", "");
PARAM_STRING_OUT("output_model_file", "File to save trained density estimation "
    "tree to.", "M");

// Output data files.
PARAM_MATRIX_IN("test", "A set of test points to estimate the density of.",
    "T");
PARAM_MATRIX_OUT("training_set_estimates", "The output density estimates on "
    "the training set from the final optimally pruned tree.", "e");
PARAM_MATRIX_OUT("test_set_estimates", "The output estimates on the test set "
    "from the final optimally pruned tree.", "E");
PARAM_MATRIX_OUT("vi", "The output variable importance values for each "
    "feature.", "i");

// Parameters for the training algorithm.
PARAM_INT_IN("folds", "The number of folds of cross-validation to perform for "
    "the estimation (0 is LOOCV)", "f", 10);
PARAM_INT_IN("min_leaf_size", "The minimum size of a leaf in the unpruned, "
    "fully grown DET.", "l", 5);
PARAM_INT_IN("max_leaf_size", "The maximum size of a leaf in the unpruned, "
    "fully grown DET.", "L", 10);
/*
PARAM_FLAG("volume_regularization", "This flag gives the used the option to use"
    "a form of regularization similar to the usual alpha-pruning in decision "
    "tree. But instead of regularizing on the number of leaves, you regularize "
    "on the sum of the inverse of the volume of the leaves (meaning you "
    "penalize low volume leaves.", "R");
*/

int main(int argc, char *argv[])
{
  CLI::ParseCommandLine(argc, argv);

  // Validate input parameters.
  if (CLI::HasParam("training") && CLI::HasParam("input_model_file"))
    Log::Fatal << "Only one of --training_file (-t) or --input_model_file (-m) "
        << "may be specified!" << endl;

  if (!CLI::HasParam("training") && !CLI::HasParam("input_model_file"))
    Log::Fatal << "Neither --training_file (-t) nor --input_model_file (-m) "
        << "are specified!" << endl;

  if (!CLI::HasParam("training"))
  {
    if (CLI::HasParam("training_set_estimates"))
      Log::Warn << "--training_set_estimates_file (-e) ignored because "
          << "--training_file (-t) is not specified." << endl;
    if (CLI::HasParam("folds"))
      Log::Warn << "--folds (-f) ignored because --training_file (-t) is not "
          << "specified." << endl;
    if (CLI::HasParam("min_leaf_size"))
      Log::Warn << "--min_leaf_size (-l) ignored because --training_file (-t) "
          << "is not specified." << endl;
    if (CLI::HasParam("max_leaf_size"))
      Log::Warn << "--max_leaf_size (-L) ignored because --training_file (-t) "
          << "is not specified." << endl;
  }
  else if (!CLI::HasParam("output_model_file") &&
           !CLI::HasParam("training_set_estimates") &&
           !CLI::HasParam("vi"))
  {
    Log::Warn << "None of --output_model_file (-M), --training_set_estimates "
        << "(-e), or --vi (-i) are specified; no output will be saved!" << endl;
  }

  if (!CLI::HasParam("test") && CLI::HasParam("test_set_estimates"))
    Log::Warn << "--test_set_estimates_file (-E) ignored because --test_file "
        << "(-T) is not specified." << endl;

  // Are we training a DET or loading from file?
  DTree<arma::mat, int>* tree;
  if (CLI::HasParam("training"))
  {
    arma::mat trainingData = std::move(CLI::GetParam<arma::mat>("training"));

    // Cross-validation here.
    size_t folds = CLI::GetParam<int>("folds");
    if (folds == 0)
    {
      folds = trainingData.n_cols;
      Log::Info << "Performing leave-one-out cross validation." << endl;
    }
    else
    {
      Log::Info << "Performing " << folds << "-fold cross validation." << endl;
    }

    const bool regularization = false;
//    const bool regularization = CLI::HasParam("volume_regularization");
    const int maxLeafSize = CLI::GetParam<int>("max_leaf_size");
    const int minLeafSize = CLI::GetParam<int>("min_leaf_size");

    // Obtain the optimal tree.
    Timer::Start("det_training");
    tree = Trainer<arma::mat, int>(trainingData, folds, regularization,
        maxLeafSize, minLeafSize, "");
    Timer::Stop("det_training");

    // Compute training set estimates, if desired.
    if (CLI::HasParam("training_set_estimates"))
    {
      // Compute density estimates for each point in the training set.
      arma::rowvec trainingDensities(trainingData.n_cols);
      Timer::Start("det_estimation_time");
      for (size_t i = 0; i < trainingData.n_cols; i++)
        trainingDensities[i] = tree->ComputeValue(trainingData.unsafe_col(i));
      Timer::Stop("det_estimation_time");

      CLI::GetParam<arma::mat>("training_set_estimates") =
          std::move(trainingDensities);
    }
  }
  else
  {
    data::Load(CLI::GetParam<string>("input_model_file"), "det_model", tree,
        true);
  }

  // Compute the density at the provided test points and output the density in
  // the given file.
  if (CLI::HasParam("test"))
  {
    arma::mat testData = std::move(CLI::GetParam<arma::mat>("test"));

    // Compute test set densities.
    Timer::Start("det_test_set_estimation");
    arma::rowvec testDensities(testData.n_cols);
    for (size_t i = 0; i < testData.n_cols; i++)
      testDensities[i] = tree->ComputeValue(testData.unsafe_col(i));
    Timer::Stop("det_test_set_estimation");

    if (CLI::HasParam("test_set_estimates"))
      CLI::GetParam<arma::mat>("test_set_estimates") = std::move(testDensities);
  }

  // Print variable importance.
  if (CLI::HasParam("vi"))
  {
    arma::vec importances;
    tree->ComputeVariableImportance(importances);
    CLI::GetParam<arma::mat>("vi") = std::move(importances.t());
  }

  // Save the model, if desired.
  if (CLI::HasParam("output_model_file"))
    data::Save(CLI::GetParam<string>("output_model_file"), "det_model", tree,
        false);

  delete tree;
}
