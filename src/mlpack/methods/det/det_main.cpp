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
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/cli.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "dt_utils.hpp"

using namespace mlpack;
using namespace mlpack::det;
using namespace std;

PROGRAM_INFO("Density Estimation With Density Estimation Trees",
    "This program performs a number of functions related to Density Estimation "
    "Trees.  The optimal Density Estimation Tree (DET) can be trained on a set "
    "of data (specified by " + PRINT_PARAM_STRING("training") + ") using "
    "cross-validation (with number of folds specified with the " +
    PRINT_PARAM_STRING("folds") + " parameter).  This trained density "
    "estimation tree may then be saved with the " +
    PRINT_PARAM_STRING("output_model") + " output parameter."
    "\n\n"
    "The variable importances (that is, the feature importance values for each "
    "dimension) may be saved with the " + PRINT_PARAM_STRING("vi") + " output"
    " parameter, and the density estimates for each training point may be saved"
    " with the " + PRINT_PARAM_STRING("training_set_estimates") + " output "
    "parameter."
    "\n\n"
    "Enabling path printing for each node outputs strings like `LRLRLR` for "
    "each entry in the test set, or training one, if test is not provided. "
    "If `lr-id` or `id-lr` is provided, it'll also print the id (tag) of the "
    "node (not just leaf!) in tree along the path to the leaf."
    "\n\n"
    "This program also can provide density estimates for a set of test points, "
    "specified in the " + PRINT_PARAM_STRING("test") + " parameter.  The "
    "density estimation tree used for this task will be the tree that was "
    "trained on the given training points, or a tree given as the parameter " +
    PRINT_PARAM_STRING("input_model") + ".  The density estimates for the test"
    " points may be saved using the " +
    PRINT_PARAM_STRING("test_set_estimates") + " output parameter.");

// Input data files.
PARAM_MATRIX_IN("training", "The data set on which to build a density "
    "estimation tree.", "t");

// Input or output model.
PARAM_MODEL_IN(DTree<>, "input_model", "Trained density estimation "
    "tree to load.", "m");
PARAM_MODEL_OUT(DTree<>, "output_model", "Output to save trained "
    "density estimation tree to.", "M");

// Output data files.
PARAM_MATRIX_IN("test", "A set of test points to estimate the density of.",
    "T");
PARAM_MATRIX_OUT("training_set_estimates", "The output density estimates on "
    "the training set from the final optimally pruned tree.", "e");
PARAM_MATRIX_OUT("test_set_estimates", "The output estimates on the test set "
    "from the final optimally pruned tree.", "E");
PARAM_MATRIX_OUT("vi", "The output variable importance values for each "
    "feature.", "i");

// Tagging and path printing options
PARAM_STRING_IN("path_format", "The format of path printing - lr|id-lr|lr-id",
                "p", "lr");

PARAM_STRING_OUT("tag_counters_file", "The file to output number of points "
                 "that went to each leaf.", "c");

PARAM_STRING_OUT("tag_file", "The file to output the tags (and possibly paths)"
                 " for each sample in the test set.", "g");

PARAM_FLAG("skip_pruning", "Whether to bypass the pruning process and output "
              "the unpruned tree only.", "s");

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


void mlpackMain()
{
  // Validate input parameters.
  if (CLI::HasParam("training") && CLI::HasParam("input_model"))
    Log::Fatal << "Only one of " << PRINT_PARAM_STRING("training") << " or " <<
      PRINT_PARAM_STRING("input_model") << " may be specified!" << endl;

  if (!CLI::HasParam("training") && !CLI::HasParam("input_model"))
    Log::Fatal << "Neither " << PRINT_PARAM_STRING("training") << " nor " <<
      PRINT_PARAM_STRING("input_model") << " are specified!" << endl;

  if (CLI::HasParam("tag_file") &&
      !CLI::HasParam("training") && !CLI::HasParam("test"))
  {
    Log::Fatal << "Neither " << PRINT_PARAM_STRING("training") << " nor " <<
      PRINT_PARAM_STRING("test") << " are specified, but needed when " <<
      PRINT_PARAM_STRING("tag_file") << " is asked." << endl;
  }

  if (!CLI::HasParam("training"))
  {
    if (CLI::HasParam("training_set_estimates"))
      Log::Warn << PRINT_PARAM_STRING("training_set_estimates") <<
        " ignored because " << PRINT_PARAM_STRING("training") <<
        " is not specified." << endl;
    if (CLI::HasParam("folds"))
      Log::Warn << PRINT_PARAM_STRING("folds") << " ignored because " <<
        PRINT_PARAM_STRING("training") << " is not specified." << endl;
    if (CLI::HasParam("min_leaf_size"))
      Log::Warn << PRINT_PARAM_STRING("min_leaf_size") << " ignored because " <<
        PRINT_PARAM_STRING("training") << " is not specified." << endl;
    if (CLI::HasParam("max_leaf_size"))
      Log::Warn << PRINT_PARAM_STRING("max_leaf_size") << " ignored because " <<
        PRINT_PARAM_STRING("training") << " is not specified." << endl;
  }
  else if (!CLI::HasParam("output_model") &&
           !CLI::HasParam("training_set_estimates") &&
           !CLI::HasParam("vi"))
  {
    Log::Warn << "None of " << PRINT_PARAM_STRING("output_model") << ", " <<
      PRINT_PARAM_STRING("training_set_estimates") << ", or " <<
      PRINT_PARAM_STRING("vi") << " are specified; no output will be saved!" <<
      endl;
  }

  if (!CLI::HasParam("test") && CLI::HasParam("test_set_estimates"))
    Log::Warn << PRINT_PARAM_STRING("test_set_estimates") << " ignored " <<
      "because " << PRINT_PARAM_STRING("test") << " is not specified." << endl;

  // Are we training a DET or loading from file?
  DTree<arma::mat, int>* tree;
  arma::mat trainingData;
  arma::mat testData;

  if (CLI::HasParam("training"))
  {
    trainingData = std::move(CLI::GetParam<arma::mat>("training"));

    const bool regularization = false;
//    const bool regularization = CLI::HasParam("volume_regularization");
    const int maxLeafSize = CLI::GetParam<int>("max_leaf_size");
    const int minLeafSize = CLI::GetParam<int>("min_leaf_size");
    const bool skipPruning = CLI::HasParam("skip_pruning");
    size_t folds = CLI::GetParam<int>("folds");

    if (folds == 0)
      folds = trainingData.n_cols;

    // Obtain the optimal tree.
    Timer::Start("det_training");
    tree = Trainer<arma::mat, int>(trainingData, folds, regularization,
                                   maxLeafSize, minLeafSize,
                                   skipPruning);
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
    tree = &CLI::GetParam<DTree<arma::mat>>("input_model");
  }

  // Compute the density at the provided test points and output the density in
  // the given file.
  if (CLI::HasParam("test"))
  {
    testData = std::move(CLI::GetParam<arma::mat>("test"));
    if (CLI::HasParam("test_set_estimates"))
    {
      // Compute test set densities.
      Timer::Start("det_test_set_estimation");
      arma::rowvec testDensities(testData.n_cols);

      for (size_t i = 0; i < testData.n_cols; i++)
        testDensities[i] = tree->ComputeValue(testData.unsafe_col(i));

      Timer::Stop("det_test_set_estimation");

      CLI::GetParam<arma::mat>("test_set_estimates") = std::move(testDensities);
    }

    // Print variable importance.
    if (CLI::HasParam("vi"))
    {
      arma::vec importances;
      tree->ComputeVariableImportance(importances);
      CLI::GetParam<arma::mat>("vi") = importances.t();
    }
  }

  if (CLI::HasParam("tag_file"))
  {
    const arma::mat& estimationData =
        CLI::HasParam("test") ? testData : trainingData;
    const string tagFile = CLI::GetParam<string>("tag_file");
    std::ofstream ofs;
    ofs.open(tagFile, std::ofstream::out);

    arma::Row<size_t> counters;

    Timer::Start("det_test_set_tagging");
    if (!ofs.is_open())
    {
      Log::Warn << "Unable to open file '" << tagFile
        << "' to save tag membership info."
        << std::endl;
    }
    else if (CLI::HasParam("path_format"))
    {
      const bool reqCounters = CLI::HasParam("tag_counters_file");
      const string pathFormat = CLI::GetParam<string>("path_format");

      PathCacher::PathFormat theFormat;
      if (pathFormat == "lr" || pathFormat == "LR")
        theFormat = PathCacher::FormatLR;
      else if (pathFormat == "lr-id" || pathFormat == "LR-ID")
        theFormat = PathCacher::FormatLR_ID;
      else if (pathFormat == "id-lr" || pathFormat == "ID-LR")
        theFormat = PathCacher::FormatID_LR;
      else
      {
        Log::Warn << "Unknown path format specified: '" << pathFormat
          << "'. Valid are: lr | lr-id | id-lr. Defaults to 'lr'." << endl;
        theFormat = PathCacher::FormatLR;
      }

      PathCacher path(theFormat, tree);
      counters.zeros(path.NumNodes());

      for (size_t i = 0; i < estimationData.n_cols; i++)
      {
        int tag = tree->FindBucket(estimationData.unsafe_col(i));

        ofs << tag << " " << path.PathFor(tag) << std::endl;
        for (; tag >= 0 && reqCounters; tag = path.ParentOf(tag))
          counters(tag) += 1;
      }

      ofs.close();

      if (reqCounters)
      {
        ofs.open(CLI::GetParam<string>("tag_counters_file"),
                 std::ofstream::out);

        for (size_t j = 0; j < counters.n_elem; ++j)
          ofs << j << " "
              << counters(j) << " "
              << path.PathFor(j) << endl;

        ofs.close();
      }
    }
    else
    {
      int numLeaves = tree->TagTree();
      counters.zeros(numLeaves);

      for (size_t i = 0; i < estimationData.n_cols; i++)
      {
        const int tag = tree->FindBucket(estimationData.unsafe_col(i));

        ofs << tag << std::endl;
        counters(tag) += 1;
      }

      if (CLI::HasParam("tag_counters_file"))
        data::Save(CLI::GetParam<string>("tag_counters_file"), counters);
    }

    Timer::Stop("det_test_set_tagging");
    ofs.close();
  }

  // Save the model, if desired.
  if (CLI::HasParam("output_model"))
    CLI::GetParam<DTree<arma::mat>>("output_model") = std::move(*tree);

  // Clean up memory, if we need to.
  if (!CLI::HasParam("input_model") && !CLI::HasParam("output_model"))
    delete tree;
}
