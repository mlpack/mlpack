/**
 * @file methods/det/det_main.cpp
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

#undef BINDING_NAME
#define BINDING_NAME det

#include <mlpack/core/util/mlpack_main.hpp>
#include "dt_utils.hpp"

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

// Program Name.
BINDING_USER_NAME("Density Estimation With Density Estimation Trees");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of density estimation trees for the density estimation "
    "task.  Density estimation trees can be trained or used to predict the "
    "density at locations given by query points.");

// Long description.
BINDING_LONG_DESC(
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
    "Enabling path printing for each node outputs the path from the root node "
    "to a leaf for each entry in the test set, or training set (if a test set "
    "is not provided).  Strings like 'LRLRLR' (indicating that traversal went "
    "to the left child, then the right child, then the left child, and so "
    "forth) will be output. If 'lr-id' or 'id-lr' are given as the " +
    PRINT_PARAM_STRING("path_format") + " parameter, then the ID (tag) of "
    "every node along the path will be printed after or before the L or R "
    "character indicating the direction of traversal, respectively."
    "\n\n"
    "This program also can provide density estimates for a set of test points, "
    "specified in the " + PRINT_PARAM_STRING("test") + " parameter.  The "
    "density estimation tree used for this task will be the tree that was "
    "trained on the given training points, or a tree given as the parameter " +
    PRINT_PARAM_STRING("input_model") + ".  The density estimates for the test"
    " points may be saved using the " +
    PRINT_PARAM_STRING("test_set_estimates") + " output parameter.");

// See also...
BINDING_SEE_ALSO("Density estimation on Wikipedia",
    "https://en.wikipedia.org/wiki/Density_estimation");
BINDING_SEE_ALSO("Density estimation trees (pdf)",
    "https://www.mlpack.org/papers/det.pdf");
BINDING_SEE_ALSO("DTree class documentation",
    "@src/mlpack/methods/det/dtree.hpp");

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
PARAM_STRING_IN("path_format", "The format of path printing: 'lr', 'id-lr', or "
    "'lr-id'.", "p", "lr");

PARAM_STRING_OUT("tag_counters_file", "The file to output the number of points "
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

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  // Validate input parameters.
  RequireOnlyOnePassed(params, { "training", "input_model" }, true);

  ReportIgnoredParam(params, {{ "training", false }}, "training_set_estimates");
  ReportIgnoredParam(params, {{ "training", false }}, "folds");
  ReportIgnoredParam(params, {{ "training", false }}, "min_leaf_size");
  ReportIgnoredParam(params, {{ "training", false }}, "max_leaf_size");

  if (params.Has("tag_file"))
    RequireAtLeastOnePassed(params, { "training", "test" }, true);

  if (params.Has("training"))
  {
    RequireAtLeastOnePassed(params, { "output_model", "training_set_estimates",
        "vi", "tag_file", "tag_counters_file" }, false,
        "no output will be saved");
  }

  ReportIgnoredParam(params, {{ "test", false }}, "test_set_estimates");

  RequireParamValue<int>(params, "folds", [](int x) { return x >= 0; }, true,
      "folds must be non-negative");
  RequireParamValue<int>(params, "max_leaf_size", [](int x) { return x > 0; },
      true, "maximum leaf size must be positive");
  RequireParamValue<int>(params, "min_leaf_size", [](int x) { return x > 0; },
      true, "minimum leaf size must be positive");

  // Are we training a DET or loading from file?
  DTree<arma::mat, int>* tree;
  arma::mat trainingData;
  arma::mat testData;

  if (params.Has("training"))
  {
    trainingData = std::move(params.Get<arma::mat>("training"));

    const bool regularization = false;
//    const bool regularization = params.Has("volume_regularization");
    const int maxLeafSize = params.Get<int>("max_leaf_size");
    const int minLeafSize = params.Get<int>("min_leaf_size");
    const bool skipPruning = params.Has("skip_pruning");
    size_t folds = params.Get<int>("folds");

    if (folds == 0)
      folds = trainingData.n_cols;

    // Obtain the optimal tree.
    timers.Start("det_training");
    tree = Trainer<arma::mat, int>(trainingData, folds, regularization,
                                   maxLeafSize, minLeafSize,
                                   skipPruning, timers);
    timers.Stop("det_training");

    // Compute training set estimates, if desired.
    if (params.Has("training_set_estimates"))
    {
      // Compute density estimates for each point in the training set.
      arma::rowvec trainingDensities(trainingData.n_cols);
      timers.Start("det_estimation_time");
      for (size_t i = 0; i < trainingData.n_cols; ++i)
        trainingDensities[i] = tree->ComputeValue(trainingData.unsafe_col(i));
      timers.Stop("det_estimation_time");

      params.Get<arma::mat>("training_set_estimates") =
          std::move(trainingDensities);
    }
  }
  else
  {
    tree = params.Get<DTree<arma::mat>*>("input_model");
  }

  // Compute the density at the provided test points and output the density in
  // the given file.
  if (params.Has("test"))
  {
    testData = std::move(params.Get<arma::mat>("test"));
    if (params.Has("test_set_estimates"))
    {
      // Compute test set densities.
      timers.Start("det_test_set_estimation");
      arma::rowvec testDensities(testData.n_cols);

      for (size_t i = 0; i < testData.n_cols; ++i)
        testDensities[i] = tree->ComputeValue(testData.unsafe_col(i));

      timers.Stop("det_test_set_estimation");

      params.Get<arma::mat>("test_set_estimates") = std::move(testDensities);
    }

    // Print variable importance.
    if (params.Has("vi"))
    {
      arma::vec importances;
      tree->ComputeVariableImportance(importances);
      params.Get<arma::mat>("vi") = importances.t();
    }
  }

  if (params.Has("tag_file"))
  {
    const arma::mat& estimationData =
        params.Has("test") ? testData : trainingData;
    const string tagFile = params.Get<string>("tag_file");
    std::ofstream ofs;
    ofs.open(tagFile, std::ofstream::out);

    arma::Row<size_t> counters;

    timers.Start("det_test_set_tagging");
    if (!ofs.is_open() && tagFile != "")
    {
      Log::Warn << "Unable to open file '" << tagFile
          << "' to save tag membership info." << std::endl;
    }
    else if (params.Has("path_format"))
    {
      const bool reqCounters = params.Has("tag_counters_file");
      const string pathFormat = params.Get<string>("path_format");

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

      for (size_t i = 0; i < estimationData.n_cols; ++i)
      {
        int tag = tree->FindBucket(estimationData.unsafe_col(i));

        ofs << tag << " " << path.PathFor(tag) << std::endl;
        for (; tag >= 0 && reqCounters; tag = path.ParentOf(tag))
          counters(tag) += 1;
      }

      ofs.close();

      if (reqCounters)
      {
        ofs.open(params.Get<string>("tag_counters_file"),
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

      for (size_t i = 0; i < estimationData.n_cols; ++i)
      {
        const int tag = tree->FindBucket(estimationData.unsafe_col(i));

        ofs << tag << std::endl;
        counters(tag) += 1;
      }

      if (params.Has("tag_counters_file"))
        data::Save(params.Get<string>("tag_counters_file"), counters);
    }

    timers.Stop("det_test_set_tagging");
    ofs.close();
  }

  // Save the model, if desired.
  params.Get<DTree<arma::mat>*>("output_model") = tree;
}
