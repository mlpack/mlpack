/**
 * @file methods/random_forest/random_forest_train_main.cpp
 * @author Ryan Curtin
 * @author Dirk Eddelbuettel
 *
 * Train a random forest object given data.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME random_forest_train

#include <mlpack/core/util/mlpack_main.hpp>
#include "random_forest.hpp"
#include "random_forest_model.hpp"

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

// Program Name.
BINDING_USER_NAME("Random Forests train");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of the standard random forest algorithm by Leo Breiman "
    "for classification.  Given labeled data, a random forest is trained.");

// Long description.
BINDING_LONG_DESC(
    "This program is an implementation of the standard random forest "
    "classification algorithm by Leo Breiman.  A random forest is "
    "trained (and returned for later use for subsequent use "
    "where predictions or class probabilities for points may be generated."
    "\n\n"
    "The training set and associated labels are specified with the " +
    PRINT_PARAM_STRING("training") + " and " + PRINT_PARAM_STRING("labels") +
    " parameters, respectively.  The labels should be in the range `[0, "
    "num_classes - 1]`. Optionally, if " +
    PRINT_PARAM_STRING("labels") + " is not specified, the labels are assumed "
    "to be the last dimension of the training dataset."
    "\n\n"
    "The " + PRINT_PARAM_STRING("minimum_leaf_size") +
    " parameter specifies the minimum number of training points that must fall "
    "into each leaf for it to be split.  The " +
    PRINT_PARAM_STRING("num_trees") +
    " controls the number of trees in the random forest.  The " +
    PRINT_PARAM_STRING("minimum_gain_split") + " parameter controls the minimum"
    " required gain for a decision tree node to split.  Larger values will "
    "force higher-confidence splits.  The " +
    PRINT_PARAM_STRING("maximum_depth") + " parameter specifies "
    "the maximum depth of the tree.  The " +
    PRINT_PARAM_STRING("subspace_dim") + " parameter is used to control the "
    "number of random dimensions chosen for an individual node's split.  If " +
    PRINT_PARAM_STRING("print_training_accuracy") + " is specified, the "
    "calculated accuracy on the training set will be printed.");

// Example.
BINDING_EXAMPLE(
    IMPORT_EXT_LIB() + "\n" +
    IMPORT_SPLIT() + "\n" +
    IMPORT_THIS("random_forest", "train", "classify", "probabilities") + "\n" +
    GET_DATASET("X", "http://datasets.mlpack.org/iris.csv") + "\n" +
    GET_DATASET("y", "http://datasets.mlpack.org/iris_labels.csv") + "\n" +
    SPLIT_TRAIN_TEST("X", "y", "X_train", "y_train", "X_test", "y_test",
    "0.2") + "\n" +
    CREATE_OBJECT("model", "random_forest") + "\n" +
    CALL_METHOD("model", "train", "training", "X_train", "labels", "y_train",
        "minimum_leaf_size", 20, "num_trees", 10,
        "print_training_accuracy", true));

// See also...
BINDING_SEE_ALSO("@decision_tree", "#decision_tree");
BINDING_SEE_ALSO("@hoeffding_tree", "#hoeffding_tree");
BINDING_SEE_ALSO("@softmax_regression", "#softmax_regression");
BINDING_SEE_ALSO("Random forest on Wikipedia",
    "https://en.wikipedia.org/wiki/Random_forest");
BINDING_SEE_ALSO("Random forests (pdf)", "https://www.eecis.udel.edu/~shatkay"
    "/Course/papers/BreimanRandomForests2001.pdf");
BINDING_SEE_ALSO("RandomForest C++ class documentation",
    "@doc/user/methods/random_forest.md");

PARAM_MATRIX_IN_REQ("training", "Training dataset.", "t");
PARAM_UROW_IN_REQ("labels", "Labels for training dataset.", "l");

PARAM_FLAG("print_training_accuracy", "If set, then the accuracy of the model "
    "on the training set will be predicted (verbose must also be specified).",
    "a");

PARAM_INT_IN("num_trees", "Number of trees in the random forest.", "N", 10);
PARAM_INT_IN("minimum_leaf_size", "Minimum number of points in each leaf "
    "node.", "n", 1);
PARAM_INT_IN("maximum_depth", "Maximum depth of the tree (0 means no limit).",
    "D", 0);

PARAM_DOUBLE_IN("minimum_gain_split", "Minimum gain needed to make a split "
    "when building a tree.", "g", 0);
PARAM_INT_IN("subspace_dim", "Dimensionality of random subspace to use for "
    "each split.  '0' will autoselect the square root of data dimensionality.",
    "d", 0);

PARAM_INT_IN("seed", "Random seed.  If 0, 'std::time(NULL)' is used.", "s", 0);

PARAM_MODEL_OUT(RandomForestModel, "output_model", "Model to save trained "
    "random forest to.", "M");

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  // Initialize random seed if needed.
  if (params.Get<int>("seed") != 0)
    RandomSeed((size_t) params.Get<int>("seed"));
  else
    RandomSeed((size_t) std::time(NULL));

  RequireParamValue<int>(params, "num_trees", [](int x) { return x > 0; }, true,
      "number of trees in forest must be positive");

  RequireParamValue<int>(params, "minimum_leaf_size",
      [](int x) { return x > 0; }, true, "minimum leaf size must be greater "
      "than 0");
  RequireParamValue<int>(params, "maximum_depth", [](int x) { return x >= 0; },
      true, "maximum depth must not be negative");
  RequireParamValue<int>(params, "subspace_dim", [](int x) { return x >= 0; },
      true, "subspace dimensionality must be nonnegative");
  RequireParamValue<double>(params, "minimum_gain_split",
      [](double x) { return x >= 0.0; }, true,
      "minimum gain for splitting must be nonnegative");

  RandomForestModel* rfModel = new RandomForestModel();

  timers.Start("rf_training");

  // Train the model on the given input data.
  arma::mat data = std::move(params.Get<arma::mat>("training"));
  arma::Row<size_t> labels =
      std::move(params.Get<arma::Row<size_t>>("labels"));

  // Make sure the subspace dimensionality is valid.
  RequireParamValue<int>(params, "subspace_dim",
      [data](int x) { return (size_t) x <= data.n_rows; }, true, "subspace "
      "dimensionality must not be greater than data dimensionality");

  const size_t numTrees = (size_t) params.Get<int>("num_trees");
  const size_t minimumLeafSize =
      (size_t) params.Get<int>("minimum_leaf_size");
  const size_t maxDepth = (size_t) params.Get<int>("maximum_depth");
  const double minimumGainSplit = params.Get<double>("minimum_gain_split");
  const size_t randomDims = (params.Get<int>("subspace_dim") == 0) ?
      (size_t) std::sqrt(data.n_rows) :
      (size_t) params.Get<int>("subspace_dim");
  MultipleRandomDimensionSelect mrds(randomDims);

  Log::Info << "Training random forest with " << numTrees << " trees..."
      << endl;

  const size_t numClasses = max(labels) + 1;

  // Train the model.
  rfModel->rf.Train(data, labels, numClasses, numTrees, minimumLeafSize,
      minimumGainSplit, maxDepth, false /*params.Has("warm_start")*/, mrds);

  timers.Stop("rf_training");

  // Did we want training accuracy?
  if (params.Has("print_training_accuracy"))
  {
    timers.Start("rf_prediction");
    arma::Row<size_t> predictions;
    rfModel->rf.Classify(data, predictions);

    const size_t correct = accu(predictions == labels);

    Log::Info << correct << " of " << labels.n_elem << " correct on training"
        << " set (" << (double(correct) / double(labels.n_elem) * 100) << ")."
        << endl;
    timers.Stop("rf_prediction");
  }

  // Save the output model.
  params.Get<RandomForestModel*>("output_model") = rfModel;
}
