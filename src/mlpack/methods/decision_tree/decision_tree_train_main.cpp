/**
 * @file methods/decision_tree/decision_tree_train_main.cpp
 * @author Ryan Curtin
 * @author Dirk Eddelbuettel
 *
 * Implementation of building decision tree for training,
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME decision_tree_train

#include <mlpack/core/util/mlpack_main.hpp>
#include "decision_tree.hpp"
#include "decision_tree_model.hpp"

using namespace std;
using namespace mlpack;
using namespace mlpack::util;

// Program Name.
BINDING_USER_NAME("Decision tree training");

// Short description.
BINDING_SHORT_DESC(
    "Training ID3-style decision tree model.");

// Long description.
BINDING_LONG_DESC(
    "Train using a decision tree.  Given a dataset containing "
    "numeric or categorical features, and associated labels for each point in "
    "the dataset, this program can train a decision tree on that data."
    "\n\n"
    "The training set and associated labels are specified with the " +
    PRINT_PARAM_STRING("training") + " and " + PRINT_PARAM_STRING("labels") +
    " parameters, respectively.  The labels should be in the range `[0, "
    "num_classes - 1]`. Optionally, if " +
    PRINT_PARAM_STRING("labels") + " is not specified, the labels are assumed "
    "to be the last dimension of the training dataset."
    "\n\n"
    "The trained model is returned, and can then be used for prediction."
    " The " + PRINT_PARAM_STRING("minimum_leaf_size") +
    " parameter specifies the minimum number of training points that must fall"
    " into each leaf for it to be split.  The " +
    PRINT_PARAM_STRING("minimum_gain_split") + " parameter specifies "
    "the minimum gain that is needed for the node to split.  The " +
    PRINT_PARAM_STRING("maximum_depth") + " parameter specifies "
    "the maximum depth of the tree.  If " +
    PRINT_PARAM_STRING("print_training_accuracy") + " is specified, the "
    "training accuracy will be printed.");

// Example.
BINDING_EXAMPLE(
    IMPORT_EXT_LIB() + "\n" +
    IMPORT_SPLIT() + "\n" +
    IMPORT_THIS("decision_tree", "train", "classify", "probabilities") + "\n" +
    GET_DATASET("X", "http://datasets.mlpack.org/iris.csv") + "\n" +
    GET_DATASET("y", "http://datasets.mlpack.org/iris_labels.csv") + "\n" +
    SPLIT_TRAIN_TEST("X", "y", "X_train", "y_train", "X_test", "y_test",
    "0.2") + "\n" +
    CREATE_OBJECT("model", "decision_tree") + "\n" +
    CALL_METHOD("model", "train", "training", "X_train", "labels", "y_train",
                "minimum_leaf_size", 20, "minimum_gain_split", 1e-3));

// See also...
BINDING_SEE_ALSO("Random forest", "#random_forest");
BINDING_SEE_ALSO("Decision trees on Wikipedia",
    "https://en.wikipedia.org/wiki/Decision_tree_learning");
BINDING_SEE_ALSO("Induction of Decision Trees (pdf)",
    "https://www.hunch.net/~coms-4771/quinlan.pdf");
BINDING_SEE_ALSO("DecisionTree C++ class documentation",
    "@doc/user/methods/decision_tree.md");

// Datasets.
PARAM_MATRIX_AND_INFO_IN_REQ("training", "Training dataset (may contain "
    "categorical variables).", "t");
PARAM_UROW_IN("labels", "Training labels.", "l");
PARAM_MATRIX_IN("weights", "The weight of labels", "w");

// Training parameters.
PARAM_INT_IN("minimum_leaf_size", "Minimum number of points in a leaf.", "n",
    20);
PARAM_DOUBLE_IN("minimum_gain_split", "Minimum gain for node splitting.", "g",
    1e-7);
PARAM_INT_IN("maximum_depth", "Maximum depth of the tree (0 means no limit).",
    "D", 0);
PARAM_FLAG("print_training_accuracy", "Print the training accuracy.", "a");

// Models.
PARAM_MODEL_OUT(DecisionTreeModel, "output_model", "Output for trained decision"
    " tree.", "M");

// Convenience typedef.
using TupleType = tuple<DatasetInfo, arma::mat>;

void BINDING_FUNCTION(util::Params& params, util::Timers& /* timers */)
{
  // Check parameters.
  RequireParamValue<int>(params, "minimum_leaf_size",
      [](int x) { return x > 0; }, true, "leaf size must be positive");

  RequireParamValue<int>(params, "maximum_depth",
      [](int x) { return x >= 0; }, true, "maximum depth must not be negative");

  RequireParamValue<double>(params, "minimum_gain_split",
      [](double x) { return (x > 0.0 && x < 1.0); }, true,
      "gain split must be a fraction in range [0,1]");

  // Load the model or build the tree.
  DecisionTreeModel* model = new DecisionTreeModel();
  model->info = std::move(std::get<0>(params.Get<TupleType>("training")));
  arma::mat trainingSet =
    std::move(std::get<1>(params.Get<TupleType>("training")));
  arma::Row<size_t> labels;
  if (params.Has("labels"))
  {
    labels = std::move(params.Get<arma::Row<size_t>>("labels"));
  }
  else
  {
    // Extract the labels as the last
    Log::Info << "Using the last dimension of training set as labels."
        << endl;
    labels = ConvTo<arma::Row<size_t>>::From(
        trainingSet.row(trainingSet.n_rows - 1));
    trainingSet.shed_row(trainingSet.n_rows - 1);
  }

  const size_t numClasses = max(labels) + 1;

  // Now build the tree.
  const size_t minLeafSize = (size_t) params.Get<int>("minimum_leaf_size");
  const size_t maxDepth = (size_t) params.Get<int>("maximum_depth");
  const double minimumGainSplit =
                         (double) params.Get<double>("minimum_gain_split");

  // Create decision tree with weighted labels.
  if (params.Has("weights"))
  {
    arma::Row<double> weights =
        std::move(params.Get<arma::Mat<double>>("weights"));
    if (params.Has("print_training_accuracy"))
    {
      model->tree.Train(trainingSet, labels, numClasses, std::move(weights),
          minLeafSize, minimumGainSplit, maxDepth);
    }
    else
    {
      model->tree.Train(std::move(trainingSet), std::move(labels), numClasses,
          std::move(weights), minLeafSize, minimumGainSplit, maxDepth);
    }
  }
  else
  {
    if (params.Has("print_training_accuracy"))
    {
      model->tree.Train(trainingSet, labels, numClasses, minLeafSize,
          minimumGainSplit, maxDepth);
    }
    else
    {
      model->tree.Train(std::move(trainingSet), std::move(labels), numClasses,
          minLeafSize, minimumGainSplit, maxDepth);
    }
  }

  // Do we need to print training error?
  if (params.Has("print_training_accuracy"))
  {
    arma::Row<size_t> predictions;
    model->tree.Classify(trainingSet, predictions);

    size_t correct = 0;
    for (size_t i = 0; i < trainingSet.n_cols; ++i)
      if (predictions[i] == labels[i])
        ++correct;

    // Print number of correct points.
    Log::Info << double(correct) / double(trainingSet.n_cols) * 100 << "% "
        << "correct on training set (" << correct << " / "
        << trainingSet.n_cols << ")." << endl;
  }

  // Return the trained model.
  params.Get<DecisionTreeModel*>("output_model") = model;
}
