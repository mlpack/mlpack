/**
 * @file methods/decision_tree/decision_tree_main.cpp
 * @author Ryan Curtin
 *
 * A command-line program to build a decision tree.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME decision_tree

#include <mlpack/core/util/mlpack_main.hpp>
#include "decision_tree.hpp"

using namespace std;
using namespace mlpack;
using namespace mlpack::data;
using namespace mlpack::util;

// Program Name.
BINDING_USER_NAME("Decision tree");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of an ID3-style decision tree for classification, which"
    " supports categorical data.  Given labeled data with numeric or "
    "categorical features, a decision tree can be trained and saved; or, an "
    "existing decision tree can be used for classification on new points.");

// Long description.
BINDING_LONG_DESC(
    "Train and evaluate using a decision tree.  Given a dataset containing "
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
    "When a model is trained, the " + PRINT_PARAM_STRING("output_model") + " "
    "output parameter may be used to save the trained model.  A model may be "
    "loaded for predictions with the " + PRINT_PARAM_STRING("input_model") +
    " parameter.  The " + PRINT_PARAM_STRING("input_model") + " parameter "
    "may not be specified when the " + PRINT_PARAM_STRING("training") + " "
    "parameter is specified.  The " + PRINT_PARAM_STRING("minimum_leaf_size") +
    " parameter specifies the minimum number of training points that must fall"
    " into each leaf for it to be split.  The " +
    PRINT_PARAM_STRING("minimum_gain_split") + " parameter specifies "
    "the minimum gain that is needed for the node to split.  The " +
    PRINT_PARAM_STRING("maximum_depth") + " parameter specifies "
    "the maximum depth of the tree.  If " +
    PRINT_PARAM_STRING("print_training_accuracy") + " is specified, the "
    "training accuracy will be printed."
    "\n\n"
    "Test data may be specified with the " + PRINT_PARAM_STRING("test") + " "
    "parameter, and if performance numbers are desired for that test set, "
    "labels may be specified with the " + PRINT_PARAM_STRING("test_labels") +
    " parameter.  Predictions for each test point may be saved via the " +
    PRINT_PARAM_STRING("predictions") + " output parameter.  Class "
    "probabilities for each prediction may be saved with the " +
    PRINT_PARAM_STRING("probabilities") + " output parameter.");

// Example.
BINDING_EXAMPLE(
    "For example, to train a decision tree with a minimum leaf size of 20 on "
    "the dataset contained in " + PRINT_DATASET("data") + " with labels " +
    PRINT_DATASET("labels") + ", saving the output model to " +
    PRINT_MODEL("tree") + " and printing the training error, one could "
    "call"
    "\n\n" +
    PRINT_CALL("decision_tree", "training", "data", "labels", "labels",
        "output_model", "tree", "minimum_leaf_size", 20, "minimum_gain_split",
        1e-3, "print_training_accuracy", true) +
    "\n\n"
    "Then, to use that model to classify points in " +
    PRINT_DATASET("test_set") + " and print the test error given the "
    "labels " + PRINT_DATASET("test_labels") + " using that model, while "
    "saving the predictions for each point to " +
    PRINT_DATASET("predictions") + ", one could call "
    "\n\n" +
    PRINT_CALL("decision_tree", "input_model", "tree", "test", "test_set",
        "test_labels", "test_labels", "predictions", "predictions"));

// See also...
BINDING_SEE_ALSO("Random forest", "#random_forest");
BINDING_SEE_ALSO("Decision trees on Wikipedia",
    "https://en.wikipedia.org/wiki/Decision_tree_learning");
BINDING_SEE_ALSO("Induction of Decision Trees (pdf)",
    "https://www.hunch.net/~coms-4771/quinlan.pdf");
BINDING_SEE_ALSO("DecisionTree C++ class documentation",
    "@doc/user/methods/decision_tree.md");

// Datasets.
PARAM_MATRIX_AND_INFO_IN("training", "Training dataset (may be categorical).",
    "t");
PARAM_UROW_IN("labels", "Training labels.", "l");
PARAM_MATRIX_AND_INFO_IN("test", "Testing dataset (may be categorical).", "T");
PARAM_MATRIX_IN("weights", "The weight of labels", "w");
PARAM_UROW_IN("test_labels", "Test point labels, if accuracy calculation "
    "is desired.", "L");

// Training parameters.
PARAM_INT_IN("minimum_leaf_size", "Minimum number of points in a leaf.", "n",
    20);
PARAM_DOUBLE_IN("minimum_gain_split", "Minimum gain for node splitting.", "g",
    1e-7);
PARAM_INT_IN("maximum_depth", "Maximum depth of the tree (0 means no limit).",
    "D", 0);
PARAM_FLAG("print_training_accuracy", "Print the training accuracy.", "a");

// Output parameters.
PARAM_MATRIX_OUT("probabilities", "Class probabilities for each test point.",
    "P");
PARAM_UROW_OUT("predictions", "Class predictions for each test point.", "p");

/**
 * This is the class that we will serialize.  It is a pretty simple wrapper
 * around DecisionTree<>.
 */
class DecisionTreeModel
{
 public:
  // The tree itself, left public for direct access by this program.
  DecisionTree<> tree;
  DatasetInfo info;

  // Create the model.
  DecisionTreeModel() { /* Nothing to do. */ }

  // Serialize the model.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(tree));
    ar(CEREAL_NVP(info));
  }
};

// Models.
PARAM_MODEL_IN(DecisionTreeModel, "input_model", "Pre-trained decision tree, "
    "to be used with test points.", "m");
PARAM_MODEL_OUT(DecisionTreeModel, "output_model", "Output for trained decision"
    " tree.", "M");

// Convenience typedef.
using TupleType = tuple<DatasetInfo, arma::mat>;

void BINDING_FUNCTION(util::Params& params, util::Timers& /* timers */)
{
  // Check parameters.
  RequireOnlyOnePassed(params, { "training", "input_model" }, true);
  ReportIgnoredParam(params, {{ "test", false }}, "test_labels");
  RequireAtLeastOnePassed(params, { "output_model", "probabilities",
      "predictions" }, false, "no output will be saved");
  ReportIgnoredParam(params, {{ "training", false }},
      "print_training_accuracy");

  ReportIgnoredParam(params, {{ "test", false }}, "predictions");
  ReportIgnoredParam(params, {{ "test", false }}, "predictions");

  RequireParamValue<int>(params, "minimum_leaf_size",
      [](int x) { return x > 0; }, true, "leaf size must be positive");

  RequireParamValue<int>(params, "maximum_depth",
      [](int x) { return x >= 0; }, true, "maximum depth must not be negative");

  RequireParamValue<double>(params, "minimum_gain_split",
      [](double x) { return (x > 0.0 && x < 1.0); }, true,
      "gain split must be a fraction in range [0,1]");

  // Load the model or build the tree.
  DecisionTreeModel* model;
  arma::mat trainingSet;
  arma::Row<size_t> labels;

  if (params.Has("training"))
  {
    model = new DecisionTreeModel();
    model->info = std::move(std::get<0>(params.Get<TupleType>("training")));
    trainingSet = std::move(std::get<1>(params.Get<TupleType>("training")));
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
        model->tree = DecisionTree<>(trainingSet, model->info, labels,
            numClasses, std::move(weights), minLeafSize, minimumGainSplit,
            maxDepth);
      }
      else
      {
        model->tree = DecisionTree<>(std::move(trainingSet), model->info,
            std::move(labels), numClasses, std::move(weights), minLeafSize,
            minimumGainSplit, maxDepth);
      }
    }
    else
    {
      if (params.Has("print_training_accuracy"))
      {
        model->tree = DecisionTree<>(trainingSet, model->info, labels,
            numClasses, minLeafSize, minimumGainSplit, maxDepth);
      }
      else
      {
        model->tree = DecisionTree<>(std::move(trainingSet), model->info,
            std::move(labels), numClasses, minLeafSize, minimumGainSplit,
            maxDepth);
      }
    }

    // Do we need to print training error?
    if (params.Has("print_training_accuracy"))
    {
      arma::Row<size_t> predictions;
      arma::mat probabilities;

      model->tree.Classify(trainingSet, predictions, probabilities);

      size_t correct = 0;
      for (size_t i = 0; i < trainingSet.n_cols; ++i)
        if (predictions[i] == labels[i])
          ++correct;

      // Print number of correct points.
      Log::Info << double(correct) / double(trainingSet.n_cols) * 100 << "% "
          << "correct on training set (" << correct << " / "
          << trainingSet.n_cols << ")." << endl;
    }
  }
  else
  {
    model = params.Get<DecisionTreeModel*>("input_model");
  }

  // Do we need to get predictions?
  if (params.Has("test"))
  {
    std::get<0>(params.GetRaw<TupleType>("test")) = model->info;
    arma::mat testPoints = std::get<1>(params.Get<TupleType>("test"));

    arma::Row<size_t> predictions;
    arma::mat probabilities;

    model->tree.Classify(testPoints, predictions, probabilities);

    // Do we need to calculate accuracy?
    if (params.Has("test_labels"))
    {
      arma::Row<size_t> testLabels =
          std::move(params.Get<arma::Row<size_t>>("test_labels"));

      size_t correct = 0;
      for (size_t i = 0; i < testPoints.n_cols; ++i)
        if (predictions[i] == testLabels[i])
          ++correct;

      // Print number of correct points.
      Log::Info << double(correct) / double(testPoints.n_cols) * 100 << "% "
          << "correct on test set (" << correct << " / " << testPoints.n_cols
          << ")." << endl;
    }

    // Do we need to save outputs?
    params.Get<arma::Row<size_t>>("predictions") = predictions;
    params.Get<arma::mat>("probabilities") = probabilities;
  }

  // Do we need to save the model?
  params.Get<DecisionTreeModel*>("output_model") = model;
}
