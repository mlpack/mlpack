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
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/io.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "decision_tree.hpp"

using namespace std;
using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::data;
using namespace mlpack::util;

// Program Name.
BINDING_NAME("Decision tree");

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
    " parameters, respectively.  The labels should be in the range [0, "
    "num_classes - 1]. Optionally, if " +
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
    PRINT_PARAM_STRING("print_training_error") + " is specified, the training "
    "error will be printed."
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
BINDING_SEE_ALSO("Decision stump", "#decision_stump");
BINDING_SEE_ALSO("Random forest", "#random_forest");
BINDING_SEE_ALSO("Decision trees on Wikipedia",
        "https://en.wikipedia.org/wiki/Decision_tree_learning");
BINDING_SEE_ALSO("Induction of Decision Trees (pdf)",
        "https://link.springer.com/content/pdf/10.1007/BF00116251.pdf");
BINDING_SEE_ALSO("mlpack::tree::DecisionTree class documentation",
        "@doxygen/classmlpack_1_1tree_1_1DecisionTree.html");

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
// This is deprecated and should be removed in mlpack 4.0.0.
PARAM_FLAG("print_training_error", "Print the training error (deprecated; will "
      "be removed in mlpack 4.0.0).", "e");
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
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & BOOST_SERIALIZATION_NVP(tree);
    ar & BOOST_SERIALIZATION_NVP(info);
  }
};

// Models.
PARAM_MODEL_IN(DecisionTreeModel, "input_model", "Pre-trained decision tree, "
    "to be used with test points.", "m");
PARAM_MODEL_OUT(DecisionTreeModel, "output_model", "Output for trained decision"
    " tree.", "M");

// Convenience typedef.
typedef tuple<DatasetInfo, arma::mat> TupleType;

static void mlpackMain()
{
  // Check parameters.
  RequireOnlyOnePassed({ "training", "input_model" }, true);
  ReportIgnoredParam({{ "test", false }}, "test_labels");
  RequireAtLeastOnePassed({ "output_model", "probabilities", "predictions" },
      false, "no output will be saved");
  ReportIgnoredParam({{ "training", false }}, "print_training_accuracy");

  ReportIgnoredParam({{ "test", false }}, "predictions");
  ReportIgnoredParam({{ "test", false }}, "predictions");

  RequireParamValue<int>("minimum_leaf_size", [](int x) { return x > 0; }, true,
      "leaf size must be positive");

  RequireParamValue<int>("maximum_depth", [](int x) { return x >= 0; }, true,
      "maximum depth must not be negative");

  RequireParamValue<double>("minimum_gain_split", [](double x)
                         { return (x > 0.0 && x < 1.0); }, true,
                         "gain split must be a fraction in range [0,1]");

  if (IO::HasParam("print_training_error"))
  {
    Log::Warn << "The option " << PRINT_PARAM_STRING("print_training_error")
        << " is deprecated and will be removed in mlpack 4.0.0." << std::endl;
  }

  // Load the model or build the tree.
  DecisionTreeModel* model;
  arma::mat trainingSet;
  arma::Row<size_t> labels;

  if (IO::HasParam("training"))
  {
    model = new DecisionTreeModel();
    model->info = std::move(std::get<0>(IO::GetParam<TupleType>("training")));
    trainingSet = std::move(std::get<1>(IO::GetParam<TupleType>("training")));
    if (IO::HasParam("labels"))
    {
      labels = std::move(IO::GetParam<arma::Row<size_t>>("labels"));
    }
    else
    {
      // Extract the labels as the last
      Log::Info << "Using the last dimension of training set as labels."
          << endl;
      labels = arma::conv_to<arma::Row<size_t>>::from(
          trainingSet.row(trainingSet.n_rows - 1));
      trainingSet.shed_row(trainingSet.n_rows - 1);
    }

    const size_t numClasses = arma::max(arma::max(labels)) + 1;

    // Now build the tree.
    const size_t minLeafSize = (size_t) IO::GetParam<int>("minimum_leaf_size");
    const size_t maxDepth = (size_t) IO::GetParam<int>("maximum_depth");
    const double minimumGainSplit =
                           (double) IO::GetParam<double>("minimum_gain_split");

    // Create decision tree with weighted labels.
    if (IO::HasParam("weights"))
    {
      arma::Row<double> weights =
          std::move(IO::GetParam<arma::Mat<double>>("weights"));
      if (IO::HasParam("print_training_error") ||
          IO::HasParam("print_training_accuracy"))
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
      if (IO::HasParam("print_training_error"))
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
    if (IO::HasParam("print_training_error") ||
        IO::HasParam("print_training_accuracy"))
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
    model = IO::GetParam<DecisionTreeModel*>("input_model");
  }

  // Do we need to get predictions?
  if (IO::HasParam("test"))
  {
    std::get<0>(IO::GetRawParam<TupleType>("test")) = model->info;
    arma::mat testPoints = std::get<1>(IO::GetParam<TupleType>("test"));

    arma::Row<size_t> predictions;
    arma::mat probabilities;

    model->tree.Classify(testPoints, predictions, probabilities);

    // Do we need to calculate accuracy?
    if (IO::HasParam("test_labels"))
    {
      arma::Row<size_t> testLabels =
          std::move(IO::GetParam<arma::Row<size_t>>("test_labels"));

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
    IO::GetParam<arma::Row<size_t>>("predictions") = predictions;
    IO::GetParam<arma::mat>("probabilities") = probabilities;
  }

  // Do we need to save the model?
  IO::GetParam<DecisionTreeModel*>("output_model") = model;
}
