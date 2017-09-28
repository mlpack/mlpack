/**
 * @file decision_tree_main.cpp
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
#include <mlpack/core/util/cli.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "decision_tree.hpp"

using namespace std;
using namespace mlpack;
using namespace mlpack::tree;

PROGRAM_INFO("Decision tree",
    "Train and evaluate using a decision tree.  Given a dataset containing "
    "numeric features and associated labels for each point in the dataset, this"
    " program can train a decision tree on that data."
    "\n\n"
    "The training file and associated labels are specified with the " +
    PRINT_PARAM_STRING("training") + " and " + PRINT_PARAM_STRING("labels") +
    " parameters, respectively.  The labels should be in the range [0, "
    "num_classes - 1]. Optionally, if " +
    PRINT_PARAM_STRING("labels") + " is not specified, the labels are assumed "
    "to be the last dimension of the training dataset."
    "\n\n"
    "When a model is trained, the " + PRINT_PARAM_STRING("output_model") + " "
    "output parameter may be used to save the trained model.  A model may be "
    "loaded for predictions with the " + PRINT_PARAM_STRING("input_model") +
    "' parameter.  The " + PRINT_PARAM_STRING("input_model") + " parameter "
    "may not be specified when the " + PRINT_PARAM_STRING("training") + " "
    "parameter is specified.  The " + PRINT_PARAM_STRING("minimum_leaf_size") +
    " parameter specifies the minimum number of training points that must fall"
    " into each leaf for it to be split.  If " +
    PRINT_PARAM_STRING("print_training_error") + " is specified, the training "
    "error will be printed."
    "\n\n"
    "Test data may be specified with the " + PRINT_PARAM_STRING("test") + " "
    "parameter, and if performance numbers are desired for that test set, "
    "labels may be specified with the " + PRINT_PARAM_STRING("test_labels") +
    " parameter.  Predictions for each test point may be saved via the " +
    PRINT_PARAM_STRING("predictions") + " output parameter.  Class "
    "probabilities for each prediction may be saved with the " +
    PRINT_PARAM_STRING("probabilities") + " output parameter."
    "\n\n"
    "For example, to train a decision tree with a minimum leaf size of 20 on "
    "the dataset contained in " + PRINT_DATASET("data") + " with labels " +
    PRINT_DATASET("labels") + ", saving the output model to " +
    PRINT_MODEL("tree") + " and printing the training error, one could "
    "call"
    "\n\n" +
    PRINT_CALL("decision_tree", "training", "data", "labels", "labels",
        "output_model", "tree", "minimum_leaf_size", 20,
        "print_training_error", true) +
    "\n\n"
    "Then, to use that model to classify points in " +
    PRINT_DATASET("test_set") + " and print the test error given the "
    "labels " + PRINT_DATASET("test_labels") + " using that model, while "
    "saving the predictions for each point to " +
    PRINT_DATASET("predictions") + ", one could call "
    "\n\n" +
    PRINT_CALL("decision_tree", "input_model", "tree", "test", "test_set",
        "test_labels", "test_labels", "predictions", "predictions"));

// Datasets.
PARAM_MATRIX_IN("training", "Matrix of training points.", "t");
PARAM_UROW_IN("labels", "Training labels.", "l");
PARAM_MATRIX_IN("test", "Matrix of test points.", "T");
PARAM_MATRIX_IN("weights", "The weight of labels", "w");
PARAM_UMATRIX_IN("test_labels", "Test point labels, if accuracy calculation "
    "is desired.", "L");

// Training parameters.
PARAM_INT_IN("minimum_leaf_size", "Minimum number of points in a leaf.", "n",
    20);
PARAM_FLAG("print_training_error", "Print the training error.", "e");

// Output parameters.
PARAM_MATRIX_OUT("probabilities", "Class probabilities for each test point.",
    "P");
PARAM_UROW_OUT("predictions", "Class predictions for each test point.", "p");

/**
 * This is the class that we will serialize.  It is a pretty simple wrapper
 * around DecisionTree<>.  In order to support categoricals, it will need to
 * also hold and serialize a DatasetInfo.
 */
class DecisionTreeModel
{
 public:
  // The tree itself, left public for direct access by this program.
  DecisionTree<> tree;

  // Create the model.
  DecisionTreeModel() { /* Nothing to do. */ }

  // Serialize the model.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(tree, "tree");
  }
};

// Models.
PARAM_MODEL_IN(DecisionTreeModel, "input_model", "Pre-trained decision tree, "
    "to be used with test points.", "m");
PARAM_MODEL_OUT(DecisionTreeModel, "output_model", "Output for trained decision"
    " tree.", "M");

void mlpackMain()
{
  // Check parameters.
  if (CLI::HasParam("training") && CLI::HasParam("input_model"))
    Log::Fatal << "Cannot specify both --training_file and --input_model_file!"
        << endl;

  if (CLI::HasParam("test_labels") && !CLI::HasParam("test"))
    Log::Warn << "--test_labels_file ignored because --test_file is not passed."
        << endl;

  if (!CLI::HasParam("output_model") && !CLI::HasParam("probabilities") &&
      !CLI::HasParam("predictions") && !CLI::HasParam("test_labels"))
    Log::Warn << "None of --output_model_file, --probabilities_file, or "
        << "--predictions_file are given, and accuracy is not being calculated;"
        << " no output will be saved!" << endl;

  if (CLI::HasParam("print_training_error") && !CLI::HasParam("training"))
    Log::Warn << "--print_training_error ignored because --training_file is not"
        << " specified." << endl;

  if (!CLI::HasParam("test"))
  {
    if (CLI::HasParam("probabilities"))
      Log::Warn << "--probabilities_file ignored because --test_file is not "
          << "specified." << endl;
    if (CLI::HasParam("predictions"))
      Log::Warn << "--predictions_file ignored because --test_file is not "
          << "specified." << endl;
  }

  // Load the model or build the tree.
  DecisionTreeModel model;

  if (CLI::HasParam("training"))
  {
    arma::mat dataset = std::move(CLI::GetParam<arma::mat>("training"));
    arma::Row<size_t> labels;
    if (CLI::HasParam("labels"))
    {
      labels = std::move(CLI::GetParam<arma::Row<size_t>>("labels"));
    }
    else
    {
      // Extract the labels as the last
      Log::Info << "Using the last dimension of training set as labels."
          << endl;
      labels = arma::conv_to<arma::Row<size_t>>::from(
          dataset.row(dataset.n_rows - 1));
      dataset.shed_row(dataset.n_rows - 1);
    }

    // Calculate number of classes.
    const size_t numClasses = arma::max(arma::max(labels)) + 1;

    // Now build the tree.
    const size_t minLeafSize = (size_t) CLI::GetParam<int>("minimum_leaf_size");

    // Create decision tree with weighted labels.
    if (CLI::HasParam("weights"))
    {
      arma::Row<double> weights =
          std::move(CLI::GetParam<arma::Mat<double>>("weights"));
      model.tree = DecisionTree<>(dataset, labels, numClasses,
          weights, minLeafSize);
    }
    else
    {
      model.tree = DecisionTree<>(dataset, labels, numClasses, minLeafSize);
    }

    // Do we need to print training error?
    if (CLI::HasParam("print_training_error"))
    {
      arma::Row<size_t> predictions;
      arma::mat probabilities;

      model.tree.Classify(dataset, predictions, probabilities);

      size_t correct = 0;
      for (size_t i = 0; i < dataset.n_cols; ++i)
        if (predictions[i] == labels[i])
          ++correct;

      // Print number of correct points.
      Log::Info << double(correct) / double(dataset.n_cols) * 100 << "%% "
          << "correct on training set (" << correct << " / " << dataset.n_cols
          << ")." << endl;
    }
  }
  else
  {
    model = std::move(CLI::GetParam<DecisionTreeModel>("input_model"));
  }

  // Do we need to get predictions?
  if (CLI::HasParam("test"))
  {
    arma::mat testPoints = std::move(CLI::GetParam<arma::mat>("test"));

    arma::Row<size_t> predictions;
    arma::mat probabilities;

    model.tree.Classify(testPoints, predictions, probabilities);

    // Do we need to calculate accuracy?
    if (CLI::HasParam("test_labels"))
    {
      arma::Row<size_t> testLabels =
          std::move(CLI::GetParam<arma::Row<size_t>>("test_labels"));

      size_t correct = 0;
      for (size_t i = 0; i < testPoints.n_cols; ++i)
        if (predictions[i] == testLabels[i])
          ++correct;

      // Print number of correct points.
      Log::Info << double(correct) / double(testPoints.n_cols) * 100 << "%% "
          << "correct on test set (" << correct << " / " << testPoints.n_cols
          << ")." << endl;
    }

    // Do we need to save outputs?
    if (CLI::HasParam("predictions"))
      CLI::GetParam<arma::Row<size_t>>("predictions") = std::move(predictions);
    if (CLI::HasParam("probabilities"))
      CLI::GetParam<arma::mat>("probabilities") = std::move(probabilities);
  }

  // Do we need to save the model?
  if (CLI::HasParam("output_model"))
    CLI::GetParam<DecisionTreeModel>("output_model") = std::move(model);
}
