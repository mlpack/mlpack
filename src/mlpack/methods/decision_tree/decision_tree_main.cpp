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
#include <mlpack/core.hpp>
#include "decision_tree.hpp"
#include <mlpack/core/data/load.hpp>
#include <mlpack/core/data/save.hpp>

using namespace std;
using namespace mlpack;
using namespace mlpack::tree;

PROGRAM_INFO("Decision tree",
    "Train and evaluate using a decision tree.  Given a dataset containing "
    "numeric features and associated labels for each point in the dataset, this"
    " program can train a decision tree on that data."
    "\n\n"
    "The training file and associated labels are specified with the "
    "--training_file and --labels_file options, respectively.  The labels "
    "should be in the range [0, num_classes - 1]."
    "\n\n"
    "When a model is trained, it may be saved to file with the "
    "--output_model_file (-M) option.  A model may be loaded from file for "
    "predictions with the --input_model_file (-m) option.  The "
    "--input_model_file option may not be specified when the --training_file "
    "option is specified.  The --minimum_leaf_size (-n) parameter specifies "
    "the minimum number of training points that must fall into each leaf for "
    "it to be split.  If --print_training_error (-e) is specified, the training"
    " error will be printed."
    "\n\n"
    "A file containing test data may be specified with the --test_file (-T) "
    "option, and if performance numbers are desired for that test set, labels "
    "may be specified with the --test_labels_file (-L) option.  Predictions f"
    "for each test point may be stored into the file specified by the "
    "--predictions_file (-p) option.  Class probabilities for each prediction "
    "will be stored in the file specified by the --probabilities_file (-P) "
    "option.");

// Datasets.
PARAM_STRING_IN("training_file", "File containing training points.", "t", "");
PARAM_STRING_IN("labels_file", "File containing training labels.", "l", "");
PARAM_STRING_IN("test_file", "File containing test points.", "T", "");
PARAM_STRING_IN("test_labels_file", "File containing test labels, if accuracy "
    "calculation is desired.", "L", "");

// Training parameters.
PARAM_INT_IN("minimum_leaf_size", "Minimum number of points in a leaf.", "n",
    20);
PARAM_FLAG("print_training_error", "Print the training error.", "e");

// Output parameters.
PARAM_STRING_IN("probabilities_file", "File to save class probabilities to for"
    " each test point.", "P", "");
PARAM_STRING_IN("predictions_file", "File to save class predictions to for "
    "each test point.", "p", "");

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
PARAM_STRING_IN("input_model_file", "File to load pre-trained decision tree "
    "from, to be used with test points.", "m", "");
PARAM_STRING_IN("output_model_file", "File to save trained decision tree to.",
    "M", "");

int main(int argc, char** argv)
{
  CLI::ParseCommandLine(argc, argv);

  // Check parameters.
  if (CLI::HasParam("training_file") && CLI::HasParam("input_model_file"))
    Log::Fatal << "Cannot specify both --training_file and --input_model_file!"
        << endl;

  if (CLI::HasParam("training_file") && !CLI::HasParam("labels_file"))
    Log::Fatal << "Must specify --labels_file when --training_file is "
        << "specified!" << endl;

  if (CLI::HasParam("test_labels_file") && !CLI::HasParam("test_file"))
    Log::Warn << "--test_labels_file ignored because --test_file is not passed."
        << endl;

  if (!CLI::HasParam("output_model_file") &&
      !CLI::HasParam("probabilities_file") &&
      !CLI::HasParam("predictions_file") &&
      !CLI::HasParam("test_labels_file"))
    Log::Warn << "None of --output_model_file, --probabilities_file, or "
        << "--predictions_file are given, and accuracy is not being calculated;"
        << " no output will be saved!" << endl;

  if (CLI::HasParam("print_training_error") && !CLI::HasParam("training_file"))
    Log::Warn << "--print_training_error ignored because --training_file is not"
        << " specified." << endl;

  if (!CLI::HasParam("test_file"))
  {
    if (CLI::HasParam("probabilities_file"))
      Log::Warn << "--probabilities_file ignored because --test_file is not "
          << "specified." << endl;
    if (CLI::HasParam("predictions_file"))
      Log::Warn << "--predictions_file ignored because --test_file is not "
          << "specified." << endl;
  }

  // Load the model or build the tree.
  DecisionTreeModel model;

  if (CLI::HasParam("training_file"))
  {
    arma::mat dataset;
    data::Load(CLI::GetParam<std::string>("training_file"), dataset, true);
    arma::Mat<size_t> labels;
    data::Load(CLI::GetParam<std::string>("labels_file"), labels, true);

    // Calculate number of classes.
    const size_t numClasses = arma::max(arma::max(labels)) + 1;

    // Now build the tree.
    const size_t minLeafSize = (size_t) CLI::GetParam<int>("minimum_leaf_size");

    model.tree = DecisionTree<>(dataset, labels.row(0), numClasses,
        minLeafSize);

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
      Log::Info << double(correct) / double(dataset.n_cols) * 100 << "\% "
          << "correct on training set (" << correct << " / " << dataset.n_cols
          << ")." << endl;
    }
  }
  else
  {
    data::Load(CLI::GetParam<std::string>("input_model_file"), "model", model,
        true);
  }

  // Do we need to get predictions?
  if (CLI::HasParam("test_file"))
  {
    arma::mat testPoints;
    data::Load(CLI::GetParam<std::string>("test_file"), testPoints, true);

    arma::Row<size_t> predictions;
    arma::mat probabilities;

    model.tree.Classify(testPoints, predictions, probabilities);

    // Do we need to calculate accuracy?
    if (CLI::HasParam("test_labels_file"))
    {
      arma::Mat<size_t> testLabels;
      data::Load(CLI::GetParam<std::string>("test_labels_file"), testLabels,
          true);

      size_t correct = 0;
      for (size_t i = 0; i < testPoints.n_cols; ++i)
        if (predictions[i] == testLabels[i])
          ++correct;

      // Print number of correct points.
      Log::Info << double(correct) / double(testPoints.n_cols) * 100 << "\% "
          << "correct on test set (" << correct << " / " << testPoints.n_cols
          << ")." << endl;
    }

    // Do we need to save outputs?
    if (CLI::HasParam("predictions_file"))
      data::Save(CLI::GetParam<std::string>("prediction_file"), predictions);
    if (CLI::HasParam("probabilities_file"))
      data::Save(CLI::GetParam<std::string>("probabilities_file"),
          probabilities);
  }

  // Do we need to save the model?
  if (CLI::HasParam("output_model_file"))
    data::Save(CLI::GetParam<std::string>("output_model_file"), "model", model);
}
