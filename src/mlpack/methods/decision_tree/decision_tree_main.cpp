/**
 * @file decision_tree_main.cpp
 * @author Ryan Curtin
 *
 * A command-line program to build a decision tree.
 */
#include <mlpack/core.hpp>
#include "decision_tree.hpp"

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
    "option is specified."
    "\n\n"
    "A file containing test data may be specified with the --test_file (-T) "
    "option, and if performance numbers are desired for that test set, labels "
    "may be specified with the --test_labels_file (-L) option.  Predictions f"
    "for each test point may be stored into the file specified by the "
    "--predictions_file (-p) option.  Class probabilities for each prediction "
    "will be stored in the file specified by the --probabilities_file (-P) "
    "option.");

// Datasets.
PARAM_MATRIX_IN("training", "Matrix of training points.", "t");
PARAM_UMATRIX_IN("labels", "Training labels.", "l");
PARAM_MATRIX_IN("test", "Matrix of test points.", "T");
PARAM_UMATRIX_IN("test_labels", "Test point labels, if accuracy calculation "
    "is desired.", "L");

// Models.
PARAM_MODEL_IN(DecisionTree<>, "input_model", "Pre-trained decision tree, to be"
    " used with test points.", "m");
PARAM_MODEL_OUT(DecisionTree<>, "output_model", "Output for trained decision "
    "tree.", "M");

// Training parameters.
PARAM_INT_IN("minimum_leaf_size", "Minimum number of points in a leaf.", "l",
    20);

// Output parameters.
PARAM_MATRIX_OUT("probabilities", "Class probabilities for each test point.",
    "P");
PARAM_UMATRIX_OUT("predictions", "Class predictions for each test point.", "p");

int main(int argc, char** argv)
{
  CLI::ParseCommandLine(argc, argv);

  // Check parameters.
  if (CLI::HasParam("training") && CLI::HasParam("input_model"))
    Log::Fatal << "Cannot specify both --training_file and --input_model_file!"
        << endl;

  if (CLI::HasParam("training") && !CLI::HasParam("labels"))
    Log::Fatal << "Must specify --labels_file when --training_file is "
        << "specified!" << endl;

  if (CLI::HasParam("test_labels") && !CLI::HasParam("test"))
    Log::Warn << "--test_labels_file ignored because --test_file is not passed."
        << endl;

  if (!CLI::HasParam("output_model") && !CLI::HasParam("probabilities") &&
      !CLI::HasParam("predictions") && !CLI::HasParam("test_labels"))
    Log::Warn << "None of --output_model_file, --probabilities_file, or "
        << "--predictions_file are given, and accuracy is not being calculated;"
        << " no output will be saved!" << endl;

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
  DecisionTree tree;

  if (CLI::HasParam("training"))
  {
    arma::mat dataset = std::move(CLI::GetParam<arma::mat>("training"));
    arma::Mat<size_t> labels =
        std::move(CLI::GetParam<arma::Mat<size_t>>("labels"));

    // Calculate number of classes.
    const size_t numClasses = arma::max(labels) + 1;

    // Now build the tree.
    const size_t minLeafSize = (size_t) CLI::GetParam<int>("minimum_leaf_size");

    tree = DecisionTree(dataset, labels.row(0), numClasses, minLeafSize);
  }
  else
  {
    tree = std::move(CLI::GetParam<DecisionTree<>>("input_model"));
  }

  // Do we need to get predictions?
  if (CLI::HasParam("test"))
  {
    arma::mat testPoints = std::move(CLI::GetParam<arma::mat>("test"));

    arma::Row<size_t> predictions;
    arma::mat probabilities;

    tree.Classify(testPoints, predictions, probabilities);

    // Do we need to calculate accuracy?
    if (CLI::HasParam("test_labels"))
    {
      arma::Mat<size_t> testLabels =
          std::move(CLI::GetParam<arma::Mat<size_t>>("test_labels"));

      size_t correct = 0;
      for (size_t i = 0; i < testPoints.n_cols; ++i)
        if (predictions[i] == testLabels[i])
          ++correct;

      // Print number of correct points.
      Log::Info << double(correct) / double(testPoints.n_cols) * 100 << "\% "
          << "correct (" << correct << " / " << testPoints.n_cols << ")."
          << endl;
    }

    // Do we need to save outputs?
    if (CLI::HasParam("predictions"))
      CLI::GetParam<arma::Mat<size_t>>("predictions") = std::move(predictions);
    if (CLI::HasParam("probabilities"))
      CLI::GetParam<arma::mat>("probabilities") = std::move(probabilities);
  }

  // Do we need to save the model?
  if (CLI::HasParam("output_model"))
    CLI::GetParam<DecisionTree<>>("output_model") = std::move(tree);
}
