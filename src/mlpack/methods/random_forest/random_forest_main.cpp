/**
 * @file random_forest_main.cpp
 * @author Ryan Curtin
 *
 * A program to build and evaluate random forests.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/random_forest/random_forest.hpp>
#include <mlpack/methods/decision_tree/random_dimension_select.hpp>
#include <mlpack/core/util/mlpack_main.hpp>

using namespace mlpack;
using namespace mlpack::tree;
using namespace std;

PROGRAM_INFO("Random forests",
    "This program is an implementation of the standard random forest "
    "classification algorithm by Leo Breiman.  A random forest can be "
    "trained and saved for later use, or a random forest may be loaded "
    "and predictions or class probabilities for points may be generated."
    "\n\n"
    "This documentation will be rewritten once #880 is merged.");

PARAM_MATRIX_IN("training", "Training dataset.", "t");
PARAM_UROW_IN("labels", "Labels for training dataset.", "l");
PARAM_MATRIX_IN("test", "Test dataset to produce predictions for.", "T");
PARAM_UROW_IN("test_labels", "Test dataset labels, if accuracy calculation is "
    "desired.", "L");

PARAM_FLAG("print_training_accuracy", "If set, then the accuracy of the model "
    "on the training set will be predicted (verbose must also be specified).",
    "a");

PARAM_INT_IN("num_trees", "Number of trees in the random forest.", "N", 10);
PARAM_INT_IN("minimum_leaf_size", "Minimum number of points in each leaf "
    "node.", "n", 20);

PARAM_MATRIX_OUT("probabilities", "Predicted class probabilities for each "
    "point in the test set.", "P");
PARAM_UROW_OUT("predictions", "Predicted classes for each point in the test "
    "set.", "p");

/**
 * This is the class that we will serialize.  It is a pretty simple wrapper
 * around DecisionTree<>.  In order to support categoricals, it will need to
 * also hold and serialize a DatasetInfo.
 */
class RandomForestModel
{
 public:
  // The tree itself, left public for direct access by this program.
  RandomForest<> rf;

  // Create the model.
  RandomForestModel() { /* Nothing to do. */ }

  // Serialize the model.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(rf, "random_forest");
  }
};

PARAM_MODEL_IN(RandomForestModel, "input_model", "Pre-trained random forest to "
    "use for classification.", "m");
PARAM_MODEL_OUT(RandomForestModel, "output_model", "Model to save trained "
    "random forest to.", "M");

void mlpackMain()
{
  // Check for incompatible input parameters.
  if (CLI::HasParam("training") && CLI::HasParam("input_model"))
  {
    Log::Fatal << "Cannot specify both --training_file and --input_model_file!"
        << endl;
  }

  if (CLI::HasParam("print_training_accuracy") && !CLI::HasParam("training"))
  {
    Log::Warn << "--print_training_accuracy ignored because no model is being "
        << "trained." << endl;
  }

  if (!CLI::HasParam("training") && !CLI::HasParam("input_model"))
  {
    Log::Fatal << "Either --training_file or --input_model_file must be "
        << "specified!" << endl;
  }

  if (CLI::HasParam("test") &&
      !CLI::HasParam("probabilities") &&
      !CLI::HasParam("predictions") &&
      !CLI::HasParam("test_labels"))
  {
    Log::Warn << "Neither --probabilities_file nor --predictions_file are "
        << "specified; no test output will be saved!" << endl;
  }

  if (!CLI::HasParam("test") && CLI::HasParam("test_labels"))
  {
    Log::Warn << "--test_labels_file ignored because --test_file not given."
        << endl;
  }

  if (!CLI::HasParam("test") &&
      !CLI::HasParam("output_model") &&
      !CLI::HasParam("print_training_accuracy"))
  {
    Log::Warn << "Neither --test_file nor --output_model_file is specified, "
        << "and --print_training_accuracy is also not given.  The trained "
        << "model will not be used or saved." << endl;
  }

  if (CLI::HasParam("training") && !CLI::HasParam("labels"))
  {
    Log::Fatal << "If --training_file is specified, then --labels_file must be "
        << "specified!" << endl;
  }

  if (CLI::HasParam("training") &&
      CLI::HasParam("num_trees") &&
      CLI::GetParam<int>("num_trees") <= 0)
  {
    Log::Fatal << "Invalid number of trees (" << CLI::GetParam<int>("num_trees")
        << "); must be greater than 0!" << endl;
  }

  if (CLI::HasParam("predictions") && !CLI::HasParam("test"))
  {
    Log::Warn << "--predictions_file ignored because --test_file not specified."
        << endl;
  }

  if (CLI::HasParam("probabilities") && !CLI::HasParam("test"))
  {
    Log::Warn << "--probabilities_file ignored because --test_file not "
        << "specified." << endl;
  }

  if (CLI::HasParam("minimum_leaf_size") &&
      CLI::GetParam<int>("minimum_leaf_size") <= 0)
  {
    Log::Fatal << "Invalid minimum leaf size ("
        << CLI::GetParam<int>("minimum_leaf_size")
        << "); must be greater than 0!" << endl;
  }

  RandomForestModel rfModel;
  if (CLI::HasParam("training"))
  {
    // Train the model on the given input data.
    arma::mat data = std::move(CLI::GetParam<arma::mat>("training"));
    arma::Row<size_t> labels =
        std::move(CLI::GetParam<arma::Row<size_t>>("labels"));
    const size_t numTrees = (size_t) CLI::GetParam<int>("num_trees");
    const size_t minimumLeafSize =
        (size_t) CLI::GetParam<int>("minimum_leaf_size");

    Log::Info << "Training random forest with " << numTrees << " trees..."
        << endl;

    const size_t numClasses = arma::max(labels) + 1;

    // Train the model.
    rfModel.rf.Train(data, labels, numClasses, numTrees, minimumLeafSize);

    // Did we want training accuracy?
    if (CLI::HasParam("print_training_accuracy"))
    {
      arma::Row<size_t> predictions;
      rfModel.rf.Classify(data, predictions);

      const size_t correct = arma::accu(predictions == labels);

      Log::Info << correct << " of " << labels.n_elem << " correct on training"
          << " set (" << (double(correct) / double(labels.n_elem) * 100) << ")."
          << endl;
    }
  }
  else
  {
    // Then we must be loading a model.
    rfModel = std::move(CLI::GetParam<RandomForestModel>("input_model"));
  }

  if (CLI::HasParam("test"))
  {
    arma::mat testData = std::move(CLI::GetParam<arma::mat>("test"));

    // Get predictions and probabilities.
    arma::Row<size_t> predictions;
    arma::mat probabilities;
    rfModel.rf.Classify(testData, predictions, probabilities);

    // Did we want to calculate test accuracy?
    if (CLI::HasParam("test_labels"))
    {
      arma::Row<size_t> testLabels =
          std::move(CLI::GetParam<arma::Row<size_t>>("test_labels"));

      const size_t correct = arma::accu(predictions == testLabels);

      Log::Info << correct << " of " << testLabels.n_elem << " correct on test"
          << " set (" << (double(correct) / double(testLabels.n_elem) * 100)
          << ")." << endl;
    }

    // Should we save the outputs?
    if (CLI::HasParam("probabilities"))
      CLI::GetParam<arma::mat>("probabilities") = std::move(probabilities);
    if (CLI::HasParam("predictions"))
      CLI::GetParam<arma::Row<size_t>>("predictions") = std::move(predictions);
  }

  // Did the user want to save the output model?
  if (CLI::HasParam("output_model"))
    CLI::GetParam<RandomForestModel>("output_model") = std::move(rfModel);
}
