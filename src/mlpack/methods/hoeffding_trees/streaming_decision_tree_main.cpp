/**
 * @file streaming_decision_tree_main.cpp
 * @author Ryan Curtin
 *
 * A command-line executable that can build a streaming decision tree.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/hoeffding_trees/streaming_decision_tree.hpp>
#include <mlpack/methods/hoeffding_trees/hoeffding_split.hpp>
#include <mlpack/methods/hoeffding_trees/binary_numeric_split.hpp>

using namespace std;
using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::data;

PARAM_STRING("training_file", "Training dataset file.", "t", "");
PARAM_STRING("labels_file", "Labels for training dataset.", "l", "");

PARAM_DOUBLE("confidence", "Confidence before splitting (between 0 and 1).",
    "c", 0.95);
PARAM_INT("max_samples", "Maximum number of samples before splitting.", "n",
    5000);

PARAM_STRING("input_model_file", "File to load trained tree from.", "m", "");
PARAM_STRING("output_model_file", "File to save trained tree to.", "M", "");

PARAM_STRING("test_file", "File of testing data.", "T", "");
PARAM_STRING("predictions_file", "File to output label predictions for test "
    "data into.", "p", "");
PARAM_STRING("probabilities_file", "In addition to predicting labels, provide "
    "prediction probabilities in this file.", "P", "");

PARAM_STRING("numeric_split_strategy", "The splitting strategy to use for "
    "numeric features: 'domingos' or 'binary'.", "N", "binary");

// Helper function for once we have chosen a tree type.
template<typename TreeType>
void PerformActions();

int main(int argc, char** argv)
{
  CLI::ParseCommandLine(argc, argv);

  // Check input parameters for validity.
  const string trainingFile = CLI::GetParam<string>("training_file");
  const string labelsFile = CLI::GetParam<string>("labels_file");
  const string inputModelFile = CLI::GetParam<string>("input_model_file");
  const string testFile = CLI::GetParam<string>("test_file");
  const string predictionsFile = CLI::GetParam<string>("predictions_file");
  const string probabilitiesFile = CLI::GetParam<string>("probabilities_file");
  const string numericSplitStrategy =
      CLI::GetParam<string>("numeric_split_strategy");

  if ((!predictionsFile.empty() || !probabilitiesFile.empty()) &&
      testFile.empty())
    Log::Fatal << "--test_file must be specified if --predictions_file or "
        << "--probabilities_file is specified." << endl;

  if (trainingFile.empty() && inputModelFile.empty())
    Log::Fatal << "One of --training_file or --input_model_file must be "
        << "specified!" << endl;

  if (!trainingFile.empty() && labelsFile.empty())
    Log::Fatal << "If --training_file is specified, --labels_file must be "
        << "specified too!" << endl;

  if (numericSplitStrategy == "domingos")
    PerformActions<StreamingDecisionTree<HoeffdingSplit<GiniImpurity,
        HoeffdingDoubleNumericSplit, HoeffdingCategoricalSplit>>>();
  else if (numericSplitStrategy == "binary")
    PerformActions<StreamingDecisionTree<HoeffdingSplit<GiniImpurity,
        BinaryDoubleNumericSplit, HoeffdingCategoricalSplit>>>();
  else
    Log::Fatal << "Unrecognized numeric split strategy ("
        << numericSplitStrategy << ")!  Must be 'domingos' or 'binary'."
        << endl;
}

template<typename TreeType>
void PerformActions()
{
  // Load necessary parameters.
  const string trainingFile = CLI::GetParam<string>("training_file");
  const string labelsFile = CLI::GetParam<string>("labels_file");
  const double confidence = CLI::GetParam<double>("confidence");
  const size_t maxSamples = (size_t) CLI::GetParam<int>("max_samples");
  const string inputModelFile = CLI::GetParam<string>("input_model_file");
  const string outputModelFile = CLI::GetParam<string>("output_model_file");
  const string testFile = CLI::GetParam<string>("test_file");
  const string predictionsFile = CLI::GetParam<string>("predictions_file");
  const string probabilitiesFile = CLI::GetParam<string>("probabilities_file");

  TreeType* tree = NULL;
  DatasetInfo datasetInfo;
  if (inputModelFile.empty())
  {
    arma::mat trainingSet;
    data::Load(trainingFile, trainingSet, datasetInfo, true);
    for (size_t i = 0; i < trainingSet.n_rows; ++i)
      Log::Info << datasetInfo.NumMappings(i) << " mappings in dimension "
          << i << "." << endl;

    arma::Col<size_t> labelsIn;
    data::Load(labelsFile, labelsIn, true, false);
    arma::Row<size_t> labels = labelsIn.t();

    // Now create the decision tree.
    Timer::Start("tree_training");
    tree = new TreeType(trainingSet, datasetInfo, labels, max(labels) + 1,
        confidence, maxSamples);
    Timer::Stop("tree_training");
  }
  else
  {
    tree = new TreeType(datasetInfo, 1, 1);
    data::Load(inputModelFile, "streamingDecisionTree", *tree, true);

    if (!trainingFile.empty())
    {
      arma::mat trainingSet;
      data::Load(trainingFile, trainingSet, datasetInfo, true);
      for (size_t i = 0; i < trainingSet.n_rows; ++i)
        Log::Info << datasetInfo.NumMappings(i) << " mappings in dimension "
            << i << "." << endl;

      arma::Col<size_t> labelsIn;
      data::Load(labelsFile, labelsIn, true, false);
      arma::Row<size_t> labels = labelsIn.t();

      // Now create the decision tree.
      Timer::Start("tree_training");
      tree->Train(trainingSet, labels);
      Timer::Stop("tree_training");
    }
  }

  // The tree is trained or loaded.  Now do any testing if we need.
  DatasetInfo testInfo;
  if (!testFile.empty())
  {
    arma::mat testSet;
    data::Load(testFile, testSet, testInfo, true);

    arma::Row<size_t> predictions;
    arma::rowvec probabilities;

    Timer::Start("tree_testing");
    tree->Classify(testSet, predictions, probabilities);
    Timer::Stop("tree_testing");

    if (!predictionsFile.empty())
      data::Save(predictionsFile, predictions);

    if (!probabilitiesFile.empty())
      data::Save(probabilitiesFile, probabilities);
  }

  // Check the accuracy on the training set.
  if (!outputModelFile.empty())
    data::Save(outputModelFile, "streamingDecisionTree", tree, true);

  // Clean up memory.
  delete tree;
}
