/**
 * @file hoeffding_tree_main.cpp
 * @author Ryan Curtin
 *
 * A command-line executable that can build a streaming decision tree.
 *
 * This file is part of mlpack 2.0.0.
 *
 * mlpack is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * mlpack is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * mlpack.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/hoeffding_trees/hoeffding_tree.hpp>
#include <mlpack/methods/hoeffding_trees/binary_numeric_split.hpp>
#include <mlpack/methods/hoeffding_trees/information_gain.hpp>
#include <queue>

using namespace std;
using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::data;

PROGRAM_INFO("Hoeffding trees",
    "This program implements Hoeffding trees, a form of streaming decision tree"
    " suited best for large (or streaming) datasets.  This program supports "
    "both categorical and numeric data stored in the ARFF format.  Given an "
    "input dataset, this program is able to train the tree with numerous "
    "training options, and save the model to a file.  The program is also able "
    "to use a trained model or a model from file in order to predict classes "
    "for a given test set."
    "\n\n"
    "The training file and associated labels are specified with the "
    "--training_file and --labels_file options, respectively.  The training "
    "file must be in ARFF format.  The training may be performed in batch mode "
    "(like a typical decision tree algorithm) by specifying the --batch_mode "
    "option, but this may not be the best option for large datasets."
    "\n\n"
    "When a model is trained, it may be saved to a file with the "
    "--output_model_file (-M) option.  A model may be loaded from file for "
    "further training or testing with the --input_model_file (-m) option."
    "\n\n"
    "A test file may be specified with the --test_file (-T) option, and if "
    "performance numbers are desired for that test set, labels may be specified"
    " with the --test_labels_file (-L) option.  Predictions for each test point"
    " will be stored in the file specified by --predictions_file (-p) and "
    "probabilities for each predictions will be stored in the file specified by"
    " the --probabilities_file (-P) option.");

PARAM_STRING("training_file", "Training dataset file.", "t", "");
PARAM_STRING("labels_file", "Labels for training dataset.", "l", "");

PARAM_DOUBLE("confidence", "Confidence before splitting (between 0 and 1).",
    "c", 0.95);
PARAM_INT("max_samples", "Maximum number of samples before splitting.", "n",
    5000);
PARAM_INT("min_samples", "Minimum number of samples before splitting.", "I",
    100);

PARAM_STRING("input_model_file", "File to load trained tree from.", "m", "");
PARAM_STRING("output_model_file", "File to save trained tree to.", "M", "");

PARAM_STRING("test_file", "File of testing data.", "T", "");
PARAM_STRING("test_labels_file", "Labels of test data.", "L", "");
PARAM_STRING("predictions_file", "File to output label predictions for test "
    "data into.", "p", "");
PARAM_STRING("probabilities_file", "In addition to predicting labels, provide "
    "prediction probabilities in this file.", "P", "");

PARAM_STRING("numeric_split_strategy", "The splitting strategy to use for "
    "numeric features: 'domingos' or 'binary'.", "N", "binary");
PARAM_FLAG("batch_mode", "If true, samples will be considered in batch instead "
    "of as a stream.  This generally results in better trees but at the cost of"
    " memory usage and runtime.", "b");
PARAM_FLAG("info_gain", "If set, information gain is used instead of Gini "
    "impurity for calculating Hoeffding bounds.", "i");
PARAM_INT("passes", "Number of passes to take over the dataset.", "s", 1);

PARAM_INT("bins", "If the 'domingos' split strategy is used, this specifies "
    "the number of bins for each numeric split.", "B", 10);
PARAM_INT("observations_before_binning", "If the 'domingos' split strategy is "
    "used, this specifies the number of samples observed before binning is "
    "performed.", "o", 100);

// Helper function for once we have chosen a tree type.
template<typename TreeType>
void PerformActions(const typename TreeType::NumericSplit& numericSplit =
    typename TreeType::NumericSplit(0));

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

  if (trainingFile.empty() && CLI::HasParam("batch_mode"))
    Log::Warn << "--batch_mode (-b) ignored; no training set provided." << endl;

  if (CLI::HasParam("passes") && CLI::HasParam("batch_mode"))
    Log::Warn << "--batch_mode (-b) ignored because --passes was specified."
        << endl;

  if (CLI::HasParam("info_gain"))
  {
    if (numericSplitStrategy == "domingos")
    {
      const size_t bins = (size_t) CLI::GetParam<int>("bins");
      const size_t observationsBeforeBinning = (size_t)
          CLI::GetParam<int>("observations_before_binning");
      HoeffdingDoubleNumericSplit<InformationGain> ns(0, bins,
          observationsBeforeBinning);
      PerformActions<HoeffdingTree<InformationGain, HoeffdingDoubleNumericSplit,
          HoeffdingCategoricalSplit>>(ns);
    }
    else if (numericSplitStrategy == "binary")
    {
      PerformActions<HoeffdingTree<InformationGain, BinaryDoubleNumericSplit,
          HoeffdingCategoricalSplit>>();
    }
    else
    {
      Log::Fatal << "Unrecognized numeric split strategy ("
          << numericSplitStrategy << ")!  Must be 'domingos' or 'binary'."
          << endl;
    }
  }
  else
  {
    if (numericSplitStrategy == "domingos")
    {
      const size_t bins = (size_t) CLI::GetParam<int>("bins");
      const size_t observationsBeforeBinning = (size_t)
          CLI::GetParam<int>("observations_before_binning");
      HoeffdingDoubleNumericSplit<GiniImpurity> ns(0, bins,
          observationsBeforeBinning);
      PerformActions<HoeffdingTree<GiniImpurity, HoeffdingDoubleNumericSplit,
          HoeffdingCategoricalSplit>>(ns);
    }
    else if (numericSplitStrategy == "binary")
    {
      PerformActions<HoeffdingTree<GiniImpurity, BinaryDoubleNumericSplit,
          HoeffdingCategoricalSplit>>();
    }
    else
    {
      Log::Fatal << "Unrecognized numeric split strategy ("
          << numericSplitStrategy << ")!  Must be 'domingos' or 'binary'."
          << endl;
    }
  }
}

template<typename TreeType>
void PerformActions(const typename TreeType::NumericSplit& numericSplit)
{
  // Load necessary parameters.
  const string trainingFile = CLI::GetParam<string>("training_file");
  const string labelsFile = CLI::GetParam<string>("labels_file");
  const double confidence = CLI::GetParam<double>("confidence");
  const size_t maxSamples = (size_t) CLI::GetParam<int>("max_samples");
  const size_t minSamples = (size_t) CLI::GetParam<int>("min_samples");
  const string inputModelFile = CLI::GetParam<string>("input_model_file");
  const string outputModelFile = CLI::GetParam<string>("output_model_file");
  const string testFile = CLI::GetParam<string>("test_file");
  const string predictionsFile = CLI::GetParam<string>("predictions_file");
  const string probabilitiesFile = CLI::GetParam<string>("probabilities_file");
  bool batchTraining = CLI::HasParam("batch_mode");
  const size_t passes = (size_t) CLI::GetParam<int>("passes");
  if (passes > 1)
    batchTraining = false; // We already warned about this earlier.

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
    if (passes > 1)
      Log::Info << "Taking " << passes << " passes over the dataset." << endl;

    tree = new TreeType(trainingSet, datasetInfo, labels, max(labels) + 1,
        batchTraining, confidence, maxSamples, 100, minSamples,
        typename TreeType::CategoricalSplit(0, 0), numericSplit);

    for (size_t i = 1; i < passes; ++i)
      tree->Train(trainingSet, labels, false);
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
      if (passes > 1)
      {
        Log::Info << "Taking " << passes << " passes over the dataset." << endl;
        for (size_t i = 0; i < passes; ++i)
          tree->Train(trainingSet, labels, false);
      }
      else
      {
        tree->Train(trainingSet, labels, batchTraining);
      }
      Timer::Stop("tree_training");
    }
  }

  if (!trainingFile.empty())
  {
    // Get training error.
    arma::mat trainingSet;
    data::Load(trainingFile, trainingSet, datasetInfo, true);
    arma::Row<size_t> predictions;
    tree->Classify(trainingSet, predictions);

    arma::Col<size_t> labelsIn;
    data::Load(labelsFile, labelsIn, true, false);
    arma::Row<size_t> labels = labelsIn.t();

    size_t correct = 0;
    for (size_t i = 0; i < labels.n_elem; ++i)
      if (labels[i] == predictions[i])
        ++correct;

    Log::Info << correct << " out of " << labels.n_elem << " correct "
        << "on training set (" << double(correct) / double(labels.n_elem) *
        100.0 << ")." << endl;
  }

  // Get the number of nods in the tree.
  std::queue<TreeType*> queue;
  queue.push(tree);
  size_t nodes = 0;
  while (!queue.empty())
  {
    TreeType* node = queue.front();
    queue.pop();
    ++nodes;

    for (size_t i = 0; i < node->NumChildren(); ++i)
      queue.push(&node->Child(i));
  }
  Log::Info << nodes << " nodes in the tree." << endl;

  // The tree is trained or loaded.  Now do any testing if we need.
  if (!testFile.empty())
  {
    arma::mat testSet;
    data::Load(testFile, testSet, datasetInfo, true);

    arma::Row<size_t> predictions;
    arma::rowvec probabilities;

    Timer::Start("tree_testing");
    tree->Classify(testSet, predictions, probabilities);
    Timer::Stop("tree_testing");

    if (CLI::HasParam("test_labels_file"))
    {
      string testLabelsFile = CLI::GetParam<string>("test_labels_file");
      arma::Col<size_t> testLabelsIn;
      data::Load(testLabelsFile, testLabelsIn, true, false);
      arma::Row<size_t> testLabels = testLabelsIn.t();

      size_t correct = 0;
      for (size_t i = 0; i < testLabels.n_elem; ++i)
      {
        if (predictions[i] == testLabels[i])
          ++correct;
      }
      Log::Info << correct << " out of " << testLabels.n_elem << " correct "
          << "on test set (" << double(correct) / double(testLabels.n_elem) *
          100.0 << ")." << endl;
    }

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
