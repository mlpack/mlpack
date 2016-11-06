/**
 * @file hoeffding_tree_main.cpp
 * @author Ryan Curtin
 *
 * A command-line executable that can build a streaming decision tree.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
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

PARAM_STRING_IN("training_file", "File containing training dataset.", "t", "");
PARAM_UMATRIX_IN("labels", "Labels for training dataset.", "l");

PARAM_DOUBLE_IN("confidence", "Confidence before splitting (between 0 and 1).",
    "c", 0.95);
PARAM_INT_IN("max_samples", "Maximum number of samples before splitting.", "n",
    5000);
PARAM_INT_IN("min_samples", "Minimum number of samples before splitting.", "I",
    100);

PARAM_STRING_IN("input_model_file", "File to load trained tree from.", "m", "");
PARAM_STRING_OUT("output_model_file", "File to save trained tree to.", "M");

PARAM_STRING_IN("test_file", "File containing testing dataset.", "T", "");
PARAM_UMATRIX_IN("test_labels", "Labels of test data.", "L");
PARAM_UMATRIX_OUT("predictions", "Matrix to output label predictions for test "
    "data into.", "p");
PARAM_MATRIX_OUT("probabilities", "In addition to predicting labels, provide "
    "rediction probabilities in this matrix.", "P");

PARAM_STRING_IN("numeric_split_strategy", "The splitting strategy to use for "
    "numeric features: 'domingos' or 'binary'.", "N", "binary");
PARAM_FLAG("batch_mode", "If true, samples will be considered in batch instead "
    "of as a stream.  This generally results in better trees but at the cost of"
    " memory usage and runtime.", "b");
PARAM_FLAG("info_gain", "If set, information gain is used instead of Gini "
    "impurity for calculating Hoeffding bounds.", "i");
PARAM_INT_IN("passes", "Number of passes to take over the dataset.", "s", 1);

PARAM_INT_IN("bins", "If the 'domingos' split strategy is used, this specifies "
    "the number of bins for each numeric split.", "B", 10);
PARAM_INT_IN("observations_before_binning", "If the 'domingos' split strategy "
    "is used, this specifies the number of samples observed before binning is "
    "performed.", "o", 100);

// Helper function for once we have chosen a tree type.
template<typename TreeType>
void PerformActions(const typename TreeType::NumericSplit& numericSplit =
    typename TreeType::NumericSplit(0));

int main(int argc, char** argv)
{
  CLI::ParseCommandLine(argc, argv);

  // Check input parameters for validity.
  const string inputModelFile = CLI::GetParam<string>("input_model_file");
  const string numericSplitStrategy =
      CLI::GetParam<string>("numeric_split_strategy");

  if ((CLI::HasParam("predictions") ||
       CLI::HasParam("probabilities")) &&
       !CLI::HasParam("test_file"))
    Log::Fatal << "--test_file must be specified if --predictions_file or "
        << "--probabilities_file is specified." << endl;

  if (!CLI::HasParam("training_file") && !CLI::HasParam("input_model_file"))
    Log::Fatal << "One of --training_file or --input_model_file must be "
        << "specified!" << endl;

  if (CLI::HasParam("training_file") && !CLI::HasParam("labels"))
    Log::Fatal << "If --training_file is specified, --labels_file must be "
        << "specified too!" << endl;

  if (!CLI::HasParam("training_file") && CLI::HasParam("batch_mode"))
    Log::Warn << "--batch_mode (-b) ignored; no training set provided." << endl;

  if (CLI::HasParam("passes") && CLI::HasParam("batch_mode"))
    Log::Warn << "--batch_mode (-b) ignored because --passes was specified."
        << endl;

  if (CLI::HasParam("test_file") &&
      !CLI::HasParam("predictions") &&
      !CLI::HasParam("probabilities") &&
      !CLI::HasParam("test_labels"))
    Log::Warn << "--test_file (-T) is specified, but none of "
        << "--predictions_file (-p), --probabilities_file (-P), or "
        << "--test_labels_file (-L) are specified, so no output will be given!"
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
  const double confidence = CLI::GetParam<double>("confidence");
  const size_t maxSamples = (size_t) CLI::GetParam<int>("max_samples");
  const size_t minSamples = (size_t) CLI::GetParam<int>("min_samples");
  const string inputModelFile = CLI::GetParam<string>("input_model_file");
  const string outputModelFile = CLI::GetParam<string>("output_model_file");
  bool batchTraining = CLI::HasParam("batch_mode");
  const size_t passes = (size_t) CLI::GetParam<int>("passes");
  if (passes > 1)
    batchTraining = false; // We already warned about this earlier.

  TreeType* tree = NULL;
  DatasetInfo datasetInfo;
  if (!CLI::HasParam("input_model_file"))
  {
    arma::mat trainingSet;
    data::Load(CLI::GetParam<string>("training_file"), trainingSet, datasetInfo,
        true);
    for (size_t i = 0; i < trainingSet.n_rows; ++i)
      Log::Info << datasetInfo.NumMappings(i) << " mappings in dimension "
          << i << "." << endl;

    arma::Mat<size_t> labels = CLI::GetParam<arma::Mat<size_t>>("labels");

    if (labels.n_rows > 1)
      labels = labels.t();
    if (labels.n_rows > 1)
      Log::Fatal << "Labels must be one-dimensional!" << endl;

    // Now create the decision tree.
    Timer::Start("tree_training");
    if (passes > 1)
      Log::Info << "Taking " << passes << " passes over the dataset." << endl;

    tree = new TreeType(trainingSet, datasetInfo, labels.row(0),
        max(labels.row(0)) + 1, batchTraining, confidence, maxSamples, 100,
        minSamples, typename TreeType::CategoricalSplit(0, 0), numericSplit);

    for (size_t i = 1; i < passes; ++i)
      tree->Train(trainingSet, labels, false);
    Timer::Stop("tree_training");
  }
  else
  {
    tree = new TreeType(datasetInfo, 1, 1);
    data::Load(inputModelFile, "streamingDecisionTree", *tree, true);

    if (CLI::HasParam("training_file"))
    {
      arma::mat trainingSet;
      data::Load(CLI::GetParam<string>("training_file"), trainingSet,
          datasetInfo, true);
      for (size_t i = 0; i < trainingSet.n_rows; ++i)
        Log::Info << datasetInfo.NumMappings(i) << " mappings in dimension "
            << i << "." << endl;

      arma::Mat<size_t> labels = CLI::GetParam<arma::Mat<size_t>>("labels");

      if (labels.n_rows > 1)
        labels = labels.t();
      if (labels.n_rows > 1)
        Log::Fatal << "Labels must be one-dimensional!" << endl;

      // Now create the decision tree.
      Timer::Start("tree_training");
      if (passes > 1)
      {
        Log::Info << "Taking " << passes << " passes over the dataset." << endl;
        for (size_t i = 0; i < passes; ++i)
          tree->Train(trainingSet, labels.row(0), false);
      }
      else
      {
        tree->Train(trainingSet, labels.row(0), batchTraining);
      }
      Timer::Stop("tree_training");
    }
  }

  if (CLI::HasParam("training_file"))
  {
    // Get training error.
    arma::mat trainingSet;
    data::Load(CLI::GetParam<string>("training_file"), trainingSet, datasetInfo,
        true);
    arma::Row<size_t> predictions;
    tree->Classify(trainingSet, predictions);

    arma::Mat<size_t> labels = CLI::GetParam<arma::Mat<size_t>>("labels");

    if (labels.n_rows > 1)
      labels = labels.t();
    if (labels.n_rows > 1)
      Log::Fatal << "Labels must be one-dimensional!" << endl;

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
  if (CLI::HasParam("test_file"))
  {
    arma::mat testSet;
    data::Load(CLI::GetParam<string>("test_file"), testSet, datasetInfo, true);

    arma::Row<size_t> predictions;
    arma::rowvec probabilities;

    Timer::Start("tree_testing");
    tree->Classify(testSet, predictions, probabilities);
    Timer::Stop("tree_testing");

    if (CLI::HasParam("test_labels"))
    {
      arma::Mat<size_t> testLabels =
          std::move(CLI::GetParam<arma::Mat<size_t>>("test_labels"));
      if (testLabels.n_rows > 1)
        testLabels = testLabels.t();
      if (testLabels.n_rows > 1)
        Log::Fatal << "Test labels must be one-dimensional!" << endl;

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

    if (CLI::HasParam("predictions"))
      CLI::GetParam<arma::Mat<size_t>>("predictions") = std::move(predictions);

    if (CLI::HasParam("probabilities"))
      CLI::GetParam<arma::mat>("probabilities") = std::move(probabilities);
  }

  // Check the accuracy on the training set.
  if (CLI::HasParam("output_model_file"))
    data::Save(outputModelFile, "streamingDecisionTree", *tree, true);

  // Clean up memory.
  delete tree;
}
