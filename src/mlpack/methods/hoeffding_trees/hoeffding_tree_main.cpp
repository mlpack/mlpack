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
#include <mlpack/methods/hoeffding_trees/hoeffding_tree_model.hpp>
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

PARAM_MATRIX_AND_INFO_IN("training", "Training dataset (may be categorical).",
    "t");
PARAM_UMATRIX_IN("labels", "Labels for training dataset.", "l");

PARAM_DOUBLE_IN("confidence", "Confidence before splitting (between 0 and 1).",
    "c", 0.95);
PARAM_INT_IN("max_samples", "Maximum number of samples before splitting.", "n",
    5000);
PARAM_INT_IN("min_samples", "Minimum number of samples before splitting.", "I",
    100);

PARAM_MODEL_IN(HoeffdingTreeModel, "input_model", "Input trained Hoeffding tree"
    " model.", "m");
PARAM_MODEL_OUT(HoeffdingTreeModel, "output_model", "Output for trained "
    "Hoeffding tree model.", "M");

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

// Convenience typedef.
typedef tuple<DatasetInfo, arma::mat> TupleType;

int main(int argc, char** argv)
{
  CLI::ParseCommandLine(argc, argv);

  // Check input parameters for validity.
  const string numericSplitStrategy =
      CLI::GetParam<string>("numeric_split_strategy");

  if ((CLI::HasParam("predictions") ||
       CLI::HasParam("probabilities")) &&
       !CLI::HasParam("test_file"))
    Log::Fatal << "--test_file must be specified if --predictions_file or "
        << "--probabilities_file is specified." << endl;

  if (!CLI::HasParam("training") && !CLI::HasParam("input_model"))
    Log::Fatal << "One of --training_file or --input_model_file must be "
        << "specified!" << endl;

  if (CLI::HasParam("training") && !CLI::HasParam("labels"))
    Log::Fatal << "If --training_file is specified, --labels_file must be "
        << "specified too!" << endl;

  if (!CLI::HasParam("training") && CLI::HasParam("batch_mode"))
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

  if ((numericSplitStrategy != "domingos") &&
      (numericSplitStrategy != "binary"))
  {
    Log::Fatal << "Unrecognized numeric split strategy ("
        << numericSplitStrategy << ")!  Must be 'domingos' or 'binary'."
        << endl;
  }

  // Do we need to load a model or do we already have one?
  HoeffdingTreeModel model;
  DatasetInfo datasetInfo;
  arma::mat trainingSet;
  arma::Mat<size_t> labels;
  if (CLI::HasParam("input_model"))
  {
    model = std::move(CLI::GetParam<HoeffdingTreeModel>("input_model"));
  }
  else
  {
    // Initialize a model.
    if (!CLI::HasParam("info_gain") && (numericSplitStrategy == "domingos"))
      model = HoeffdingTreeModel(HoeffdingTreeModel::GINI_HOEFFDING);
    else if (!CLI::HasParam("info_gain") && (numericSplitStrategy == "binary"))
      model = HoeffdingTreeModel(HoeffdingTreeModel::GINI_BINARY);
    else if (CLI::HasParam("info_gain") && (numericSplitStrategy == "domingos"))
      model = HoeffdingTreeModel(HoeffdingTreeModel::INFO_HOEFFDING);
    else if (CLI::HasParam("info_gain") && (numericSplitStrategy == "binary"))
      model = HoeffdingTreeModel(HoeffdingTreeModel::INFO_BINARY);
  }

  // Now, do we need to train?
  if (CLI::HasParam("training"))
  {
    // Load necessary parameters for training.
    const double confidence = CLI::GetParam<double>("confidence");
    const size_t maxSamples = (size_t) CLI::GetParam<int>("max_samples");
    const size_t minSamples = (size_t) CLI::GetParam<int>("min_samples");
    bool batchTraining = CLI::HasParam("batch_mode");
    const size_t bins = (size_t) CLI::GetParam<int>("bins");
    const size_t observationsBeforeBinning = (size_t)
        CLI::GetParam<int>("observations_before_binning");
    size_t passes = (size_t) CLI::GetParam<int>("passes");
    if (passes > 1)
      batchTraining = false; // We already warned about this earlier.

    // We need to train the model.  First, load the data.
    datasetInfo = std::move(std::get<0>(CLI::GetParam<TupleType>("training")));
    trainingSet = std::move(std::get<1>(CLI::GetParam<TupleType>("training")));
    for (size_t i = 0; i < trainingSet.n_rows; ++i)
      Log::Info << datasetInfo.NumMappings(i) << " mappings in dimension "
          << i << "." << endl;

    labels = CLI::GetParam<arma::Mat<size_t>>("labels");

    if (labels.n_rows > 1)
      labels = labels.t();
    if (labels.n_rows > 1)
      Log::Fatal << "Labels must be one-dimensional!" << endl;

    // Next, create the model with the right type.  Then build the tree with the
    // appropriate type of instantiated numeric split type.  This is a little
    // bit ugly.  Maybe there is a nicer way to get this numeric split
    // information to the trees, but this is ok for now.
    Timer::Start("tree_training");

    // Do we need to initialize a model?
    if (!CLI::HasParam("input_model"))
    {
      // Build the model.
      model.BuildModel(trainingSet, datasetInfo, labels.row(0),
          arma::max(labels.row(0)) + 1, batchTraining, confidence, maxSamples,
          100, minSamples, bins, observationsBeforeBinning);
      --passes; // This model-building takes one pass.
    }

    // Now pass over the trees as many times as we need to.
    if (batchTraining)
    {
      // We only need to do batch training if we've not already called
      // BuildModel.
      if (CLI::HasParam("input_model"))
        model.Train(trainingSet, labels.row(0), true);
    }
    else
    {
      for (size_t p = 0; p < passes; ++p)
        model.Train(trainingSet, labels.row(0), false);
    }

    Timer::Stop("tree_training");
  }

  // Do we need to evaluate the training set error?
  if (CLI::HasParam("training"))
  {
    // Get training error.
    arma::Row<size_t> predictions;
    model.Classify(trainingSet, predictions);

    size_t correct = 0;
    for (size_t i = 0; i < labels.n_elem; ++i)
      if (labels[i] == predictions[i])
        ++correct;

    Log::Info << correct << " out of " << labels.n_elem << " correct "
        << "on training set (" << double(correct) / double(labels.n_elem) *
        100.0 << ")." << endl;
  }

  // Get the number of nodes in the tree.
  Log::Info << model.NumNodes() << " nodes in the tree." << endl;

  // The tree is trained or loaded.  Now do any testing if we need.
  if (CLI::HasParam("test_file"))
  {
    arma::mat testSet;
    data::Load(CLI::GetParam<string>("test_file"), testSet, datasetInfo, true);

    arma::Row<size_t> predictions;
    arma::rowvec probabilities;

    Timer::Start("tree_testing");
    model.Classify(testSet, predictions, probabilities);
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
  if (CLI::HasParam("output_model"))
    CLI::GetParam<HoeffdingTreeModel>("output_model") = std::move(model);

  CLI::Destroy();
}
