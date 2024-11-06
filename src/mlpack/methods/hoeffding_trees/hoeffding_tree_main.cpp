/**
 * @file methods/hoeffding_trees/hoeffding_tree_main.cpp
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

#undef BINDING_NAME
#define BINDING_NAME hoeffding_tree

#include <mlpack/core/util/mlpack_main.hpp>

#include <mlpack/methods/hoeffding_trees/hoeffding_tree.hpp>
#include <mlpack/methods/hoeffding_trees/binary_numeric_split.hpp>
#include <mlpack/methods/hoeffding_trees/information_gain.hpp>
#include <mlpack/methods/hoeffding_trees/hoeffding_tree_model.hpp>

using namespace std;
using namespace mlpack;
using namespace mlpack::data;
using namespace mlpack::util;

// Program Name.
BINDING_USER_NAME("Hoeffding trees");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of Hoeffding trees, a form of streaming decision tree "
    "for classification.  Given labeled data, a Hoeffding tree can be trained "
    "and saved for later use, or a pre-trained Hoeffding tree can be used for "
    "predicting the classifications of new points.");

// Long description.
BINDING_LONG_DESC(
    "This program implements Hoeffding trees, a form of streaming decision tree"
    " suited best for large (or streaming) datasets.  This program supports "
    "both categorical and numeric data.  Given an input dataset, this program "
    "is able to train the tree with numerous training options, and save the "
    "model to a file.  The program is also able to use a trained model or a "
    "model from file in order to predict classes for a given test set."
    "\n\n"
    "The training file and associated labels are specified with the " +
    PRINT_PARAM_STRING("training") + " and " + PRINT_PARAM_STRING("labels") +
    " parameters, respectively. Optionally, if " +
    PRINT_PARAM_STRING("labels") + " is not specified, the labels are assumed "
    "to be the last dimension of the training dataset."
    "\n\n"
    "The training may be performed in batch mode "
    "(like a typical decision tree algorithm) by specifying the " +
    PRINT_PARAM_STRING("batch_mode") + " option, but this may not be the best "
    "option for large datasets."
    "\n\n"
    "When a model is trained, it may be saved via the " +
    PRINT_PARAM_STRING("output_model") + " output parameter.  A model may be "
    "loaded from file for further training or testing with the " +
    PRINT_PARAM_STRING("input_model") + " parameter."
    "\n\n"
    "Test data may be specified with the " + PRINT_PARAM_STRING("test") + " "
    "parameter, and if performance statistics are desired for that test set, "
    "labels may be specified with the " + PRINT_PARAM_STRING("test_labels") +
    " parameter.  Predictions for each test point may be saved with the " +
    PRINT_PARAM_STRING("predictions") + " output parameter, and class "
    "probabilities for each prediction may be saved with the " +
    PRINT_PARAM_STRING("probabilities") + " output parameter.");

// Example.
BINDING_EXAMPLE(
    "For example, to train a Hoeffding tree with confidence 0.99 with data " +
    PRINT_DATASET("dataset") + ", saving the trained tree to " +
    PRINT_MODEL("tree") + ", the following command may be used:"
    "\n\n" +
    PRINT_CALL("hoeffding_tree", "training", "dataset", "confidence", 0.99,
        "output_model", "tree") +
    "\n\n"
    "Then, this tree may be used to make predictions on the test set " +
    PRINT_DATASET("test_set") + ", saving the predictions into " +
    PRINT_DATASET("predictions") + " and the class probabilities into " +
    PRINT_DATASET("class_probs") + " with the following command: "
    "\n\n" +
    PRINT_CALL("hoeffding_tree", "input_model", "tree", "test", "test_set",
        "predictions", "predictions", "probabilities", "class_probs"));

// See also...
BINDING_SEE_ALSO("@decision_tree", "#decision_tree");
BINDING_SEE_ALSO("@random_forest", "#random_forest");
BINDING_SEE_ALSO("Mining High-Speed Data Streams (pdf)",
    "http://dm.cs.washington.edu/papers/vfdt-kdd00.pdf");
BINDING_SEE_ALSO("HoeffdingTree class documentation",
    "@doc/user/methods/hoeffding_tree.md");

PARAM_MATRIX_AND_INFO_IN("training", "Training dataset (may be categorical).",
    "t");
PARAM_UROW_IN("labels", "Labels for training dataset.", "l");

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

PARAM_MATRIX_AND_INFO_IN("test", "Testing dataset (may be categorical).", "T");
PARAM_UROW_IN("test_labels", "Labels of test data.", "L");
PARAM_UROW_OUT("predictions", "Matrix to output label predictions for test "
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
using TupleType = tuple<DatasetInfo, arma::mat>;

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  // Check input parameters for validity.
  const string numericSplitStrategy =
      params.Get<string>("numeric_split_strategy");

  RequireAtLeastOnePassed(params, { "training", "input_model" }, true);

  RequireAtLeastOnePassed(params, { "output_model", "predictions",
      "probabilities", "test_labels" }, false, "no output will be given");

  ReportIgnoredParam(params, {{ "test", false }}, "probabilities");
  ReportIgnoredParam(params, {{ "test", false }}, "predictions");

  ReportIgnoredParam(params, {{ "training", false }}, "batch_mode");
  ReportIgnoredParam(params, {{ "training", false }}, "passes");

  if (params.Has("test"))
  {
    RequireAtLeastOnePassed(params, { "predictions", "probabilities",
        "test_labels" }, false, "no output will be given");
  }

  RequireParamInSet<string>(params, "numeric_split_strategy", { "domingos",
      "binary" }, true, "unrecognized numeric split strategy");

  // Do we need to load a model or do we already have one?
  HoeffdingTreeModel* model;
  DatasetInfo datasetInfo;
  arma::mat trainingSet;
  arma::Row<size_t> labels;
  if (params.Has("input_model"))
  {
    model = params.Get<HoeffdingTreeModel*>("input_model");
  }
  else
  {
    // Initialize a model.
    if (!params.Has("info_gain") && (numericSplitStrategy == "domingos"))
      model = new HoeffdingTreeModel(HoeffdingTreeModel::GINI_HOEFFDING);
    else if (!params.Has("info_gain") && (numericSplitStrategy == "binary"))
      model = new HoeffdingTreeModel(HoeffdingTreeModel::GINI_BINARY);
    else if (params.Has("info_gain") && (numericSplitStrategy == "domingos"))
      model = new HoeffdingTreeModel(HoeffdingTreeModel::INFO_HOEFFDING);
    else
      model = new HoeffdingTreeModel(HoeffdingTreeModel::INFO_BINARY);
  }

  // Now, do we need to train?
  if (params.Has("training"))
  {
    // Load necessary parameters for training.
    const double confidence = params.Get<double>("confidence");
    const size_t maxSamples = (size_t) params.Get<int>("max_samples");
    const size_t minSamples = (size_t) params.Get<int>("min_samples");
    bool batchTraining = params.Has("batch_mode");
    const size_t bins = (size_t) params.Get<int>("bins");
    const size_t observationsBeforeBinning = (size_t)
        params.Get<int>("observations_before_binning");
    size_t passes = (size_t) params.Get<int>("passes");
    if (passes > 1)
      batchTraining = false; // We already warned about this earlier.

    // We need to train the model.  First, load the data.
    datasetInfo = std::move(std::get<0>(params.Get<TupleType>("training")));
    trainingSet = std::move(std::get<1>(params.Get<TupleType>("training")));
    for (size_t i = 0; i < trainingSet.n_rows; ++i)
      Log::Info << datasetInfo.NumMappings(i) << " mappings in dimension "
          << i << "." << endl;

    if (params.Has("labels"))
    {
      labels = std::move(params.Get<arma::Row<size_t>>("labels"));
    }
    else
    {
      // Extract the labels from the last dimension of training set.
      Log::Info << "Using the last dimension of training set as labels."
          << endl;
      labels = ConvTo<arma::Row<size_t>>::From(
          trainingSet.row(trainingSet.n_rows - 1));
      trainingSet.shed_row(trainingSet.n_rows - 1);
    }

    // Next, create the model with the right type.  Then build the tree with the
    // appropriate type of instantiated numeric split type.  This is a little
    // bit ugly.  Maybe there is a nicer way to get this numeric split
    // information to the trees, but this is ok for now.
    timers.Start("tree_training");

    // Do we need to initialize a model?
    if (!params.Has("input_model"))
    {
      // Build the model.
      model->BuildModel(trainingSet, datasetInfo, labels,
          max(labels) + 1, batchTraining, confidence, maxSamples,
          100, minSamples, bins, observationsBeforeBinning);
      --passes; // This model-building takes one pass.
    }

    // Now pass over the trees as many times as we need to.
    if (batchTraining)
    {
      // We only need to do batch training if we've not already called
      // BuildModel.
      if (params.Has("input_model"))
        model->Train(trainingSet, labels, true);
    }
    else
    {
      for (size_t p = 0; p < passes; ++p)
        model->Train(trainingSet, labels, false);
    }

    timers.Stop("tree_training");
  }

  // Do we need to evaluate the training set error?
  if (params.Has("training"))
  {
    // Get training error.
    arma::Row<size_t> predictions;
    model->Classify(trainingSet, predictions);

    size_t correct = 0;
    for (size_t i = 0; i < labels.n_elem; ++i)
      if (labels[i] == predictions[i])
        ++correct;

    Log::Info << correct << " out of " << labels.n_elem << " correct "
        << "on training set (" << double(correct) / double(labels.n_elem) *
        100.0 << ")." << endl;
  }

  // Get the number of nodes in the tree.
  Log::Info << model->NumNodes() << " nodes in the tree." << endl;

  // The tree is trained or loaded.  Now do any testing if we need.
  if (params.Has("test"))
  {
    // Before loading, pre-set the dataset info by getting the raw parameter
    // (that doesn't call data::Load()).
    std::get<0>(params.GetRaw<TupleType>("test")) = datasetInfo;
    arma::mat testSet = std::get<1>(params.Get<TupleType>("test"));

    arma::Row<size_t> predictions;
    arma::rowvec probabilities;

    timers.Start("tree_testing");
    model->Classify(testSet, predictions, probabilities);
    timers.Stop("tree_testing");

    if (params.Has("test_labels"))
    {
      arma::Row<size_t> testLabels =
          std::move(params.Get<arma::Row<size_t>>("test_labels"));

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

    params.Get<arma::Row<size_t>>("predictions") = std::move(predictions);
    params.Get<arma::mat>("probabilities") = std::move(probabilities);
  }

  // Check the accuracy on the training set.
  params.Get<HoeffdingTreeModel*>("output_model") = model;
}
