/**
 * @file methods/hoeffding_trees/hoeffding_tree_train_main.cpp
 * @author Ryan Curtin
 * @author Dirk Eddelbuettel
 *
 * Implementation of training for a streaming decision tree.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME hoeffding_tree_train

#include <mlpack/core/util/mlpack_main.hpp>

#include <mlpack/methods/hoeffding_trees/hoeffding_tree.hpp>
#include <mlpack/methods/hoeffding_trees/binary_numeric_split.hpp>
#include <mlpack/methods/hoeffding_trees/information_gain.hpp>
#include <mlpack/methods/hoeffding_trees/hoeffding_tree_model.hpp>

using namespace std;
using namespace mlpack;
using namespace mlpack::util;

// Program Name.
BINDING_USER_NAME("Hoeffding trees training");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of Hoeffding trees, a form of streaming decision tree "
    "for classification.  Given labeled data a Hoeffding tree can be trained "
    "for later use of predicting the classifications of new points.");

// Long description.
BINDING_LONG_DESC(
    "Implements Hoeffding trees, a form of streaming decision tree"
    " suited best for large (or streaming) datasets, supporting "
    "both categorical and numeric data.  Given an input dataset, it "
    "is able to train the tree with numerous training options, and return "
    "the model."
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
    "option for large datasets.");

// Example.
BINDING_EXAMPLE(
    IMPORT_EXT_LIB() + "\n" +
    IMPORT_SPLIT() + "\n" +
    IMPORT_THIS("hoeffding_trees", "train", "classify", "probabilities") +
    "\n" +
    GET_DATASET("X", "http://datasets.mlpack.org/iris.csv") + "\n" +
    GET_DATASET("y", "http://datasets.mlpack.org/iris_labels.csv") + "\n" +
    SPLIT_TRAIN_TEST("X", "y", "X_train", "y_train", "X_test", "y_test",
    "0.2") + "\n" +
    CREATE_OBJECT("model", "hoeffding_trees") + "\n" +
    CALL_METHOD("model", "train", "training", "X_train", "labels", "y_train"));
    // FIXME: pick example params

// See also...
BINDING_SEE_ALSO("@decision_tree", "#decision_tree");
BINDING_SEE_ALSO("@random_forest", "#random_forest");
BINDING_SEE_ALSO("Mining High-Speed Data Streams (pdf)",
    "https://www.cs.rhodes.edu/~welshc/COMP465_S15/Papers/kdd00.pdf");
BINDING_SEE_ALSO("HoeffdingTree class documentation",
    "@doc/user/methods/hoeffding_tree.md");

PARAM_MATRIX_AND_INFO_IN_REQ("training",
    "Training dataset (may be categorical).", "t");
PARAM_UROW_IN("labels", "Labels for training dataset.", "l");

PARAM_DOUBLE_IN("confidence", "Confidence before splitting (between 0 and 1).",
    "c", 0.95);
PARAM_INT_IN("max_samples", "Maximum number of samples before splitting.", "n",
    5000);
PARAM_INT_IN("min_samples", "Minimum number of samples before splitting.", "I",
    100);

PARAM_MODEL_OUT(HoeffdingTreeModel, "output_model", "Output for trained "
    "Hoeffding tree model.", "M");

PARAM_MATRIX_AND_INFO_IN("test", "Testing dataset (may be categorical).", "T");
PARAM_UROW_IN("test_labels", "Labels of test data.", "L");

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

  RequireParamInSet<string>(params, "numeric_split_strategy", { "domingos",
      "binary" }, true, "unrecognized numeric split strategy");

  // Do we need to load a model or do we already have one?
  HoeffdingTreeModel* model;
  DatasetInfo datasetInfo;
  arma::mat trainingSet;
  arma::Row<size_t> labels;

  // Initialize a model.
  if (!params.Has("info_gain") && (numericSplitStrategy == "domingos"))
    model = new HoeffdingTreeModel(HoeffdingTreeModel::GINI_HOEFFDING);
  else if (!params.Has("info_gain") && (numericSplitStrategy == "binary"))
    model = new HoeffdingTreeModel(HoeffdingTreeModel::GINI_BINARY);
  else if (params.Has("info_gain") && (numericSplitStrategy == "domingos"))
    model = new HoeffdingTreeModel(HoeffdingTreeModel::INFO_HOEFFDING);
  else
    model = new HoeffdingTreeModel(HoeffdingTreeModel::INFO_BINARY);

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

  // Build the model.
  model->BuildModel(trainingSet, datasetInfo, labels,
        max(labels) + 1, batchTraining, confidence, maxSamples,
        100, minSamples, bins, observationsBeforeBinning);
  --passes; // This model-building takes one pass.


  // Now pass over the trees as many times as we need to.
  if (batchTraining)
  {
    // We only need to do batch training if we've not already called
    // BuildModel.
    for (size_t p = 0; p < passes; ++p)
      model->Train(trainingSet, labels, false);
  }

  timers.Stop("tree_training");

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

  // Get the number of nodes in the tree.
  Log::Info << model->NumNodes() << " nodes in the tree." << endl;

  // Check the accuracy on the training set.
  params.Get<HoeffdingTreeModel*>("output_model") = model;
}
