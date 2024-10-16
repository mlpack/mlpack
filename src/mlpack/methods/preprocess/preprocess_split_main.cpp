/**
 * @file methods/preprocess/preprocess_split_main.cpp
 * @author Keon Kim
 *
 * A binding to split a dataset.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME preprocess_split

#include <mlpack/core/util/mlpack_main.hpp>

// Program Name.
BINDING_USER_NAME("Split Data");

// Short description.
BINDING_SHORT_DESC(
    "A utility to split data into a training and testing dataset.  This can "
    "also split labels according to the same split.");

// Long description.
BINDING_LONG_DESC(
    "This utility takes a dataset and optionally labels and splits them into a "
    "training set and a test set. Before the split, the points in the dataset "
    "are randomly reordered. The percentage of the dataset to be used as the "
    "test set can be specified with the " + PRINT_PARAM_STRING("test_ratio") +
    " parameter; the default is 0.2 (20%)."
    "\n\n"
    "The output training and test matrices may be saved with the " +
    PRINT_PARAM_STRING("training") + " and " + PRINT_PARAM_STRING("test") +
    " output parameters."
    "\n\n"
    "Optionally, labels can also be split along with the data by specifying "
    "the " + PRINT_PARAM_STRING("input_labels") + " parameter.  Splitting "
    "labels works the same way as splitting the data. The output training and "
    "test labels may be saved with the " +
    PRINT_PARAM_STRING("training_labels") + " and " +
    PRINT_PARAM_STRING("test_labels") + " output parameters, respectively.");

// Example.
BINDING_EXAMPLE(
    "So, a simple example where we want to split the dataset " +
    PRINT_DATASET("X") + " into " + PRINT_DATASET("X_train") + " and " +
    PRINT_DATASET("X_test") + " with 60% of the data in the training set and "
    "40% of the dataset in the test set, we could run "
    "\n\n" +
    PRINT_CALL("preprocess_split", "input", "X", "training", "X_train", "test",
        "X_test", "test_ratio", 0.4) +
    "\n\n"
    "Also by default the dataset is shuffled and split; you can provide the " +
    PRINT_PARAM_STRING("no_shuffle") + " option to avoid shuffling the "
    "data; an example to avoid shuffling of data is:"
    "\n\n" +
    PRINT_CALL("preprocess_split", "input", "X", "training", "X_train", "test",
        "X_test", "test_ratio", 0.4, "no_shuffle", true) +
    "\n\n"
    "If we had a dataset " + PRINT_DATASET("X") + " and associated labels " +
    PRINT_DATASET("y") + ", and we wanted to split these into " +
    PRINT_DATASET("X_train") + ", " + PRINT_DATASET("y_train") + ", " +
    PRINT_DATASET("X_test") + ", and " + PRINT_DATASET("y_test") + ", with 30% "
    "of the data in the test set, we could run"
    "\n\n" +
    PRINT_CALL("preprocess_split", "input", "X", "input_labels", "y",
        "test_ratio", 0.3, "training", "X_train", "training_labels", "y_train",
        "test", "X_test", "test_labels", "y_test"));

BINDING_EXAMPLE(
    "To maintain the ratio of each class in the train and test sets, the" +
    PRINT_PARAM_STRING("stratify_data") + " option can be used."
    "\n\n" +
    PRINT_CALL("preprocess_split", "input", "X", "training", "X_train", "test",
        "X_test", "test_ratio", 0.4, "stratify_data", true));

// See also...
BINDING_SEE_ALSO("@preprocess_binarize", "#preprocess_binarize");
BINDING_SEE_ALSO("@preprocess_describe", "#preprocess_describe");
#if BINDING_TYPE == BINDING_TYPE_CLI
BINDING_SEE_ALSO("@preprocess_imputer", "#preprocess_imputer");
#endif

// Define parameters for data.
PARAM_MATRIX_IN_REQ("input", "Matrix containing data.", "i");
PARAM_MATRIX_OUT("training", "Matrix to save training data to.", "t");
PARAM_MATRIX_OUT("test", "Matrix to save test data to.", "T");

// Define optional parameters.
PARAM_UMATRIX_IN("input_labels", "Matrix containing labels.", "I");
PARAM_UMATRIX_OUT("training_labels", "Matrix to save train labels to.", "l");
PARAM_UMATRIX_OUT("test_labels", "Matrix to save test labels to.", "L");

// Define optional test ratio, default is 0.2 (Test 20% Train 80%).
PARAM_DOUBLE_IN("test_ratio", "Ratio of test set; if not set,"
    "the ratio defaults to 0.2", "r", 0.2);

PARAM_INT_IN("seed", "Random seed (0 for std::time(NULL)).", "s", 0);
PARAM_FLAG("no_shuffle", "Avoid shuffling the data before splitting.", "S");
PARAM_FLAG("stratify_data", "Stratify the data according to labels", "z")

using namespace mlpack;
using namespace mlpack::data;
using namespace mlpack::util;
using namespace arma;
using namespace std;

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  // Parse command line options.
  const double testRatio = params.Get<double>("test_ratio");
  const bool shuffleData = params.Get<bool>("no_shuffle");
  const bool stratifyData = params.Get<bool>("stratify_data");

  if (params.Get<int>("seed") == 0)
    RandomSeed(std::time(NULL));
  else
    RandomSeed((size_t) params.Get<int>("seed"));

  // Make sure the user specified output filenames.
  RequireAtLeastOnePassed(params, { "training" }, false, "no training set will "
      "be saved");
  RequireAtLeastOnePassed(params, { "test" }, false, "no test set will be "
      "saved");

  // Check on label parameters.
  if (params.Has("input_labels"))
  {
    RequireAtLeastOnePassed(params, { "training_labels" }, false, "no training "
        "set labels will be saved");
    RequireAtLeastOnePassed(params, { "test_labels" }, false, "no test set "
        "labels will be saved");
  }
  else
  {
    ReportIgnoredParam(params, {{ "input_labels", true }}, "training_labels");
    ReportIgnoredParam(params, {{ "input_labels", true }}, "test_labels");
  }

  // Check test_ratio.
  RequireParamValue<double>(params, "test_ratio",
      [](double x) { return x >= 0.0 && x <= 1.0; }, true,
      "test ratio must be between 0.0 and 1.0");

  // Load the data.
  arma::mat& data = params.Get<arma::mat>("input");

  // If parameters for labels exist, we must split the labels too.
  if (params.Has("input_labels"))
  {
    arma::Mat<size_t>& labels =
        params.Get<arma::Mat<size_t>>("input_labels");
    arma::Row<size_t> labelsRow = labels.row(0);

    timers.Start("splitting_data");
    arma::mat trainData, testData;
    arma::Row<size_t> trainLabels, testLabels;
    if (stratifyData)
    {
      data::StratifiedSplit(data, labelsRow, trainData, testData, trainLabels,
          testLabels, testRatio, !shuffleData);
    }
    else
    {
      data::Split(data, labelsRow, trainData, testData, trainLabels, testLabels,
          testRatio, !shuffleData);
    }
    timers.Stop("splitting_data");

    Log::Info << "Training data contains " << trainData.n_cols << " points."
        << endl;
    Log::Info << "Test data contains " << testData.n_cols << " points." << endl;

    if (params.Has("training"))
      params.Get<arma::mat>("training") = std::move(trainData);
    if (params.Has("test"))
      params.Get<arma::mat>("test") = std::move(testData);
    if (params.Has("training_labels"))
      params.Get<arma::Mat<size_t>>("training_labels") = std::move(trainLabels);
    if (params.Has("test_labels"))
      params.Get<arma::Mat<size_t>>("test_labels") = std::move(testLabels);
  }
  else // We have no labels, so just split the dataset.
  {
    timers.Start("splitting_data");
    arma::mat trainData, testData;
    data::Split(data, trainData, testData, testRatio, !shuffleData);
    timers.Stop("splitting_data");

    Log::Info << "Training data contains " << trainData.n_cols << " points."
        << endl;
    Log::Info << "Test data contains " << testData.n_cols << " points." << endl;

    if (params.Has("training"))
      params.Get<arma::mat>("training") = std::move(trainData);
    if (params.Has("test"))
      params.Get<arma::mat>("test") = std::move(testData);
  }
}
