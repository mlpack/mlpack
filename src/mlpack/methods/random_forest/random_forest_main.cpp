/**
 * @file methods/random_forest/random_forest_main.cpp
 * @author Ryan Curtin
 *
 * A program to build and evaluate random forests.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME random_forest

#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/methods/random_forest/random_forest.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

// Program Name.
BINDING_USER_NAME("Random forests");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of the standard random forest algorithm by Leo Breiman "
    "for classification.  Given labeled data, a random forest can be trained "
    "and saved for future use; or, a pre-trained random forest can be used for "
    "classification.");

// Long description.
BINDING_LONG_DESC(
    "This program is an implementation of the standard random forest "
    "classification algorithm by Leo Breiman.  A random forest can be "
    "trained and saved for later use, or a random forest may be loaded "
    "and predictions or class probabilities for points may be generated."
    "\n\n"
    "The training set and associated labels are specified with the " +
    PRINT_PARAM_STRING("training") + " and " + PRINT_PARAM_STRING("labels") +
    " parameters, respectively.  The labels should be in the range `[0, "
    "num_classes - 1]`. Optionally, if " +
    PRINT_PARAM_STRING("labels") + " is not specified, the labels are assumed "
    "to be the last dimension of the training dataset."
    "\n\n"
    "When a model is trained, the " + PRINT_PARAM_STRING("output_model") + " "
    "output parameter may be used to save the trained model.  A model may be "
    "loaded for predictions with the " + PRINT_PARAM_STRING("input_model") +
    "parameter. The " + PRINT_PARAM_STRING("input_model") + " parameter may "
    "not be specified when the " + PRINT_PARAM_STRING("training") + " parameter"
    " is specified.  The " + PRINT_PARAM_STRING("minimum_leaf_size") +
    " parameter specifies the minimum number of training points that must fall "
    "into each leaf for it to be split.  The " +
    PRINT_PARAM_STRING("num_trees") +
    " controls the number of trees in the random forest.  The " +
    PRINT_PARAM_STRING("minimum_gain_split") + " parameter controls the minimum"
    " required gain for a decision tree node to split.  Larger values will "
    "force higher-confidence splits.  The " +
    PRINT_PARAM_STRING("maximum_depth") + " parameter specifies "
    "the maximum depth of the tree.  The " +
    PRINT_PARAM_STRING("subspace_dim") + " parameter is used to control the "
    "number of random dimensions chosen for an individual node's split.  If " +
    PRINT_PARAM_STRING("print_training_accuracy") + " is specified, the "
    "calculated accuracy on the training set will be printed."
    "\n\n"
    "Test data may be specified with the " + PRINT_PARAM_STRING("test") + " "
    "parameter, and if performance measures are desired for that test set, "
    "labels for the test points may be specified with the " +
    PRINT_PARAM_STRING("test_labels") + " parameter.  Predictions for each "
    "test point may be saved via the " + PRINT_PARAM_STRING("predictions") +
    "output parameter.  Class probabilities for each prediction may be saved "
    "with the " + PRINT_PARAM_STRING("probabilities") + " output parameter.");

// Example.
BINDING_EXAMPLE(
    "For example, to train a random forest with a minimum leaf size of 20 "
    "using 10 trees on the dataset contained in " + PRINT_DATASET("data") +
    "with labels " + PRINT_DATASET("labels") + ", saving the output random "
    "forest to " + PRINT_MODEL("rf_model") + " and printing the training "
    "error, one could call"
    "\n\n" +
    PRINT_CALL("random_forest", "training", "data", "labels", "labels",
        "minimum_leaf_size", 20, "num_trees", 10, "output_model", "rf_model",
        "print_training_accuracy", true) +
    "\n\n"
    "Then, to use that model to classify points in " +
    PRINT_DATASET("test_set") + " and print the test error given the labels " +
    PRINT_DATASET("test_labels") + " using that model, while saving the "
    "predictions for each point to " + PRINT_DATASET("predictions") + ", one "
    "could call "
    "\n\n" +
    PRINT_CALL("random_forest", "input_model", "rf_model", "test", "test_set",
        "test_labels", "test_labels", "predictions", "predictions"));

// See also...
BINDING_SEE_ALSO("@decision_tree", "#decision_tree");
BINDING_SEE_ALSO("@hoeffding_tree", "#hoeffding_tree");
BINDING_SEE_ALSO("@softmax_regression", "#softmax_regression");
BINDING_SEE_ALSO("Random forest on Wikipedia",
    "https://en.wikipedia.org/wiki/Random_forest");
BINDING_SEE_ALSO("Random forests (pdf)", "https://www.eecis.udel.edu/~shatkay"
    "/Course/papers/BreimanRandomForests2001.pdf");
BINDING_SEE_ALSO("RandomForest C++ class documentation",
    "@doc/user/methods/random_forest.md");

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
    "node.", "n", 1);
PARAM_INT_IN("maximum_depth", "Maximum depth of the tree (0 means no limit).",
    "D", 0);
PARAM_MATRIX_OUT("probabilities", "Predicted class probabilities for each "
    "point in the test set.", "P");
PARAM_UROW_OUT("predictions", "Predicted classes for each point in the test "
    "set.", "p");

PARAM_DOUBLE_IN("minimum_gain_split", "Minimum gain needed to make a split "
    "when building a tree.", "g", 0);
PARAM_INT_IN("subspace_dim", "Dimensionality of random subspace to use for "
    "each split.  '0' will autoselect the square root of data dimensionality.",
    "d", 0);

PARAM_INT_IN("seed", "Random seed.  If 0, 'std::time(NULL)' is used.", "s", 0);
PARAM_FLAG("warm_start", "If true and passed along with `training` and "
    "`input_model` then trains more trees on top of existing model.", "w");

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
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(rf));
  }
};

PARAM_MODEL_IN(RandomForestModel, "input_model", "Pre-trained random forest to "
    "use for classification.", "m");
PARAM_MODEL_OUT(RandomForestModel, "output_model", "Model to save trained "
    "random forest to.", "M");

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  // Initialize random seed if needed.
  if (params.Get<int>("seed") != 0)
    RandomSeed((size_t) params.Get<int>("seed"));
  else
    RandomSeed((size_t) std::time(NULL));

  // Check for incompatible input parameters.
  if (!params.Has("warm_start"))
  {
    RequireOnlyOnePassed(params, { "training", "input_model" }, true);
  }
  else
  {
    // When warm_start is passed, training and input_model must also be passed.
    RequireNoneOrAllPassed(params, {"warm_start", "training", "input_model"},
        true);
  }

  ReportIgnoredParam(params, {{ "training", false }},
      "print_training_accuracy");
  ReportIgnoredParam(params, {{ "test", false }}, "test_labels");

  RequireAtLeastOnePassed(params, { "test", "output_model",
      "print_training_accuracy" }, false, "the trained forest model will not "
      "be used or saved");

  if (params.Has("training"))
  {
    RequireAtLeastOnePassed(params, { "labels" }, true, "must pass labels when "
        "training set given");
  }

  RequireParamValue<int>(params, "num_trees", [](int x) { return x > 0; }, true,
      "number of trees in forest must be positive");

  ReportIgnoredParam(params, {{ "test", false }}, "predictions");
  ReportIgnoredParam(params, {{ "test", false }}, "probabilities");

  RequireParamValue<int>(params, "minimum_leaf_size",
      [](int x) { return x > 0; }, true, "minimum leaf size must be greater "
      "than 0");
  RequireParamValue<int>(params, "maximum_depth", [](int x) { return x >= 0; },
      true, "maximum depth must not be negative");
  RequireParamValue<int>(params, "subspace_dim", [](int x) { return x >= 0; },
      true, "subspace dimensionality must be nonnegative");
  RequireParamValue<double>(params, "minimum_gain_split",
      [](double x) { return x >= 0.0; }, true,
      "minimum gain for splitting must be nonnegative");

  ReportIgnoredParam(params, {{ "training", false }}, "num_trees");
  ReportIgnoredParam(params, {{ "training", false }}, "minimum_leaf_size");

  RandomForestModel* rfModel;
  // Input model is loaded when we are either doing warm-started training or
  // else we are making predictions only or both.
  if (params.Has("input_model"))
    rfModel = params.Get<RandomForestModel*>("input_model");
  // Handles the case when we are training new forest from scratch.
  else
    rfModel = new RandomForestModel();

  if (params.Has("training"))
  {
    timers.Start("rf_training");

    // Train the model on the given input data.
    arma::mat data = std::move(params.Get<arma::mat>("training"));
    arma::Row<size_t> labels =
        std::move(params.Get<arma::Row<size_t>>("labels"));

    // Make sure the subspace dimensionality is valid.
    RequireParamValue<int>(params, "subspace_dim",
        [data](int x) { return (size_t) x <= data.n_rows; }, true, "subspace "
        "dimensionality must not be greater than data dimensionality");

    const size_t numTrees = (size_t) params.Get<int>("num_trees");
    const size_t minimumLeafSize =
        (size_t) params.Get<int>("minimum_leaf_size");
    const size_t maxDepth = (size_t) params.Get<int>("maximum_depth");
    const double minimumGainSplit = params.Get<double>("minimum_gain_split");
    const size_t randomDims = (params.Get<int>("subspace_dim") == 0) ?
        (size_t) std::sqrt(data.n_rows) :
        (size_t) params.Get<int>("subspace_dim");
    MultipleRandomDimensionSelect mrds(randomDims);

    Log::Info << "Training random forest with " << numTrees << " trees..."
        << endl;

    const size_t numClasses = max(labels) + 1;

    // Train the model.
    rfModel->rf.Train(data, labels, numClasses, numTrees, minimumLeafSize,
        minimumGainSplit, maxDepth, params.Has("warm_start"), mrds);

    timers.Stop("rf_training");

    // Did we want training accuracy?
    if (params.Has("print_training_accuracy"))
    {
      timers.Start("rf_prediction");
      arma::Row<size_t> predictions;
      rfModel->rf.Classify(data, predictions);

      const size_t correct = accu(predictions == labels);

      Log::Info << correct << " of " << labels.n_elem << " correct on training"
          << " set (" << (double(correct) / double(labels.n_elem) * 100) << ")."
          << endl;
      timers.Stop("rf_prediction");
    }
  }

  if (params.Has("test"))
  {
    arma::mat testData = std::move(params.Get<arma::mat>("test"));
    timers.Start("rf_prediction");

    // Get predictions and probabilities.
    arma::Row<size_t> predictions;
    arma::mat probabilities;
    rfModel->rf.Classify(testData, predictions, probabilities);

    // Did we want to calculate test accuracy?
    if (params.Has("test_labels"))
    {
      arma::Row<size_t> testLabels =
          std::move(params.Get<arma::Row<size_t>>("test_labels"));

      const size_t correct = accu(predictions == testLabels);

      Log::Info << correct << " of " << testLabels.n_elem << " correct on test"
          << " set (" << (double(correct) / double(testLabels.n_elem) * 100)
          << ")." << endl;
      timers.Stop("rf_prediction");
    }

    // Save the outputs.
    params.Get<arma::mat>("probabilities") = std::move(probabilities);
    params.Get<arma::Row<size_t>>("predictions") = std::move(predictions);
  }

  // Save the output model.
  params.Get<RandomForestModel*>("output_model") = rfModel;
}
