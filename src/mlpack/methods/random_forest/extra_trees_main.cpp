/**
 * @file methods/random_forest/extra_trees_main.cpp
 * @author Pranshu Srivastava
 *
 * A program to build and evaluate extra trees.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>
#include <mlpack/core/util/io.hpp>

#ifdef BINDING_NAME
  #undef BINDING_NAME
#endif
#define BINDING_NAME extra_trees

#include <mlpack/methods/random_forest/random_forest.hpp>
#include <mlpack/methods/decision_tree/random_dimension_select.hpp>
#include <mlpack/core/util/mlpack_main.hpp>

using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::util;
using namespace std;
using namespace arma;

// Program Name.
BINDING_USER_NAME("Extra Trees");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of the  Extra Trees algorithm"
    "for classification. Given labelled data, an extra tree can be trained"
    "and saved for future use; or,  a pre-trained extra tree can be used for"
    "classification.");
// Long description.
BINDING_LONG_DESC(
    "This program is an implementation of the extra trees algorithm. "
    "The extra trees algorithm was developed by Geurts,Ernst and Wehenkel"
    "and can be used as an alternative to the  random forest algorithm."
    "It splits the dataset using a random split and does not bootstrap "
    "the features of the dataset."
    "It can be trained and saved for later use, or it maybe loaded and class"
    "probabilities for points may be generated."
    "\n\n"
    "The training set and the associated labels are specified with the" +
    PRINT_PARAM_STRING("training") + "and" + PRINT_PARAM_STRING("labels") +
    "parameters, respectively. The labels should be in the range of [0,"
    "num_classes -1].Optionally if" +
    PRINT_PARAM_STRING("labels") + "is not specified, the labels are assumed"
    "to be the last dimension of the training dataset."
    "\n\n"
    "When a model is trained, the " +
    PRINT_PARAM_STRING("output_model") + " "
    "output parameter may be used to save the trained model. A model may be"
    "loaded for predictions with the" +
    PRINT_PARAM_STRING("input_model") +"parameter. "
    "The " + PRINT_PARAM_STRING("input_model") + "parameter may not be"
    "specified when the " +
    PRINT_PARAM_STRING("training ") + "parameter"
    "is specified. The " +
    PRINT_PARAM_STRING("minimum_leaf_size") +
    "parameter specifies the minimum number of training points that must fall"
    "into each leaf for it to be split. The " +
    PRINT_PARAM_STRING("maximum_depth") +
    "parameter specifies the maximum depth of a tree.  If " +
    PRINT_PARAM_STRING("print_training_accuracy") + " is specified, the "
    "calculated accuracy on the training set will be printed."
    "\n\n"
    "Test data may be specified with the " +
    PRINT_PARAM_STRING("test") + " "
    "parameter, and if performance measures are desired for that test set, "
    "labels for the test points may be specified with the " +
    PRINT_PARAM_STRING("test_labels") + " parameter.  Predictions for each "
    "test point may be saved via the " +
    PRINT_PARAM_STRING("predictions") +
    "output parameter.  Class probabilities for each prediction may be saved "
    "with the " +
    PRINT_PARAM_STRING("probabilities") + " output parameter.");

// Example.
BINDING_EXAMPLE(
    "For example, to train a extra trees algorithm with a minimum leaf size of"
    "20 using 10 trees on the dataset contained in " +
    PRINT_DATASET("data") +"with labels " +
    PRINT_DATASET("labels") + ", saving the output random "
    "forest to " +
    PRINT_MODEL("et_model") + " and printing the training "
    "error, one could call"
    "\n\n" +
    PRINT_CALL("extra_trees", "training", "data", "labels", "labels",
        "minimum_leaf_size", 20, "num_trees", 10, "output_model",
        "rf_model", "print_training_accuracy", true) +
        "\n\n"
    "Then, to use that model to classify points in " +
    PRINT_DATASET("test_set") + " and print the test error given the labels" +
    PRINT_DATASET("test_labels") + " using that model, while saving the "
    "predictions for each point to " +
    PRINT_DATASET("predictions") + ", one "
    "could call "
    "\n\n" +
    PRINT_CALL("extra_trees", "input_model", "et_model", "test", "test_set",
        "test_labels", "test_labels", "predictions", "predictions"));

//See also..
BINDING_SEE_ALSO("@decision_tree", "#decision_tree");
BINDING_SEE_ALSO("@hoeffding_tree", "#hoeffding_tree");
BINDING_SEE_ALSO("@softmax_regression", "#softmax_regression");
BINDING_SEE_ALSO("Extra Trees on Wikipedia",
     "https://en.wikipedia.org/wiki/Random_forest#ExtraTrees");
PARAM_MATRIX_IN("training", "Training dataset.", "t");
PARAM_UROW_IN("labels", "Labels for training dataset.", "l");
PARAM_MATRIX_IN("test", "Test dataset to produce predictions for.", "T");
PARAM_UROW_IN("test_labels", "Test dataset labels, if accuracy calculation is"
        "desired.", "L");
PARAM_MATRIX_IN("training", "Training dataset.", "t");
PARAM_UROW_IN("labels", "Labels for training dataset.", "l");
PARAM_MATRIX_IN("test", "Test dataset to produce predictions for.", "T");
PARAM_UROW_IN("test_labels", "Test dataset labels, if accuracy calculation is"
    "desired.","L");

PARAM_FLAG("print_training_accuracy", "If set, then the accuracy of the model"
    "on the training set will be predicted (verbose must also be"
    "specified).","a");

PARAM_INT_IN("num_trees", "Number of trees in the extra trees.", "N", 10);
PARAM_INT_IN("minimum_leaf_size", "Minimum number of points in each leaf"
    "node.","n", 1);
PARAM_INT_IN("maximum_depth", "Maximum depth of the tree (0 means no limit).",
    "D", 0);
PARAM_MATRIX_OUT("probabilities", "Predicted class probabilities for each"
    "point in the test set.","P");
PARAM_UROW_OUT("predictions", "Predicted classes for each point in the test"
    "set.","p");

PARAM_DOUBLE_IN("minimum_gain_split", "Minimum gain needed to make a split"
    "when building a tree.","g", 0);
PARAM_INT_IN("subspace_dim", "Dimensionality of random subspace to use for"
    "each split.  '0' will autoselect the square root of data"
    "dimensionality.","d", 0);

PARAM_INT_IN("seed", "Random seed.  If 0, 'std::time(NULL)' is used.", "s", 0);
PARAM_FLAG("warm_start", "If true and passed along with `training` and "
    "`input_model` then trains more trees on top of existing model.","w");

class ExtraTreesModel
{
  public:
   // The tree itself.
   ExtraTrees<> et;

   //Creating the model.
   ExtraTreesModel()
   { /* Nothing yo do. */
   }

  // Serialize the model.
  template <typename Archive>
  void serialize(Archive &ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(et));
  }
};

PARAM_MODEL_IN(ExtraTreesModel, "input_model", "Pre-trained extra trees "
    "algorithm to use for classification.","m");
PARAM_MODEL_OUT(ExtraTreesModel, "output_model", "Model to save trained "
    "extra trees to.", "M");

void BINDING_FUNCTION(util::Params& params, util::Timers&)
{
  // When warm start is not passed, either the training or the input model
  //needs to be passed.
  if (!params.Has("warm_start"))
    RequireOnlyOnePassed(params, { "training", "input_model"}, true);
  // When warm_start is passed, training and input_model must also be passed.
  else
    RequireNoneOrAllPassed(params, {"warm_start", "training", "input_model"}, true);

  RequireNoneOrAllPassed(params,{"warm_start", "training", "input_model"}, true);

  ReportIgnoredParam(params, {{"training", false}}, "print_training_accuracy");
  ReportIgnoredParam(params, {{"test", false}}, "test_labels");

  RequireAtLeastOnePassed(params,{ "test", "output_model", "print_training_accuracy"},
    false, "the trained forest model will not be used or saved");
  if (params.Has("training"))
  {
    RequireAtLeastOnePassed(params,{ "labels" }, true, "must pass labels when"
        "training set is given");
  }
  RequireParamValue<int>( params,
    "num_trees", [](int x)
    { return x > 0; },
    true,
    "number of trees in forest must be positive");

  ReportIgnoredParam(params, {{"test", false}}, "predictions");
  ReportIgnoredParam(params, {{"test", false}}, "probabilities");

  RequireParamValue<int>(params,
    "minimum_leaf_size", [](int x)
    { return x > 0; },
    true,
    "minimum leaf size must be greater than 0");
  RequireParamValue<int>(params,
    "maximum_depth", [](int x)
    { return x >= 0; },
    true,
    "maximum depth must not be negative");
  RequireParamValue<int>(params,
    "subspace_dim", [](int x)
    { return x >= 0; },
    true,
    "subspace dimensionality must be nonnegative");
  RequireParamValue<double>(params,
    "minimum_gain_split",
    [](double x)
    { return x >= 0.0; },
    true,
    "minimum gain for splitting must be nonnegative");
  ReportIgnoredParam(params, {{"training", false}}, "minimum_leaf_size");
  ExtraTreesModel *etModel;

  if (params.Has("input_model"))
    etModel = params.Get<ExtraTreesModel*>("input_model");
  else
    etModel = new ExtraTreesModel();

  if (params.Has("training"))
  {
    Timer::Start("et_training");
    // Training the model on the given input data.
    mat data = std::move(params.Get<mat>("training"));
    Row<size_t> labels = std::move(params.Get<Row<size_t>>("labels"));
    // Make sure the subspace dimensionality is valid.
    RequireParamValue<int>(params, "subspace_dim",
        [data](int x){ return (size_t)x <= data.n_rows; },
        true, "subspace "
        "dimensionality must not be greater than data dimensionality");
    const size_t numTrees = (size_t)params.Get<int>("num_trees");
    const size_t minimumLeafSize =
        (size_t)params.Get<int>("minimum_leaf_size");
    const size_t maxDepth = (size_t)params.Get<int>("maximum_depth");
    const double minimumGainSplit = params.Get<double>
        ("minimum_gain_split");
    const size_t randomDims = (params.Get<int>("subspace_dim") == 0) ?
        (size_t)std::sqrt(data.n_rows) :
        (size_t)params.Get<int>("subspace_dim");
        MultipleRandomDimensionSelect mrds(randomDims);

    Log::Info << "Training extra trees with " << numTrees << " trees..."
        << endl;
    const size_t numClasses = max(labels) + 1;
    // Training the model.
    etModel->et.Train(data, labels, numClasses, numTrees, minimumLeafSize,
        minimumGainSplit, maxDepth, params.Has("warm_start"));
    Timer::Stop("et_training");
  }
  if (params.Has("test"))
  {
    mat testData = move(params.Get<mat>("test"));
    Timer::Start("rf_prediction");
    // Get Predictions and probabilities.
    arma::Row<size_t> predictions;
    arma::mat probabilities;
    etModel->et.Classify(testData, predictions, probabilities);

    // Save the outputs.
    params.Get<arma::mat>("probabilities") = std::move(probabilities);
    params.Get<arma::Row<size_t>>("predictions") = std::move(predictions);
    }
    // Save the output model.
    params.Get<ExtraTreesModel*>("output_model") = etModel;
}
