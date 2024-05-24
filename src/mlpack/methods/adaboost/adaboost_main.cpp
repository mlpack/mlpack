/**
 * @file methods/adaboost/adaboost_main.cpp
 * @author Udit Saxena
 *
 * Implementation of the AdaBoost main program.
 *
 * @code
 * @article{Schapire:1999:IBA:337859.337870,
 *   author = {Schapire, Robert E. and Singer, Yoram},
 *   title = {Improved Boosting Algorithms Using Confidence-rated Predictions},
 *   journal = {Machine Learning},
 *   issue_date = {Dec. 1999},
 *   volume = {37},
 *   number = {3},
 *   month = dec,
 *   year = {1999},
 *   issn = {0885-6125},
 *   pages = {297--336},
 *   numpages = {40},
 *   url = {http://dx.doi.org/10.1023/A:1007614523901},
 *   doi = {10.1023/A:1007614523901},
 *   acmid = {337870},
 *   publisher = {Kluwer Academic Publishers},
 *   address = {Hingham, MA, USA},
 *   keywords = {boosting algorithms, decision trees, multiclass classification,
 *   output coding}
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME adaboost

#include <mlpack/core/util/mlpack_main.hpp>

#include "adaboost.hpp"
#include "adaboost_model.hpp"

using namespace mlpack;
using namespace std;
using namespace arma;
using namespace mlpack::util;

// Program Name.
BINDING_USER_NAME("AdaBoost");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of the AdaBoost.MH (Adaptive Boosting) algorithm for "
    "classification.  This can be used to train an AdaBoost model on labeled "
    "data or use an existing AdaBoost model to predict the classes of new "
    "points.");

// Long description.
BINDING_LONG_DESC(
    "This program implements the AdaBoost (or Adaptive "
    "Boosting) algorithm. The variant of AdaBoost implemented here is "
    "AdaBoost.MH. It uses a weak learner, either decision stumps or "
    "perceptrons, and over many iterations, creates a strong learner that is a "
    "weighted ensemble of weak learners. It runs these iterations until a "
    "tolerance value is crossed for change in the value of the weighted "
    "training error."
    "\n\n"
    "For more information about the algorithm, see the paper \"Improved "
    "Boosting Algorithms Using Confidence-Rated Predictions\", by R.E. Schapire"
    " and Y. Singer."
    "\n\n"
    "This program allows training of an AdaBoost model, and then application of"
    " that model to a test dataset.  To train a model, a dataset must be passed"
    " with the " + PRINT_PARAM_STRING("training") + " option.  Labels can be "
    "given with the " + PRINT_PARAM_STRING("labels") + " option; if no labels "
    "are specified, the labels will be assumed to be the last column of the "
    "input dataset.  Alternately, an AdaBoost model may be loaded with the " +
    PRINT_PARAM_STRING("input_model") + " option."
    "\n\n"
    "Once a model is trained or loaded, it may be used to provide class "
    "predictions for a given test dataset.  A test dataset may be specified "
    "with the " + PRINT_PARAM_STRING("test") + " parameter.  The predicted "
    "classes for each point in the test dataset are output to the " +
    PRINT_PARAM_STRING("predictions") + " output parameter.  The AdaBoost "
    "model itself is output to the " + PRINT_PARAM_STRING("output_model") +
    " output parameter.");

// Example.
BINDING_EXAMPLE(
    "For example, to run AdaBoost on an input dataset " +
    PRINT_DATASET("data") + " with labels " + PRINT_DATASET("labels") +
    "and perceptrons as the weak learner type, storing the trained model in " +
    PRINT_MODEL("model") + ", one could use the following command: "
    "\n\n" +
    PRINT_CALL("adaboost", "training", "data", "labels", "labels",
        "output_model", "model", "weak_learner", "perceptron") +
    "\n\n"
    "Similarly, an already-trained model in " + PRINT_MODEL("model") + " can"
    " be used to provide class predictions from test data " +
    PRINT_DATASET("test_data") + " and store the output in " +
    PRINT_DATASET("predictions") + " with the following command: "
    "\n\n" +
    PRINT_CALL("adaboost", "input_model", "model", "test", "test_data",
        "predictions", "predictions"));

// See also...
BINDING_SEE_ALSO("AdaBoost on Wikipedia", "https://en.wikipedia.org/wiki/"
    "AdaBoost");
BINDING_SEE_ALSO("Improved boosting algorithms using confidence-rated "
    "predictions (pdf)", "http://rob.schapire.net/papers/SchapireSi98.pdf");
BINDING_SEE_ALSO("Perceptron", "#perceptron");
BINDING_SEE_ALSO("Decision Trees", "#decision_tree");
BINDING_SEE_ALSO("AdaBoost C++ class documentation",
    "@doc/user/methods/adaboost.md");

// Input for training.
PARAM_MATRIX_IN("training", "Dataset for training AdaBoost.", "t");
PARAM_UROW_IN("labels", "Labels for the training set.", "l");

// Classification options.
PARAM_MATRIX_IN("test", "Test dataset.", "T");
PARAM_UROW_OUT("predictions", "Predicted labels for the test set.", "P");
PARAM_MATRIX_OUT("probabilities", "Predicted class probabilities for each "
    "point in the test set.", "p");

// Training options.
PARAM_INT_IN("iterations", "The maximum number of boosting iterations to be run"
    " (0 will run until convergence.)", "i", 1000);
PARAM_DOUBLE_IN("tolerance", "The tolerance for change in values of the "
    "weighted error during training.", "e", 1e-10);
PARAM_STRING_IN("weak_learner", "The type of weak learner to use: "
    "'decision_stump', or 'perceptron'.", "w", "decision_stump");

// Loading/saving of a model.
PARAM_MODEL_IN(AdaBoostModel, "input_model", "Input AdaBoost model.", "m");
PARAM_MODEL_OUT(AdaBoostModel, "output_model", "Output trained AdaBoost model.",
    "M");

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  // Check input parameters and issue warnings/errors as necessary.

  // The user cannot specify both a training file and an input model file.
  RequireOnlyOnePassed(params, { "training", "input_model" });

  // The weak learner must make sense.
  RequireParamInSet<std::string>(params, "weak_learner",
      { "decision_stump", "perceptron" }, true, "unknown weak learner type");

  // --labels can't be specified without --training.
  ReportIgnoredParam(params, {{ "training", false }}, "labels");

  // Sanity check on iterations.
  RequireParamValue<int>(params, "iterations", [](int x) { return x > 0; },
      true, "invalid number of iterations specified");

  // If a weak learner is specified with a model, it will be ignored.
  ReportIgnoredParam(params, {{ "input_model", true }}, "weak_learner");

  // Training parameters are ignored if no training file is given.
  ReportIgnoredParam(params, {{ "training", false }}, "tolerance");
  ReportIgnoredParam(params, {{ "training", false }}, "iterations");

  // If we gave an input model but no test set, issue a warning.
  if (params.Has("input_model"))
  {
    RequireAtLeastOnePassed(params, { "test" }, false,
        "no task will be performed");
  }

  RequireAtLeastOnePassed(params, { "output_model", "predictions" }, false,
      "no results will be saved");

  ReportIgnoredParam(params, {{ "test", false }}, "predictions");

  AdaBoostModel* m;
  if (params.Has("training"))
  {
    mat trainingData = std::move(params.Get<arma::mat>("training"));
    m = new AdaBoostModel();

    // Load labels.
    arma::Row<size_t> labelsIn;

    if (params.Has("labels"))
    {
      // Load labels.
      labelsIn = std::move(params.Get<arma::Row<size_t>>("labels"));
    }
    else
    {
      // Extract the labels as the last dimension of the training data.
      Log::Info << "Using the last dimension of training set as labels."
          << endl;
      labelsIn = ConvTo<arma::Row<size_t>>::From(
          trainingData.row(trainingData.n_rows - 1));
      trainingData.shed_row(trainingData.n_rows - 1);
    }

    // Helpers for normalizing the labels.
    Row<size_t> labels;

    // Normalize the labels.
    data::NormalizeLabels(labelsIn, labels, m->Mappings());

    // Get other training parameters.
    const double tolerance = params.Get<double>("tolerance");
    const size_t iterations = (size_t) params.Get<int>("iterations");
    const string weakLearner = params.Get<string>("weak_learner");
    if (weakLearner == "decision_stump")
      m->WeakLearnerType() = AdaBoostModel::WeakLearnerTypes::DECISION_STUMP;
    else if (weakLearner == "perceptron")
      m->WeakLearnerType() = AdaBoostModel::WeakLearnerTypes::PERCEPTRON;

    const size_t numClasses = m->Mappings().n_elem;
    Log::Info << numClasses << " classes in dataset." << endl;

    timers.Start("adaboost_training");
    m->Train(trainingData, labels, numClasses, iterations, tolerance);
    timers.Stop("adaboost_training");
  }
  else
  {
    // We have a specified input model.
    m = params.Get<AdaBoostModel*>("input_model");
  }

  // Perform classification, if desired.
  if (params.Has("test"))
  {
    mat testingData = std::move(params.Get<arma::mat>("test"));

    if (testingData.n_rows != m->Dimensionality())
      Log::Fatal << "Test data dimensionality (" << testingData.n_rows << ") "
          << "must be the same as the model dimensionality ("
          << m->Dimensionality() << ")!" << endl;

    Row<size_t> predictedLabels(testingData.n_cols);
    mat probabilities;

    if (params.Has("probabilities"))
    {
      timers.Start("adaboost_classification");
      m->Classify(testingData, predictedLabels, probabilities);
      timers.Stop("adaboost_classification");
    }
    else
    {
      timers.Start("adaboost_classification");
      m->Classify(testingData, predictedLabels);
      timers.Stop("adaboost_classification");
    }

    Row<size_t> results;
    data::RevertLabels(predictedLabels, m->Mappings(), results);

    // Save the predicted labels.
    if (params.Has("predictions"))
      params.Get<arma::Row<size_t>>("predictions") = std::move(results);
    if (params.Has("probabilities"))
      params.Get<arma::mat>("probabilities") = std::move(probabilities);
  }

  params.Get<AdaBoostModel*>("output_model") = m;
}
