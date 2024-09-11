/**
 * @file methods/adaboost/adaboost_train_main.cpp
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
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/io.hpp>

#undef BINDING_NAME
#define BINDING_NAME adaboost_train

#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/core/data/normalize_labels.hpp>
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
    "Training AdaBoost model.");

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
    " and Y. Singer.");

// Example.
BINDING_EXAMPLE(
    IMPORT_EXT_LIB() + "\n" +
    IMPORT_SPLIT() + "\n" +
    IMPORT_THIS("adaboost") + "\n" +
    GET_DATASET("X", "https://example.com") + "\n" +
    GET_DATASET("y", "https://example.com") + "\n" +
    SPLIT_TRAIN_TEST("X", "y", "X_train", "y_train", "X_test", "y_test",
    "0.2") + "\n" +
    CREATE_OBJECT("model", "adaboost") + "\n" +
    CALL_METHOD("model", "train", "training", "X_train", "labels", "y_train"));

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
PARAM_MATRIX_IN_REQ("training", "Dataset for training AdaBoost.", "t");
PARAM_UROW_IN("labels", "Labels for the training set.", "l");

// Training options.
PARAM_INT_IN("iterations", "The maximum number of boosting iterations to be run"
    " (0 will run until convergence.)", "i", 1000);
PARAM_DOUBLE_IN("tolerance", "The tolerance for change in values of the "
    "weighted error during training.", "e", 1e-10);
PARAM_STRING_IN("weak_learner", "The type of weak learner to use: "
    "'decision_stump', or 'perceptron'.", "w", "decision_stump");

// Allow saving the model.
PARAM_MODEL_OUT(AdaBoostModel, "output_model", "Output trained AdaBoost model.",
    "M");

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  // The weak learner must make sense.
  RequireParamInSet<std::string>(params, "weak_learner",
      { "decision_stump", "perceptron" }, true, "unknown weak learner type");

  // Sanity check on iterations.
  RequireParamValue<int>(params, "iterations", [](int x) { return x > 0; },
      true, "invalid number of iterations specified");

  // Sanity check on tolerance value.
  RequireParamValue<double>(params, "tolerance", [](double x) { return x > 0; },
      true, "invalid tolerance specified");

  mat trainingData = std::move(params.Get<arma::mat>("training"));
  AdaBoostModel* m = new AdaBoostModel();

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
    if (trainingData.n_rows < 2)
    {
      Log::Fatal << "Cannot extract labels from last dimension as total "
          << "dimensions are less than 2." << endl;
    }

    labelsIn = ConvTo<arma::Row<size_t>>::From(
        trainingData.row(trainingData.n_rows - 1));
    trainingData.shed_row(trainingData.n_rows - 1);
  }

  if (labelsIn.n_cols != trainingData.n_cols)
  {
    Log::Fatal << "Number of samples in training data is not "
        << "equal to the number of samples in labels." << endl;
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

  params.Get<AdaBoostModel*>("output_model") = m;
}
