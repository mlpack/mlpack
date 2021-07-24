/**
 * @file methods/adaboost/adaboost_fit_main.cpp
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

#ifdef BINDING_NAME
  #undef BINDING_NAME
#endif
#define BINDING_NAME adaboost_fit

#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/core/data/normalize_labels.hpp>
#include "adaboost.hpp"
#include "adaboost_model.hpp"

using namespace mlpack;
using namespace std;
using namespace arma;
using namespace mlpack::adaboost;
using namespace mlpack::tree;
using namespace mlpack::perceptron;
using namespace mlpack::util;

// Program Name.
BINDING_USER_NAME("AdaBoost Training");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of the AdaBoost.MH (Adaptive Boosting) algorithm for "
    "classification.  This can be used to train an AdaBoost model on labeled "
    "data");

// Long description.
BINDING_LONG_DESC(
    "This program implements the AdaBoost (or Adaptive "
    "Boosting) algorithm. The variant of AdaBoost implemented here is "
    "AdaBoost.MH. It uses a weak learner, either decision stumps or "
    "perceptrons, and over many iterations, creates a strong learner that is a "
    "weighted ensemble of weak learners. It runs these iterations until a "
    "tolerance value is crossed for change in the value of the weighted "
    "training error.");

// Example.
BINDING_EXAMPLE(
    "For example, to run AdaBoost on an input dataset: ");

// See also...
BINDING_SEE_ALSO("AdaBoost on Wikipedia", "https://en.wikipedia.org/wiki/"
        "AdaBoost");
BINDING_SEE_ALSO("Improved boosting algorithms using confidence-rated "
        "predictions (pdf)", "http://rob.schapire.net/papers/SchapireSi98.pdf");
BINDING_SEE_ALSO("Perceptron", "#perceptron");
BINDING_SEE_ALSO("Decision Stump", "#decision_stump");
BINDING_SEE_ALSO("mlpack::adaboost::AdaBoost C++ class documentation",
        "@doxygen/classmlpack_1_1adaboost_1_1AdaBoost.html");

// Input for training.
PARAM_MATRIX_IN("training", "Dataset for training AdaBoost.", "t");
PARAM_UROW_IN("labels", "Labels for the training set.", "l");

// Training options.
PARAM_INT_IN("iterations", "The maximum number of boosting iterations to be run"
    " (0 will run until convergence.)", "i", 1000);
PARAM_DOUBLE_IN("tolerance", "The tolerance for change in values of the "
    "weighted error during training.", "e", 1e-10);
PARAM_STRING_IN("weak_learner", "The type of weak learner to use: "
    "'decision_stump', or 'perceptron'.", "w", "decision_stump");

// Loading/saving of a model.
// PARAM_MODEL_IN(AdaBoostModel, "input_model", "Input AdaBoost model.", "m");
PARAM_MODEL_OUT(AdaBoostModel, "output_model", "Output trained AdaBoost model.",
    "M");

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  // Check input parameters and issue warnings/errors as necessary.
  RequireOnlyOnePassed(params, {"training"});
  RequireOnlyOnePassed(params, {"labels"});

  // The weak learner must make sense.
  RequireParamInSet<std::string>(params, "weak_learner",
      { "decision_stump", "perceptron" }, true, "unknown weak learner type");

  // Sanity check on iterations.
  RequireParamValue<int>(params, "iterations", [](int x) { return x > 0; },
      true, "invalid number of iterations specified");

  AdaBoostModel* m;

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
    if(trainingData.n_rows < 2)
    {
      Log::Fatal << "Cannot extract labels from last dimension as total "
          << "dimensions are less than 2." << endl;
    }

    labelsIn = conv_to<Row<size_t>>::from(
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

  params.Get<AdaBoostModel*>("output_model") = m;
}
