/**
 * @file methods/adaboost/adaboost_probabilities_main.cpp
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
#define BINDING_NAME adaboost_probabilities

#include <mlpack/core/util/mlpack_main.hpp>

#include "adaboost.hpp"
#include "adaboost_model.hpp"

using namespace mlpack;
using namespace std;
using namespace arma;
using namespace mlpack::util;

// Program Name.
BINDING_USER_NAME("AdaBoost Probability Prediction");

// Short description.
BINDING_SHORT_DESC("Class probabilities from model.");

// Long description.
BINDING_LONG_DESC("");

// Example.
BINDING_EXAMPLE(
    CALL_METHOD("model", "probabilities", "test", "X_test"));

// Classification options.
PARAM_MATRIX_IN_REQ("test", "Test dataset.", "T");
PARAM_MATRIX_OUT("probabilities", "Predicted class probabilities for each "
    "point in the test set.", "p");

// Loading/saving of a model.
PARAM_MODEL_IN_REQ(AdaBoostModel, "input_model", "Input AdaBoost model.", "m");

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  AdaBoostModel* m = params.Get<AdaBoostModel*>("input_model");

  mat testingData = std::move(params.Get<arma::mat>("test"));

  if (testingData.n_rows != m->Dimensionality())
    Log::Fatal << "Test data dimensionality (" << testingData.n_rows << ") "
        << "must be the same as the model dimensionality ("
        << m->Dimensionality() << ")!" << endl;

  Row<size_t> predictedLabels(testingData.n_cols);
  mat probabilities;

  timers.Start("adaboost_classification");
  m->Classify(testingData, predictedLabels, probabilities);
  timers.Stop("adaboost_classification");

  params.Get<arma::mat>("probabilities") = std::move(probabilities);
}
