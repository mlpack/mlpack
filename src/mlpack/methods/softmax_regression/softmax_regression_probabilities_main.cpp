/**
 * @file methods/softmax_regression/softmax_regression_probabilities_main.cpp
 *
 * Implementaton of softmax regression classification on new data.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME softmax_regression_probabilities

#include <mlpack/core/util/mlpack_main.hpp>

#include <mlpack/methods/softmax_regression/softmax_regression.hpp>
#include <mlpack/methods/softmax_regression/softmax_regression_utils.hpp>

using namespace std;
using namespace mlpack;
using namespace mlpack::util;

// Program Name.
BINDING_USER_NAME("Softmax Regression Classification");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of softmax regression for classification, which is a "
    "multiclass generalization of logistic regression.  Given a "
    "pre-trained softmax regression model, new points are classified.");

// Long description.
BINDING_LONG_DESC("");

// Example.
BINDING_EXAMPLE(
    CALL_METHOD("model", "probabilities", "test", "X_test"));

// See also...
BINDING_SEE_ALSO("@logistic_regression", "#logistic_regression");
BINDING_SEE_ALSO("@random_forest", "#random_forest");
BINDING_SEE_ALSO("Multinomial logistic regression (softmax regression) on "
    "Wikipedia",
    "https://en.wikipedia.org/wiki/Multinomial_logistic_regression");
BINDING_SEE_ALSO("SoftmaxRegression C++ class documentation",
    "@doc/user/methods/softmax_regression.md");

// Required options.
PARAM_MODEL_IN_REQ(SoftmaxRegression<>, "input_model", "Existing model "
    "(parameters).", "m");

PARAM_MATRIX_IN_REQ("test", "Matrix containing test dataset.", "T");
PARAM_MATRIX_OUT("probabilities", "Matrix to save class probabilities for test "
    "dataset into.", "P");
PARAM_UROW_IN("test_labels", "Matrix containing test labels.", "L");

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  SoftmaxRegression<>* sm = params.Get<SoftmaxRegression<>*>("input_model");
  RunPredictionStep(params, timers, sm->NumClasses(), *sm, false, true);
}
