/**
 * @file methods/lars/lars_predict_main.cpp
 * @author Nishant Mehta
 * @author Dirk Eddelbuettel
 *
 * Executable for LARS.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME lars_predict

#include <mlpack/core/util/mlpack_main.hpp>

#include "lars.hpp"

using namespace arma;
using namespace std;
using namespace mlpack;
using namespace mlpack::util;

// Program Name.
BINDING_USER_NAME("LARS");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of Least Angle Regression (stagewise/lasso), also known"
    " as LARS.  This program can use a pre-trained LARS/LASSO/Elastic Net "
    "model to output regression predictions from a test set.");

// Long description.
BINDING_LONG_DESC("");

// Example.
BINDING_EXAMPLE(
    CALL_METHOD("model", "classify", "test", "X_test"));
#if 0
    "For example, the following command trains a model on the data " +
    PRINT_DATASET("data") + " and responses " + PRINT_DATASET("responses") +
    " with lambda1 set to 0.4 and lambda2 set to 0 (so, LASSO is being "
    "solved), and then the model is saved to " + PRINT_MODEL("lasso_model") +
    ":"
    "\n\n" +
    PRINT_CALL("lars", "input", "data", "responses", "responses", "lambda1",
        0.4, "lambda2", 0.0, "output_model", "lasso_model") +
    "\n\n"
    "The following command uses the " + PRINT_MODEL("lasso_model") + " to "
    "provide predicted responses for the data " + PRINT_DATASET("test") + " "
    "and save those responses to " + PRINT_DATASET("test_predictions") + ": "
    "\n\n" +
    PRINT_CALL("lars", "input_model", "lasso_model", "test", "test",
        "output_predictions", "test_predictions"));
#endif

// See also...
BINDING_SEE_ALSO("@linear_regression", "#linear_regression");
BINDING_SEE_ALSO("Least angle regression (pdf)",
    "https://mlpack.org/papers/lars.pdf");
BINDING_SEE_ALSO("LARS C++ class documentation", "@doc/user/methods/lars.md");

PARAM_MODEL_IN_REQ(LARS<>, "input_model", "Trained LARS model to use.", "m");

PARAM_TMATRIX_IN_REQ("test", "Matrix containing points to regress on (test "
    "points).", "t");

PARAM_TMATRIX_OUT("output_predictions", "If --test_file is specified, this "
    "file is where the predicted responses will be saved.", "o");

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  //RequireOnlyOnePassed(params, { "responses" }, true, "responses must "
  //    "also be specified");

  // Initialize the object.
  LARS<>* lars = params.Get<LARS<>*>("input_model");

  // Load test points.
  mat testPoints = std::move(params.Get<arma::mat>("test"));

  // Make sure the dimensionality is right.  We haven't transposed, so, we
  // check n_cols not n_rows.
  if (testPoints.n_cols != lars->BetaPath().back().n_elem)
    Log::Fatal << "Dimensionality of test set (" << testPoints.n_cols << ") "
        << "is not equal to the dimensionality of the model ("
        << lars->BetaPath().back().n_elem << ")!" << endl;

  arma::rowvec predictions;
  timers.Start("lars_prediction");
  lars->Predict(testPoints.t(), predictions, false);
  timers.Stop("lars_prediction");

  // Save test predictions (one per line).
  params.Get<arma::mat>("output_predictions") = predictions.t();
}
