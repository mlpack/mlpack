/**
 * @file methods/lars/lars_main.cpp
 * @author Nishant Mehta
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
#define BINDING_NAME lars

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
    "An implementation of Least Angle Regression (Stagewise/laSso), also known"
    " as LARS.  This can train a LARS/LASSO/Elastic Net model and use that "
    "model or a pre-trained model to output regression predictions for a test "
    "set.");

// Long description.
BINDING_LONG_DESC(
    "An implementation of LARS: Least Angle Regression (Stagewise/laSso).  "
    "This is a stage-wise homotopy-based algorithm for L1-regularized linear "
    "regression (LASSO) and L1+L2-regularized linear regression (Elastic Net)."
    "\n\n"
    "This program is able to train a LARS/LASSO/Elastic Net model or load a "
    "model from file, output regression predictions for a test set, and save "
    "the trained model to a file.  The LARS algorithm is described in more "
    "detail below:"
    "\n\n"
    "Let X be a matrix where each row is a point and each column is a "
    "dimension, and let y be a vector of targets."
    "\n\n"
    "The Elastic Net problem is to solve"
    "\n\n"
    "  min_beta 0.5 || X * beta - y ||_2^2 + lambda_1 ||beta||_1 +\n"
    "      0.5 lambda_2 ||beta||_2^2"
    "\n\n"
    "If lambda1 > 0 and lambda2 = 0, the problem is the LASSO.\n"
    "If lambda1 > 0 and lambda2 > 0, the problem is the Elastic Net.\n"
    "If lambda1 = 0 and lambda2 > 0, the problem is ridge regression.\n"
    "If lambda1 = 0 and lambda2 = 0, the problem is unregularized linear "
    "regression."
    "\n\n"
    "For efficiency reasons, it is not recommended to use this algorithm with"
    " " + PRINT_PARAM_STRING("lambda1") + " = 0.  In that case, use the "
    "'linear_regression' program, which implements both unregularized linear "
    "regression and ridge regression."
    "\n\n"
    "To train a LARS/LASSO/Elastic Net model, the " +
    PRINT_PARAM_STRING("input") + " and " + PRINT_PARAM_STRING("responses") +
    " parameters must be given.  The " + PRINT_PARAM_STRING("lambda1") +
    ", " + PRINT_PARAM_STRING("lambda2") + ", and " +
    PRINT_PARAM_STRING("use_cholesky") + " parameters control the training "
    "options.  A trained model can be saved with the " +
    PRINT_PARAM_STRING("output_model") + ".  If no training is desired at all,"
    " a model can be passed via the " + PRINT_PARAM_STRING("input_model") +
    " parameter."
    "\n\n"
    "The program can also provide predictions for test data using either the "
    "trained model or the given input model.  Test points can be specified with"
    " the " + PRINT_PARAM_STRING("test") + " parameter.  Predicted responses "
    "to the test points can be saved with the " +
    PRINT_PARAM_STRING("output_predictions") + " output parameter.");

// Example.
BINDING_EXAMPLE(
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

// See also...
BINDING_SEE_ALSO("@linear_regression", "#linear_regression");
BINDING_SEE_ALSO("Least angle regression (pdf)",
    "https://mlpack.org/papers/lars.pdf");
BINDING_SEE_ALSO("LARS C++ class documentation", "@doc/user/methods/lars.md");

PARAM_TMATRIX_IN("input", "Matrix of covariates (X).", "i");
PARAM_MATRIX_IN("responses", "Matrix of responses/observations (y).", "r");

PARAM_MODEL_IN(LARS<>, "input_model", "Trained LARS model to use.", "m");
PARAM_MODEL_OUT(LARS<>, "output_model", "Output LARS model.", "M");

PARAM_TMATRIX_IN("test", "Matrix containing points to regress on (test "
    "points).", "t");

PARAM_TMATRIX_OUT("output_predictions", "If --test_file is specified, this "
    "file is where the predicted responses will be saved.", "o");

PARAM_DOUBLE_IN("lambda1", "Regularization parameter for l1-norm penalty.", "l",
    0);
PARAM_DOUBLE_IN("lambda2", "Regularization parameter for l2-norm penalty.", "L",
    0);
PARAM_FLAG("use_cholesky", "Use Cholesky decomposition during computation "
    "rather than explicitly computing the full Gram matrix.", "c");
PARAM_FLAG("no_intercept", "Do not fit an intercept in the model.", "n");
PARAM_FLAG("no_normalize", "Do not normalize data to unit variance before "
    "modeling.", "N");

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  double lambda1 = params.Get<double>("lambda1");
  double lambda2 = params.Get<double>("lambda2");
  bool useCholesky = params.Has("use_cholesky");
  bool noIntercept = params.Has("no_intercept");
  bool noNormalize = params.Has("no_normalize");

  // Check parameters -- make sure everything given makes sense.
  RequireOnlyOnePassed(params, { "input", "input_model" }, true);
  if (params.Has("input"))
  {
    RequireOnlyOnePassed(params, { "responses" }, true, "if input data is "
        "specified, responses must also be specified");
  }
  ReportIgnoredParam(params, {{ "input", false }}, "responses");
  ReportIgnoredParam(params, {{ "input", false }}, "no_intercept");
  ReportIgnoredParam(params, {{ "input", false }}, "no_normalize");

  RequireAtLeastOnePassed(params, { "output_predictions", "output_model" },
      false, "no results will be saved");
  ReportIgnoredParam(params, {{ "test", true }}, "output_predictions");

  LARS<>* lars;
  if (params.Has("input"))
  {
    // Initialize the object.
    lars = new LARS<>(useCholesky, lambda1, lambda2);
    lars->FitIntercept(!noIntercept);
    lars->NormalizeData(!noNormalize);

    // Load covariates.  We can avoid LARS transposing our data by choosing to
    // not transpose this data (that's why we used PARAM_TMATRIX_IN).
    mat matX = std::move(params.Get<arma::mat>("input"));

    // Load responses.  The responses should be a one-dimensional vector, and it
    // seems more likely that these will be stored with one response per line
    // (one per row).  So we should not transpose upon loading.
    mat matY = std::move(params.Get<arma::mat>("responses"));

    // Make sure y is oriented the right way.
    if (matY.n_cols == 1)
      matY = trans(matY);
    if (matY.n_rows > 1)
      Log::Fatal << "Only one column or row allowed in responses file!" << endl;

    if (matY.n_elem != matX.n_rows)
      Log::Fatal << "Number of responses must be equal to number of rows of X!"
          << endl;

    arma::rowvec y = std::move(matY);
    timers.Start("lars_regression");
    lars->Train(matX, y, false /* do not transpose */);
    timers.Stop("lars_regression");
  }
  else // We must have --input_model_file.
  {
    lars = params.Get<LARS<>*>("input_model");
  }

  if (params.Has("test"))
  {
    Log::Info << "Regressing on test points." << endl;

    // Load test points.
    mat testPoints = std::move(params.Get<arma::mat>("test"));

    // Make sure the dimensionality is right.  We haven't transposed, so, we
    // check n_cols not n_rows.
    if (testPoints.n_cols != lars->BetaPath().back().n_elem)
      Log::Fatal << "Dimensionality of test set (" << testPoints.n_cols << ") "
          << "is not equal to the dimensionality of the model ("
          << lars->BetaPath().back().n_elem << ")!" << endl;

    arma::rowvec predictions;
    lars->Predict(testPoints.t(), predictions, false);

    // Save test predictions (one per line).
    params.Get<arma::mat>("output_predictions") = predictions.t();
  }

  params.Get<LARS<>*>("output_model") = lars;
}
