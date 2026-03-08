/**
 * @file methods/lars/lars_train_main.cpp
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
#define BINDING_NAME lars_train

#include <mlpack/core/util/mlpack_main.hpp>

#include "lars.hpp"

using namespace arma;
using namespace std;
using namespace mlpack;
using namespace mlpack::util;

// Program Name.
BINDING_USER_NAME("LARS Training");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of Least Angle Regression (stagewise/lasso), also known"
    " as LARS.  This can train a LARS/LASSO/Elastic Net model, and save the"
    " pre-trained model for later use to output regression predictions from a"
    " test set.");

// Long description.
BINDING_LONG_DESC(
    "An implementation of LARS: Least Angle Regression (stagewise/lasso).  "
    "This is a stage-wise homotopy-based algorithm for L1-regularized linear "
    "regression (LASSO) and L1+L2-regularized linear regression (Elastic Net)."
    "\n\n"
    "This program is able to train a LARS/LASSO/Elastic Net model or load a "
    "model from a file, output regression predictions for a test set, and save"
    " the trained model to a file.  The LARS algorithm is described in more "
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
    "\n\n");

// Example.
BINDING_EXAMPLE(
    IMPORT_EXT_LIB() + "\n" +
    IMPORT_SPLIT() + "\n" +
    IMPORT_THIS("lars") + "\n" +
    GET_DATASET("X",
      "http://datasets.mlpack.org/admission_predict.csv") + "\n" +
    GET_DATASET("y",
      "http://datasets.mlpack.org/admission_predict.responses.csv") + "\n" +
    SPLIT_TRAIN_TEST("X", "y", "X_train", "y_train", "X_test",
      "y_test", "0.2") + "\n" +
    CREATE_OBJECT("model", "lars") + "\n" +
    CALL_METHOD("model", "train", "input", "X_train", "responses", "y_train",
      "lambda1", 1e-5, "lambda2", 1e-6, "output_model", "lars_model"));

// See also...
BINDING_SEE_ALSO("@lars_predict", "#lars_predict");
BINDING_SEE_ALSO("@linear_regression", "#linear_regression");
BINDING_SEE_ALSO("Least angle regression (pdf)",
    "https://mlpack.org/papers/lars.pdf");
BINDING_SEE_ALSO("LARS C++ class documentation", "@doc/user/methods/lars.md");

PARAM_TMATRIX_IN_REQ("input", "Matrix of covariates (X).", "i");
PARAM_ROW_IN_REQ("responses", "Row vector of responses/observations (y).", "r");

PARAM_MODEL_OUT(LARS<>, "output_model", "Output LARS model.", "M");

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

  RequireAtLeastOnePassed(params, { "output_model" },
      false, "no results will be saved");

  // Load covariates.  We can avoid LARS transposing our data by choosing to
  // not transpose this data (that's why we used PARAM_TMATRIX_IN).
  mat matX = std::move(params.Get<arma::mat>("input"));

  // Load responses.  The responses should be a one-dimensional vector.
  arma::rowvec y = std::move(params.Get<arma::rowvec>("responses"));

  if (y.n_elem != matX.n_rows)
    Log::Fatal << "Number of responses must be equal to number of rows of X!"
        << endl;

  // Initialize the object.
  LARS<>* lars = new LARS<>(useCholesky, lambda1, lambda2);
  lars->FitIntercept(!noIntercept);
  lars->NormalizeData(!noNormalize);

  timers.Start("lars_regression");
  lars->Train(matX, y, false /* do not transpose */);
  timers.Stop("lars_regression");

  params.Get<LARS<>*>("output_model") = lars;
}
