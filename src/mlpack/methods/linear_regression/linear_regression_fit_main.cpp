/**
 * @file methods/linear_regression/linear_regression_main.cpp
 * @author James Cline
 *
 * Main function for least-squares linear regression.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/io.hpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "linear_regression.hpp"

using namespace mlpack;
using namespace mlpack::regression;
using namespace mlpack::util;
using namespace arma;
using namespace std;

// Program Name.
BINDING_NAME("Linear Regression Fit");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of simple linear regression and ridge regression using.");

// Long description.
BINDING_LONG_DESC(
    "An implementation of simple linear regression and simple ridge regression.");

// Example.
BINDING_EXAMPLE(
    "For example, to run a linear regression.");

// See also...
BINDING_SEE_ALSO("Linear/ridge regression tutorial",
       "@doxygen/lrtutorial.html");

PARAM_MATRIX_IN_METHSPEC("training", "Matrix containing training set X (regressors).",
    "t");
PARAM_ROW_IN_METHSPEC("training_responses", "Optional vector containing y "
    "(responses). If not given, the responses are assumed to be the last row "
    "of the input file.", "r");

PARAM_MODEL_OUT(LinearRegression, "output_model", "Output LinearRegression "
    "model.", "M");

PARAM_DOUBLE_IN("lambda", "Tikhonov regularization for ridge regression.  If 0,"
    " the method reduces to linear regression.", "l", 0.0);

#ifndef MAKING_WRAPPER
static void mlpackMain()
{
  const double lambda = IO::GetParam<double>("lambda");

  mat regressors;
  rowvec responses;

  LinearRegression* lr;

  Timer::Start("load_regressors");
  regressors = std::move(IO::GetParam<mat>("training"));
  Timer::Stop("load_regressors");

  // Are the responses in a separate file?
  if (!IO::HasParam("training_responses"))
  {
    // The initial predictors for y, Nx1.
    if (regressors.n_rows < 2)
    {
      Log::Fatal << "Can't get responses from training data "
          "since it has less than 2 rows." << endl;
    }
    responses = regressors.row(regressors.n_rows - 1);
    regressors.shed_row(regressors.n_rows - 1);
  }
  else
  {
    // The initial predictors for y, Nx1.
    Timer::Start("load_responses");
    responses = IO::GetParam<rowvec>("training_responses");
    Timer::Stop("load_responses");

    if (responses.n_cols != regressors.n_cols)
    {
      Log::Fatal << "The responses must have the same number of columns "
          "as the training set." << endl;
    }
  }

  Timer::Start("regression");
  lr = new LinearRegression(regressors, responses, lambda);
  Timer::Stop("regression");

  // Save the model if needed.
  IO::GetParam<LinearRegression*>("output_model") = lr;
}
#endif
