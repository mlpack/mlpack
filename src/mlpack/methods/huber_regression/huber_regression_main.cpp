/**
 * @file methods/huber_regression/huber_regression_main.cpp
 * @author Anna Sai Nikhil
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

#undef BINDING_NAME
#define BINDING_NAME huber_regression

#include <mlpack/core/util/mlpack_main.hpp>

#include "huber_regression.hpp"

using namespace mlpack;
using namespace mlpack::util;
using namespace arma;
using namespace std;

// Program Name.
BINDING_USER_NAME("Simple Linear Regression");

// Short description.
BINDING_SHORT_DESC(
  "Train a linear regression model.");

// Long description.
BINDING_LONG_DESC(
  "An implementation of simple linear regression and simple ridge regression "
  "using ordinary least squares. This solves the problem"
  "\n\n"
  "  y = X * b + e");



void BINDING_FUNCTION(util::Params& params, util::Timers& timer)
{

  mat regressors;
  rowvec responses;

  timer.Start("load_regressors");
  regressors = std::move(params.Get<mat>("training"));
  timer.Stop("load_regressors");

  // Are the responses in a separate file?
  if (!params.Has("training_responses"))
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
    timer.Start("load_responses");
    responses = params.Get<rowvec>("training_responses");
    timer.Stop("load_responses");

    if (responses.n_cols != regressors.n_cols)
    {
      Log::Fatal << "The responses must have the same number of columns "
          "as the training set." << endl;
    }
  }

  if(regressors.n_cols != responses.n_cols)
    Log::Fatal << "Regressors and Responses must have the same number of data points!" << endl;

  timer.Start("regression");
  HuberRegressor<>* hr = new HuberRegressor<>(regressors, responses);
  timer.Stop("regression");

  // Save the model if needed.
  params.Get<HuberRegressor<>*>("output_model") = hr;
}
