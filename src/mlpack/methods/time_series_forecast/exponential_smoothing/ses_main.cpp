/**
 * @file methods/linear_regression/linear_regression_main.cpp
 * @author Aditi Pandey
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

#ifdef BINDING_NAME
  #undef BINDING_NAME
#endif
#define BINDING_NAME single_exponential_smoothing

#include <mlpack/core/util/mlpack_main.hpp>

#include "ses.hpp"

using namespace mlpack;
using namespace mlpack::single_exponential_smoothing;
using namespace mlpack::util;
using namespace arma;
using namespace std;

// Program Name.
BINDING_USER_NAME("Single Exponential Smoothing Forecast");

// Short description.
BINDING_SHORT_DESC(
    "Single Exponential Smoothing, SES for short, also called "
    "Simple Exponential Smoothing, is a time series forecasting "
    "method for univariate data without a trend or seasonality. ");

// Long description.
BINDING_LONG_DESC(
    "A simple exponential smoothing is one of the simplest ways to "
    "forecast a time series. The basic idea of this model is to assume "
    "that the future will be more or less the same as the (recent) past. "
    "Thus, the only pattern that this model will learn from demand history "
    "is its level. That means that this previous forecast includes "
    "everything the model learned so far based on demand history. "
    "The smoothing parameter (or learning rate) alpha will determine how "
    "much importance is given to the most recent demand observation. "
    "The above can be represented mathematically as,"
    "\n\n"
    " st = αxt+(1 – α)st-1 = st-1 + α(xt – st-1)"
    "\n\n");

// Example.


// See also...
BINDING_SEE_ALSO("https://en.wikipedia.org/wiki/Exponential_smoothing");


// Parameters for program.
PARAM_MATRIX_IN_REQ("input", "Input dataset.", "i");
PARAM_ROW_IN("labels", "Labels to be predicted.", "l");
PARAM_ROW_OUT("output", "Vector to save the predictions to.", "o");


static void mlpackMain()
{
  // to be added
}
