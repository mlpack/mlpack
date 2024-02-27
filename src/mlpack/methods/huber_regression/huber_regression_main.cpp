/**
 * @file methods/linear_regression/huber_regression_main.cpp
 * @author Anna Sai Nikhil
 *
 * Main function for huber regression.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME linear_regression

#include <mlpack/core/util/mlpack_main.hpp>

#include "huber_regression.hpp"

using namespace mlpack;
using namespace mlpack::util;
using namespace arma;
using namespace std;

// Program Name.
BINDING_USER_NAME("Huber Regression");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of simple linear regression and ridge regression using "
    "ordinary least squares.  Given a dataset and responses, a model can be "
    "trained and saved for later use, or a pre-trained model can be used to "
    "output regression predictions for a test set.");
