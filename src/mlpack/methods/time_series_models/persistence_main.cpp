/**
 * @file mothods/time_series_models/persistence_main.cpp
 * @author Rishabh Garg
 *
 * Main executable to run Persistence Model.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/io.hpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "persistence.hpp"

using namespace mlpack;
using namespace mlpack::ts;
using namespace mlpack::util;
using namespace std;

// Program Name.
BINDING_NAME("Persistence Model");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of Persistence Model for time series analysis. "
    "It is considered as the baseline model for all time series analysis. "
    "It is also known as naive model. Given a dataset, it returns the value "
    "of the previous timestamp for the current timestamp.");

// Long description.
BINDING_LONG_DESC(
    "An implementation of Persistence Model for time series analysis. It is "
    "also known as the 'Naive Model'. This method is often used as a baseline "
    "for all the time series analysis."
    "\n\n"
    "This program takes an input data (specified with the " +
    PRINT_PARAM_STRING("input") + " parameter) and makes predictions "
    "according to the rule `y_hat(t) = y(t-1)`."
    "\n\n"
    "The labels should be either supplied with the " +
    PRINT_PARAM_STRING("labels") + " parameter else the program will assume "
    "the last column of the dataset specified with "
    PRINT_PARAM_STRING("input") + " as the target labels."
    "\n\n"
    "To store the predicted labels " + PRINT_PARAM_STRING("output") +
    " parameter can be specified.")

//Example.
BINDING_EXAMPLE(
    "For example, to make predictions using the Persistence Model on " +
    PRINT_DATASET("data") + " and " + PRINT_DATASET("labels") + ", storing "
    "the predicted output to " + PRINT_DATASET("predictions") +
    ", the following command can be used:"
    "\n\n" +
    PRINT_CALL("persistence", "input", "data", "labels", "labels",
        "output", "predictions"));

// See also...
BINDING_SEE_ALSO("Persistence Model on Wikipedia",
        "https://en.wikipedia.org/wiki/Forecasting#Na%C3%AFve_approach");

// Parameters for program.
PARAM_MATRIX_IN_REQ("input", "Input dataset.", "i");
PARAM_MATRIX_IN("labels", "Labels to be predicted.", "l");
PARAM_MATRIX_OUT("output", "Matrix to save the predictions to.", "o");

