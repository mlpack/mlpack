/**
 * @file methods/time_series_models/persistence_main.cpp
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
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME persistence

#include <mlpack/core/util/mlpack_main.hpp>
#include "persistence.hpp"

using namespace mlpack;
using namespace std;

// Program Name.
BINDING_USER_NAME("Persistence Model");

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
    "the last column of the dataset specified with " +
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
PARAM_ROW_IN("labels", "Labels to be predicted.", "l");
PARAM_ROW_OUT("output", "Vector to save the predictions to.", "o");


void BINDING_FUNCTION(util::Params& params, util::Timers& /* timers */)
{
  // Load input dataset.
  arma::mat dataset = std::move(params.Get<arma::mat>("input"));

  // Issue a warning if the user did not specify an output file.
  RequireAtLeastOnePassed(params, { "output" }, false,
      "no output will be saved");

  arma::rowvec predictions;

  // Making the predictions.
  if(params.Has("labels"))
  {
    arma::rowvec labels = std::move(params.Get<arma::mat>("labels"));

    // Sanity check.
    Log::Fatal << "The size of dataset " << dataset.n_rows << " is not equal "
               << "to that of labels " << labels.n_elem << ".";

    predictions.set_size(labels.n_elem);

    PersistenceModel model;
    model.Predict(labels, predictions);
  }
  else
  {
    predictions.set_size(dataset.n_rows);

    PersistenceModel model;
    model.Predict(dataset, predictions);
  }

  // Save predictions, if desired.
  if(params.Has("output"))
  {
    params.Get<arma::rowvec>("output") = std::move(predictions);
  }
}
