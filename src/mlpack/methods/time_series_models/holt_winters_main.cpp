/**
 * @file mothods/time_series_models/holt_winters_main.cpp
 * @author Suvarsha Chennareddy
 *
 * Main executable to run Holt Winters Model.
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
#define BINDING_NAME holt_winters

#include <mlpack/core/util/mlpack_main.hpp>
#include "holt_winters.hpp"

using namespace mlpack;
using namespace mlpack::ts;
using namespace mlpack::util;
using namespace std;

// Program Name.
BINDING_USER_NAME("Holt Winters Model");

// Short description.
BINDING_SHORT_DESC(
	"The Holt Winters method is a popular and effective approach "
	"to forecasting seasonal time series.But different "
	"methods will give different forecasts;"
	"method is multiplicative or additive and"
	"how the smoothing parameters are selected.");

// Long description.
BINDING_LONG_DESC(
	"An implementation of Holt Winters Model for time series analysis. It is "
	"also known as the 'Triple Exponential Smoothing Model'."
	"\n\n"
	"This program takes input data (specified with the " +
	PRINT_PARAM_STRING("input") + " parameter). The last coloumn is taken as the "
	"time series. The model then makes  '" + PRINT_PARAM_STRING("numberOfForecasts") + 
	"' predictions according to the mmethod specified by" + 
	PRINT_PARAM_STRING("method") + "('M' for Multiplicative and "
	"'A' or anything else for Additive) and smoothing paramaters either "
	"trained or specified by " + PRINT_PARAM_STRING("alpha") + ", " +
	PRINT_PARAM_STRING("beta") + ", " + PRINT_PARAM_STRING("gamma") + "."
	"\n\n"
	"To store the predicted labels " + PRINT_PARAM_STRING("output") +
	" parameter can be specified.");

//Example.
BINDING_EXAMPLE(
	"For example, to make predictions using the Holt Winters Model on " +
	PRINT_DATASET("data") + " and " + ", storing "
	"the predicted output to " + PRINT_DATASET("predictions") +
	", the following command can be used:"
	"\n\n" +
	PRINT_CALL("holt_winters_model", "input", "data", "method", "method type",
		"period", "seasonal period", "alpha", "level smoothing parameter",
		"beta", "trend smoothing parameter", "gamma", "seasonal smoothing parameter",
		"numberOfForecasts", "The number of forecasts into the future",
		"output", "predictions"));

// See also...
BINDING_SEE_ALSO("Holt Winters Model on Wikipedia",
	"https://en.wikipedia.org/wiki/Exponential_smoothing");
BINDING_SEE_ALSO("Post about Holt Winters Model methods and initialization",
	"https://robjhyndman.com/hyndsight/hw-initialization/");

// Parameters for program.
PARAM_MATRIX_IN_REQ("input", "Input dataset.", "i");
PARAM_STRING_IN("method",
	"The method going to be used to predict (either 'A' or 'M')", "m", "A");
PARAM_INT_IN_REQ("period", "The seasonal period of the input time series", "p");
PARAM_DOUBLE_IN("alpha", "The level smoothing parameter.", "a",
	0.5);
PARAM_DOUBLE_IN("beta", "The trend smoothing parameter.", "b",
	0.5);
PARAM_DOUBLE_IN("gamma", "The seasonal smoothing parameter.", "g",
	0.5);

PARAM_ROW_OUT("output", "Vector to save the predictions to.", "o");

PARAM_INT_IN_REQ("numberOfForecasts", "The number of forecasts into the future", "h");


void BINDING_FUNCTION(util::Params& params, util::Timers& /* timers */)
{
	// Load input dataset.
	arma::mat dataset = std::move(params.Get<arma::mat>("input"));
	arma::Row<double> timeSeries = dataset.col(dataset.n_cols - 1).t();
	double alpha = params.Get<double>("alpha");
	double beta = params.Get<double>("beta");
	double gamma = params.Get<double>("gamma");
	size_t period = (size_t) params.Get<int>("period");
	size_t H = (size_t) params.Get<int>("numberOfForecasts");
	char method = params.Get<std::string>("method")[0];


	// Issue a warning if the user did not specify an output file.
	RequireAtLeastOnePassed(params, { "output" }, false,
		"no output will be saved");

	arma::Row<double> predictions(dataset.n_rows + H);

	// Making the predictions.
	if (method == 'M')
	{
		HoltWintersModel<'M'>model(timeSeries, period, alpha, beta, gamma);
		model.Predict(predictions, H);
	}
	else
	{
		HoltWintersModel<'A'>model(timeSeries, period, alpha, beta, gamma);
		model.Predict(predictions, H);
	}
	// Save predictions, if desired.
	if (params.Has("output"))
	{
		params.Get<arma::rowvec>("output") = std::move(predictions);
	}
}
