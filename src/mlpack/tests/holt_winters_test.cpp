/**
 * @file tests/holt_winters_test.cpp
 * @author Suvarsha Chennareddy
 *
 * Test the Holt Winters Model for time series.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/time_series_models/holt_winters.hpp>

#include "test_catch_tools.hpp"
#include "catch.hpp"

using namespace mlpack;
using namespace mlpack::ts;
/**
 * Checks predictions using the multiplicative method.
 */
TEST_CASE("MultiplicativeMethodTest", "[HoltWintersModelTest]")
{
	arma::Row<double> input("10.0 14.0 8.0 25.0 16.0 22.0 14.0 35.0 \
		15.0 27.0 18.0 40.0 28.0 40.0 25.0 65.0");

	arma::Row<double> predictions(input.n_elem);

	arma::Row<double> labels("19.4863219 25.2905334 15.7103141 \
		37.5441898 14.7947244 28.6527549 20.1099415 42.474688 \
		31.2796483 43.0564249 26.7261166 71.0462546");

	HoltWintersModel<'M', arma::Row<double>> model(input, 4);

	model.Predict(predictions, 0);

	for (size_t i = 0; i < labels.n_elem; ++i)
		REQUIRE(predictions(i+4) == Approx(labels(i)).margin(0.0005));
}

/**
 *  Checks predictions using the additive  method.
 */
TEST_CASE("AdditiveMethodTest", "[HoltWintersModelTest]")
{
	arma::Row<double> input("10.0 14.0 8.0 25.0 16.0 22.0 14.0 35.0 \
		15.0 27.0 18.0 40.0 28.0 40.0 25.0 65.0");

	arma::Row<double> predictions(input.n_elem);

	arma::Row<double> labels("18.90625 25.1953125 16.0410156 36.9536133 \
		14.6558838 27.9641418 19.3489151 41.9206257 31.0658631 43.440116 \
		26.7006356 70.5288205");

	HoltWintersModel<'A', arma::Row<double>> model(input, 4);

	model.Predict(predictions, 0);


	for (size_t i = 0; i < labels.n_elem; ++i)
		REQUIRE(predictions(i + 4) == Approx(labels(i)).margin(0.0005));
}
/**
 * Checks predictions using the multiplicative method with a trained model.
 */
TEST_CASE("TrainedMultiplicativeModelPredictionTest", "[HoltWintersModelTest]")
{
	arma::Row<double> input("1.0 3.0 1.0 3.0 1.0 3.0 1.0 3.0 1.0 3.0");

	arma::Row<double> predictions(input.n_elem);

	arma::Row<double> labels("1.0 3.0 1.0 3.0 1.0 3.0 1.0 3.0");

	HoltWintersModel<'M', arma::Row<double>> model(input, 2);

	model.Train();

	model.Predict(predictions, 0);


	for (size_t i = 0; i < labels.n_elem; ++i)
		REQUIRE(predictions(i + 2) == Approx(labels(i)).margin(0.0005));
}

/**
 * Checks predictions using the additive method with a trained model.
 */
TEST_CASE("TrainedAdditiveModelPredictionTest", "[HoltWintersModelTest]")
{
	arma::Row<double> input("1.0 3.0 1.0 3.0 1.0 3.0 1.0 3.0 1.0 3.0");

	arma::Row<double> predictions(input.n_elem);

	arma::Row<double> labels("1.0 3.0 1.0 3.0 1.0 3.0 1.0 3.0");

	HoltWintersModel<'A', arma::Row<double>> model(input, 2);

	model.Train();

	model.Predict(predictions, 0);

	for (size_t i = 0; i < labels.n_elem; ++i)
		REQUIRE(predictions(i + 2) == Approx(labels(i)).margin(0.0005));
}