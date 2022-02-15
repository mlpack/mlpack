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
	arma::Row<size_t> input("10 14 8 25 16 22 14 35 15 27 18 40 28 40 25 65");

	arma::Row<double> predictions(10);

	arma::Row<double> labels("2.68125 4.82362689 3.508499 5.57006575 \
		4.49598037 6.53427619 5.48237865 7.52714329");

	HoltWintersModel<'M', arma::Row<size_t>> model(input, 2);

	model.Predict(predictions, 0);

	for (size_t i = 2; i < predictions.n_elem; ++i)
		REQUIRE(predictions(i) == Approx(labels(i)).margin(0.0005));
}

/**
 *  Checks predictions using the additive  method.
 */
TEST_CASE("AdditiveMethodTest", "[HoltWintersModelTest]")
{
	arma::Row<size_t> input("10 14 8 25 16 22 14 35 15 27 18 40 28 40 25 65");

	arma::Row<double> predictions(10);

	arma::Row<double> labels("2.625 4.53125 3.5390625 5.51757812 \
		4.51416016 6.50793457 5.50552368 7.50336456");

	HoltWintersModel<'A', arma::Row<size_t>> model(input, 2);

	model.Predict(predictions, 0);

	for (size_t i = 2; i < predictions.n_elem; ++i)
		REQUIRE(predictions(i) == Approx(labels(i)).margin(0.0005));
}
/**
 * Checks predictions using the multiplicative method with a trained model.
 */
TEST_CASE("TrainedMultiplicativeModelPredictionTest", "[HoltWintersModelTest]")
{
	arma::Row<size_t> input("1 3 1 3 1 3 1 3 1 3");

	arma::Row<double> predictions(10);

	arma::Row<double> labels("1 3 1 3 1 3 1 3");

	HoltWintersModel<'M', arma::Row<size_t>> model(input, 2);

	model.Train();

	model.Predict(predictions, 0);

	for (size_t i = 2; i < predictions.n_elem; ++i)
		REQUIRE(predictions(i) == Approx(labels(i)).margin(0.0005));
}

/**
 * Checks predictions using the additive method with a trained model.
 */
TEST_CASE("TrainedAdditiveModelPredictionTest", "[HoltWintersModelTest]")
{
	arma::Row<size_t> input("1 3 1 3 1 3 1 3 1 3");

	arma::Row<double> predictions(10);

	arma::Row<double> labels("1 3 1 3 1 3 1 3");

	HoltWintersModel<'A', arma::Row<size_t>> model(input, 2);

	model.Train();

	model.Predict(predictions, 0);

	for (size_t i = 2; i < predictions.n_elem; ++i)
		REQUIRE(predictions(i) == Approx(labels(i)).margin(0.0005));
}