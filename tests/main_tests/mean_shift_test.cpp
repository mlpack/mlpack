/**
 * @file tests/main_tests/mean_shift_test.cpp
 * @author Tan Jun An
 *
 * Test RUN_BINDING() of mean_shift_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/mean_shift/mean_shift_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "main_test_fixture.hpp"
#include "../test_catch_tools.hpp"
#include "../catch.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(MeanShiftTestFixture);

/**
 * Ensure that the output has 1 extra row for the labels and
 * check the number of points for output remain the same.
 */
TEST_CASE_METHOD(
    MeanShiftTestFixture, "MeanShiftOutputDimensionTest",
    "[MeanShiftMainTest][BindingTests]")
{
  arma::mat x;
  x.randu(3, 100); // 100 points in 3 dimension

  // Input random data points.
  SetInputParam("input", std::move(x));

  RUN_BINDING();

  // Now check that the output has 1 extra row for labels.
  REQUIRE(params.Get<arma::mat>("output").n_rows == 3 + 1);
  // Check number of output points are the same.
  REQUIRE(params.Get<arma::mat>("output").n_cols == 100);
}

/**
 * Ensure that if we ask for labels_only, output has 1 row and
 * same number of columns for each point's label.
 */
TEST_CASE_METHOD(
    MeanShiftTestFixture, "MeanShiftLabelOnlyOutputDimensionTest",
    "[MeanShiftMainTest][BindingTests]")
{
  arma::mat x;
  x.randu(3, 100); // 100 points in 3 dimension

  // Input random data points.
  SetInputParam("input", std::move(x));
  SetInputParam("labels_only", true);

  RUN_BINDING();

  // Check that there is only 1 row containing all the labels.
  REQUIRE(params.Get<arma::mat>("output").n_rows == 1);
  // Check number of output points are the same.
  REQUIRE(params.Get<arma::mat>("output").n_cols == 100);
}

/**
 * Ensure that the updated input file has 1 extra row for the labels 
 * and check the number of points remain the same if the --in_place
 * flag is set.
 */
TEST_CASE_METHOD(
    MeanShiftTestFixture, "MeanShiftInPlaceTest",
    "[MeanShiftMainTest][BindingTests]")
{
  arma::mat x;
  if (!data::Load("iris_test.csv", x))
    FAIL("Cannot load test dataset iris_test.csv!");

  // Get initial number of rows and columns in file.
  int numRows = x.n_rows;
  int numCols = x.n_cols;

  // Input random data points.
  SetInputParam("input", std::move(x));
  SetInputParam("in_place", true);

  RUN_BINDING();

  // Now check that the output has 1 extra row for labels.
  REQUIRE(params.Get<arma::mat>("output").n_rows ==
      (arma::uword) (numRows + 1));
  // Check number of output points are the same.
  REQUIRE(params.Get<arma::mat>("output").n_cols == (arma::uword) numCols);
}

/**
 * Ensure that force_convergence is used by testing that the
 * force_convergence flag makes a difference in the program.
 */
TEST_CASE_METHOD(
    MeanShiftTestFixture, "MeanShiftForceConvergenceTest",
    "[MeanShiftMainTest][BindingTests]")
{
  arma::mat x;
  if (!data::Load("iris_test.csv", x))
    FAIL("Cannot load test dataset iris_test.csv!");

  // Input random data points.
  SetInputParam("input", x);
  // Set a very small max_iterations.
  SetInputParam("max_iterations", (int) 1);

  RUN_BINDING();

  const int numCentroids1 = params.Get<arma::mat>("centroid").n_cols;

  CleanMemory();
  ResetSettings();

  // Input same random data points.
  SetInputParam("input", std::move(x));
  // Set the same small max_iterations.
  SetInputParam("max_iterations", (int) 1);
  // Set the force_convergence flag on.
  SetInputParam("force_convergence", true);

  RUN_BINDING();

  const int numCentroids2 = params.Get<arma::mat>("centroid").n_cols;
  // Resulting number of centroids should be different.
  REQUIRE(numCentroids1 != numCentroids2);
}

/**
 * Ensure that radius is used by testing that the radius
 * makes a difference in the program.
 */
TEST_CASE_METHOD(
    MeanShiftTestFixture, "MeanShiftRadiusTest",
    "[MeanShiftMainTest][BindingTests]")
{
  arma::mat x;
  if (!data::Load("iris_test.csv", x))
    FAIL("Cannot load test dataset iris_test.csv!");

  // Input random data points.
  SetInputParam("input", x);
  // Set a small radius.
  SetInputParam("radius", (double) 0.1);

  RUN_BINDING();

  const int numCentroids1 = params.Get<arma::mat>("centroid").n_cols;

  CleanMemory();
  ResetSettings();

  // Input same random data points.
  SetInputParam("input", std::move(x));
  // Set a larger radius.
  SetInputParam("radius", (double) 1.0);

  RUN_BINDING();

  const int numCentroids2 = params.Get<arma::mat>("centroid").n_cols;
  // Resulting number of centroids should be different.
  REQUIRE(numCentroids1 != numCentroids2);
}

/**
 * Ensure that max_iterations is used by testing that the
 * max_iteration makes a difference in the program.
 */
TEST_CASE_METHOD(
    MeanShiftTestFixture, "MeanShiftMaxIterationsTest",
    "[MeanShiftMainTest][BindingTests]")
{
  arma::mat x;
  if (!data::Load("iris_test.csv", x))
    FAIL("Cannot load test dataset iris_test.csv!");

  // Input random data points.
  SetInputParam("input", x);
  // Set a small max_iterations.
  SetInputParam("max_iterations", (int) 4);

  RUN_BINDING();

  const int numCentroids1 = params.Get<arma::mat>("centroid").n_cols;

  CleanMemory();
  ResetSettings();

  // Input same random data points.
  SetInputParam("input", std::move(x));
  // Set a larger max_iterations.
  SetInputParam("max_iterations", (int) 20);

  RUN_BINDING();

  const int numCentroids2 = params.Get<arma::mat>("centroid").n_cols;
  // Resulting number of centroids should be different.
  REQUIRE(numCentroids1 != numCentroids2);
}

/**
 * Ensure that we can't specify an invalid max number of iterations.
 */
TEST_CASE_METHOD(
    MeanShiftTestFixture, "MeanShiftInvalidMaxIterationsTest",
    "[MeanShiftMainTest][BindingTests]")
{
  arma::mat x;
  x.randu(3, 100); // 100 points in 3 dimension

  // Input random data points.
  SetInputParam("input", std::move(x));
  // Input invalid max number of iterations.
  SetInputParam("max_iterations", (int) -1);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}
