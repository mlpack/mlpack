/**
 * @file tests/main_tests/approx_kfn_test.cpp
 * @author Namrata Mukhija
 *
 * Test RUN_BINDING() of approx_kfn_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/approx_kfn/approx_kfn_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "main_test_fixture.hpp"

#include "../catch.hpp"
#include "../test_catch_tools.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(ApproxKFNTestFixture);

/**
 * Check that we can't specify both a reference set and an input model.
 */
TEST_CASE_METHOD(ApproxKFNTestFixture, "ApproxKFNRefModelTest",
                 "[ApproxKFNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(2, 80); // 80 points in 2 dimensions.

  // Random input, any k <= reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);

  // The memory will be cleaned by CleanMemory().
  ApproxKFNModel* m = new ApproxKFNModel();
  SetInputParam("input_model", m);

  // Input pre-trained model.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Check that we can't specify an invalid k.
 */
TEST_CASE_METHOD(ApproxKFNTestFixture, "ApproxKFNInvalidKTest",
                 "[ApproxKFNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(2, 80); // 80 points in 2 dimensions.

  SetInputParam("reference", std::move(referenceData));
  // Random input, k > reference points.
  SetInputParam("k", (int) 81); // Invalid.

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Make sure that the dimensions of neighbors and distances is correct given a
 * value of k.
 */
TEST_CASE_METHOD(ApproxKFNTestFixture, "ApproxKFNOutputDimensionTest",
                 "[ApproxKFNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(2, 80); // 80 points in 2 dimensions.

  // Random input, any k <= reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);

  RUN_BINDING();

  // Check the neighbors matrix has 10 points for each of the 80 input points.
  REQUIRE(params.Get<arma::Mat<size_t>>("neighbors").n_rows == 10);
  REQUIRE(params.Get<arma::Mat<size_t>>("neighbors").n_cols == 80);

  // Check the distances matrix has 10 points for each of the 80 input points.
  REQUIRE(params.Get<arma::mat>("distances").n_rows == 10);
  REQUIRE(params.Get<arma::mat>("distances").n_cols == 80);
}

/**
 * Check that we can't specify an invalid algorithm.
 */
TEST_CASE_METHOD(ApproxKFNTestFixture, "ApproxKFNInvalidAlgorithmTest",
                 "[ApproxKFNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(2, 80); // 80 points in 2 dimensions.

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("algorithm", (string) "any_algo"); // Invalid.

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Check that we can't specify num_projections as zero.
 */
TEST_CASE_METHOD(ApproxKFNTestFixture, "ApproxKFNZeroNumProjTest",
                 "[ApproxKFNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(2, 80); // 80 points in 2 dimensions.

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);
  SetInputParam("num_projections", (int) 0); // Invalid.

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Check that we can't specify num_projections as negative.
 */
TEST_CASE_METHOD(ApproxKFNTestFixture, "ApproxKFNNegativeNumProjTest",
                 "[ApproxKFNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(2, 80); // 80 points in 2 dimensions.

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);
  SetInputParam("num_projections", (int) -5); // Invalid.

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Check that we can't specify num_tables as zero.
 */
TEST_CASE_METHOD(ApproxKFNTestFixture, "ApproxKFNZeroNumTablesTest",
                 "[ApproxKFNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(2, 80); // 80 points in 2 dimensions.

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);
  SetInputParam("num_tables", (int) 0); // Invalid.

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Check that we can't specify num_tables as negative.
 */
TEST_CASE_METHOD(ApproxKFNTestFixture, "ApproxKFNNegativeNumTablesTest",
                 "[ApproxKFNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(2, 80); // 80 points in 2 dimensions.

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);
  SetInputParam("num_tables", (int) -5); // Invalid.

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Ensuring that a saved model can be loaded and used again correctly.
 */
TEST_CASE_METHOD(ApproxKFNTestFixture, "ApproxKFNModelReuseTest",
                 "[ApproxKFNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(2, 80); // 80 points in 2 dimensions.

  arma::mat queryData;
  queryData.randu(2, 40); // 40 points in 2 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("query", queryData);
  SetInputParam("k", (int) 10);

  RUN_BINDING();

  arma::Mat<size_t> neighbors;
  arma::mat distances;
  neighbors = std::move(params.Get<arma::Mat<size_t>>("neighbors"));
  distances = std::move(params.Get<arma::mat>("distances"));
  ApproxKFNModel* model =
      new ApproxKFNModel(*params.Get<ApproxKFNModel*>("output_model"));

  CleanMemory();
  ResetSettings();

  // Input saved model, pass the same query and keep k unchanged.
  SetInputParam("input_model", model);
  SetInputParam("query", queryData);
  SetInputParam("k", (int) 10);

  RUN_BINDING();

  // Check that initial output matrices and the output matrices using
  // saved model are equal.
  CheckMatrices(neighbors, params.Get<arma::Mat<size_t>>("neighbors"));
  CheckMatrices(distances, params.Get<arma::mat>("distances"));
}

/**
 * Ensuring that num_tables has some effects on output.
 */
TEST_CASE_METHOD(ApproxKFNTestFixture, "ApproxKFNNumTablesChangeTest",
                 "[ApproxKFNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(2, 80); // 80 points in 2 dimensions.

  SetInputParam("reference", std::move(referenceData));
  // Random input, k <= reference points.
  SetInputParam("k", (int) 5);
  // Random input, num_tables > 0.
  SetInputParam("num_tables", (int) 1);
  SetInputParam("num_projections", (int) 10);

  // First solution.
  RUN_BINDING();

  // Get the distances matrix after first training.
  arma::mat firstOutputDistances =
      std::move(params.Get<arma::mat>("distances"));

  // Reset the settings.
  CleanMemory();
  ResetSettings();

  // Second setting.
  referenceData.randu(2, 80); // 80 points in 2 dimensions.
  SetInputParam("reference", std::move(referenceData));
  // Same input as first setting, k <= reference points.
  SetInputParam("k", (int) 5);
  // Random input, num_tables > 0.
  SetInputParam("num_tables", (int) 4);

  SetInputParam("num_projections", (int) 10);
  // Second solution.
  RUN_BINDING();

  // Get the distances matrix after second training.
  arma::mat secondOutputDistances =
      std::move(params.Get<arma::mat>("distances"));

  // Check that the size of distance matrices (FirstOutputDistances and
  // SecondOutputDistances) are not equal which ensures num_tables changes
  // the output model.
  CheckMatricesNotEqual(firstOutputDistances, secondOutputDistances);
}

/**
 * Ensuring that num_projections has some effects on output.
 */
TEST_CASE_METHOD(ApproxKFNTestFixture, "ApproxKFNNumProjectionsChangeTest",
                 "[ApproxKFNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(2, 80); // 80 points in 2 dimensions.

  // First setting.
  SetInputParam("reference", std::move(referenceData));
  // Random input, k <= reference points.
  SetInputParam("k", (int) 5);
  // Random input, num_tables > 0.
  SetInputParam("num_projections", (int) 4);
  SetInputParam("num_tables", (int) 3);
  // First solution.
  RUN_BINDING();

  // Get the distances matrix after first training.
  arma::mat firstOutputDistances =
      std::move(params.Get<arma::mat>("distances"));

  // Reset the settings.
  CleanMemory();
  ResetSettings();

  // Second setting.
  referenceData.randu(2, 80); // 80 points in 2 dimensions.
  SetInputParam("reference", std::move(referenceData));
  // Same input as first setting, k <= reference points.
  SetInputParam("k", (int) 5);
  // Random input, num_tables > 0.
  SetInputParam("num_projections", (int) 6);
  SetInputParam("num_tables", (int) 3);

  // Second solution.
  RUN_BINDING();

  // Get the distances matrix after second training.
  arma::mat secondOutputDistances =
      std::move(params.Get<arma::mat>("distances"));

  // Check that the size of distance matrices (FirstOutputDistances and
  // SecondOutputDistances) are not equal which ensures num_tables changes
  // the output model.
  CheckMatricesNotEqual(firstOutputDistances, secondOutputDistances);
}

/**
 * Make sure that the dimensions of the exact distances matrix are correct.
 */
TEST_CASE_METHOD(ApproxKFNTestFixture, "ApproxKFNExactDistDimensionTest",
                 "[ApproxKFNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(2, 80); // 80 points in 2 dimensions.

  SetInputParam("reference", std::move(referenceData));
  // Random input, any k <= reference points.
  SetInputParam("k", (int) 10);
  SetInputParam("calculate_error", (bool) true);

  // Random matrix specifying exact distances of each point to its k neighbors.
  // Note that the values in the matrix do not matter as we are only concernec
  // with the dimensions of the matrix passed.
  arma::mat exactDistances;
  exactDistances.randu(9, 90); // Wrong size (should be (1, 80)).
  SetInputParam("exact_distances", std::move(exactDistances));

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Make sure that the two strategie (Drusilla Select and QDAFN) output
 * different results.
 */
TEST_CASE_METHOD(ApproxKFNTestFixture, "ApproxKFNDifferentAlgoTest",
                 "[ApproxKFNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(6, 100); // 100 points in 6 dimensions.

  SetInputParam("reference", std::move(referenceData));
  // Random input, any k <= reference points.
  SetInputParam("k", (int) 10);
  SetInputParam("algorithm", (string) "ds");

  // First solution.
  RUN_BINDING();

  // Get the distances and neighbors matrix after first training.
  arma::mat firstOutputDistances =
  std::move(params.Get<arma::mat>("distances"));
  arma::Mat<size_t> firstOutputNeighbors =
    std::move(params.Get<arma::Mat<size_t>>("neighbors"));

  // Reset the settings.
  CleanMemory();
  ResetSettings();

  // Second solution.
  SetInputParam("reference", std::move(referenceData));
  // Random input, any k <= reference points.
  SetInputParam("k", (int) 10);
  SetInputParam("algorithm", (string) "qdafn");

  // Get the distances and neighbors matrix after second training.
  arma::mat secondOutputDistances =
      std::move(params.Get<arma::mat>("distances"));
  arma::Mat<size_t> secondOutputNeighbors =
      std::move(params.Get<arma::Mat<size_t>>("neighbors"));

  // Check that the distance matrices (firstOutputDistances and
  // secondOutputDistances) and neighbor matrices (firstOutputNeighbors and
  // secondOutputNeighbors) are not equal. This ensures that the two strategies
  // result in different outputs.
  CheckMatricesNotEqual(firstOutputDistances, secondOutputDistances);
  CheckMatricesNotEqual(firstOutputNeighbors, secondOutputNeighbors);
}
