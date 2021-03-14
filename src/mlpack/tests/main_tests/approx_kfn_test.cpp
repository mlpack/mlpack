/**
 * @file tests/main_tests/approx_kfn_test.cpp
 * @author Namrata Mukhija
 *
 * Test mlpackMain() of approx_kfn_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <string>

#define BINDING_TYPE BINDING_TYPE_TEST
static const std::string testName = "ApproxK-FurthestNeighbors";

#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "test_helper.hpp"
#include <mlpack/methods/approx_kfn/approx_kfn_main.cpp>

#include "../catch.hpp"
#include "../test_catch_tools.hpp"

using namespace mlpack;

struct ApproxKFNTestFixture
{
 public:
  ApproxKFNTestFixture()
  {
    // Cache in the options for this program.
    IO::RestoreSettings(testName);
  }

  ~ApproxKFNTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    IO::ClearSettings();
  }
};

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
  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
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

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
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

  mlpackMain();

  // Check the neighbors matrix has 10 points for each of the 80 input points.
  REQUIRE(IO::GetParam<arma::Mat<size_t>>("neighbors").n_rows == 10);
  REQUIRE(IO::GetParam<arma::Mat<size_t>>("neighbors").n_cols == 80);

  // Check the distances matrix has 10 points for each of the 80 input points.
  REQUIRE(IO::GetParam<arma::mat>("distances").n_rows == 10);
  REQUIRE(IO::GetParam<arma::mat>("distances").n_cols == 80);
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

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
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

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
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

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
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

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
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

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
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

  mlpackMain();

  arma::Mat<size_t> neighbors;
  arma::mat distances;
  neighbors = std::move(IO::GetParam<arma::Mat<size_t>>("neighbors"));
  distances = std::move(IO::GetParam<arma::mat>("distances"));
  ApproxKFNModel* model =
      new ApproxKFNModel(*IO::GetParam<ApproxKFNModel*>("output_model"));

  bindings::tests::CleanMemory();

  // Reset passed parameters.
  IO::GetSingleton().Parameters()["reference"].wasPassed = false;
  IO::GetSingleton().Parameters()["query"].wasPassed = false;

  // Input saved model, pass the same query and keep k unchanged.
  SetInputParam("input_model", model);
  SetInputParam("query", queryData);

  mlpackMain();

  // Check that initial output matrices and the output matrices using
  // saved model are equal.
  CheckMatrices(neighbors, IO::GetParam<arma::Mat<size_t>>("neighbors"));
  CheckMatrices(distances, IO::GetParam<arma::mat>("distances"));
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
  mlpackMain();

  // Get the distances matrix after first training.
  arma::mat firstOutputDistances =
      std::move(IO::GetParam<arma::mat>("distances"));

  // Reset the settings.
  bindings::tests::CleanMemory();
  IO::ClearSettings();
  IO::RestoreSettings(testName);

  // Second setting.
  referenceData.randu(2, 80); // 80 points in 2 dimensions.
  SetInputParam("reference", std::move(referenceData));
  // Same input as first setting, k <= reference points.
  SetInputParam("k", (int) 5);
  // Random input, num_tables > 0.
  SetInputParam("num_tables", (int) 4);

  SetInputParam("num_projections", (int) 10);
  // Second solution.
  mlpackMain();

  // Get the distances matrix after second training.
  arma::mat secondOutputDistances =
      std::move(IO::GetParam<arma::mat>("distances"));

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
  mlpackMain();

  // Get the distances matrix after first training.
  arma::mat firstOutputDistances =
      std::move(IO::GetParam<arma::mat>("distances"));

  // Reset the settings.
  bindings::tests::CleanMemory();
  IO::ClearSettings();
  IO::RestoreSettings(testName);
  // Second setting.
  referenceData.randu(2, 80); // 80 points in 2 dimensions.
  SetInputParam("reference", std::move(referenceData));
  // Same input as first setting, k <= reference points.
  SetInputParam("k", (int) 5);
  // Random input, num_tables > 0.
  SetInputParam("num_projections", (int) 6);
  SetInputParam("num_tables", (int) 3);

  // Second solution.
  mlpackMain();

  // Get the distances matrix after second training.
  arma::mat secondOutputDistances =
      std::move(IO::GetParam<arma::mat>("distances"));

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

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
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
  mlpackMain();

  // Get the distances and neighbors matrix after first training.
  arma::mat firstOutputDistances =
  std::move(IO::GetParam<arma::mat>("distances"));
  arma::Mat<size_t> firstOutputNeighbors =
    std::move(IO::GetParam<arma::Mat<size_t>>("neighbors"));

  // Reset the settings.
  bindings::tests::CleanMemory();
  IO::ClearSettings();
  IO::RestoreSettings(testName);

  // Second solution.
  SetInputParam("reference", std::move(referenceData));
  // Random input, any k <= reference points.
  SetInputParam("k", (int) 10);
  SetInputParam("algorithm", (string) "qdafn");

  // Get the distances and neighbors matrix after second training.
  arma::mat secondOutputDistances =
      std::move(IO::GetParam<arma::mat>("distances"));
  arma::Mat<size_t> secondOutputNeighbors =
      std::move(IO::GetParam<arma::Mat<size_t>>("neighbors"));

  // Check that the distance matrices (firstOutputDistances and
  // secondOutputDistances) and neighbor matrices (firstOutputNeighbors and
  // secondOutputNeighbors) are not equal. This ensures that the two strategies
  // result in different outputs.
  CheckMatricesNotEqual(firstOutputDistances, secondOutputDistances);
  CheckMatricesNotEqual(firstOutputNeighbors, secondOutputNeighbors);
}
