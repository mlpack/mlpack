/**
 * @file tests/main_tests/lsh_test.cpp
 * @author Manish Kumar
 *
 * Test RUN_BINDING() of lsh_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/lsh/lsh_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "main_test_fixture.hpp"

#include "../catch.hpp"
#include "../test_catch_tools.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(LSHTestFixture);

/**
 * Check that output neighbors and distances have valid dimensions.
 */
TEST_CASE_METHOD(LSHTestFixture, "LSHOutputDimensionTest",
                 "[LSHMainTest][BindingTests]")
{
  arma::mat reference = arma::randu<arma::mat>(5, 100);

  SetInputParam("reference", std::move(reference));
  SetInputParam("k", (int) 6);

  RUN_BINDING();

  // Check the neighbors matrix has 6 points for each of the 100 input points.
  REQUIRE(params.Get<arma::Mat<size_t>>("neighbors").n_rows == 6);
  REQUIRE(params.Get<arma::Mat<size_t>>("neighbors").n_cols == 100);

  // Check the distances matrix has 6 points for each of the 100 input points.
  REQUIRE(params.Get<arma::mat>("distances").n_rows == 6);
  REQUIRE(params.Get<arma::mat>("distances").n_cols == 100);
}

/**
 * Ensure that bucket_size, second_hash_size & number of nearest neighbors
 * are always positive.
 */
TEST_CASE_METHOD(LSHTestFixture, "LSHParamValidityTest",
                 "[LSHMainTest][BindingTests]")
{
  arma::mat reference = arma::randu<arma::mat>(5, 100);

  // Test for bucket_size.

  SetInputParam("reference", reference);
  SetInputParam("k", (int) 6);
  SetInputParam("bucket_size", (int) -1.0);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);

  CleanMemory();
  ResetSettings();

  // Test for second_hash_size.

  SetInputParam("reference", reference);
  SetInputParam("k", (int) 6);
  SetInputParam("second_hash_size", (int) -1.0);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);

  CleanMemory();
  ResetSettings();

  // Test for number of nearest neighbors.

  SetInputParam("reference", std::move(reference));
  SetInputParam("k", (int) -2);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Make sure only one of reference data or pre-trained model is passed.
 */
TEST_CASE_METHOD(LSHTestFixture, "LSHModelValidityTest",
                 "[LSHMainTest][BindingTests]")
{
  arma::mat reference = arma::randu<arma::mat>(5, 100);

  SetInputParam("reference", std::move(reference));
  SetInputParam("k", (int) 6);

  RUN_BINDING();

  SetInputParam("input_model", params.Get<LSHSearch<>*>("output_model"));

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Check learning process using different tables.
 */
TEST_CASE_METHOD(LSHTestFixture, "LSHDiffTablesTest",
                 "[LSHMainTest][BindingTests]")
{
  arma::mat reference = arma::randu<arma::mat>(5, 100);

  SetInputParam("reference", reference);
  SetInputParam("k", (int) 6);

  FixedRandomSeed();
  RUN_BINDING();

  arma::Mat<size_t> neighbors = params.Get<arma::Mat<size_t>>("neighbors");
  arma::mat distances = params.Get<arma::mat>("distances");

  CleanMemory();
  ResetSettings();

  // Train model using tables equals to 40.

  SetInputParam("reference", std::move(reference));
  SetInputParam("k", (int) 6);
  SetInputParam("tables", (int) 40);

  FixedRandomSeed();
  RUN_BINDING();

  // Check that initial outputs and final outputs using two models are
  // different.
  REQUIRE(arma::accu(neighbors ==
      params.Get<arma::Mat<size_t>>("neighbors")) < neighbors.n_elem);
  REQUIRE(arma::accu(distances ==
      params.Get<arma::mat>("distances")) < distances.n_elem);
}

/**
 * Check learning process using different projections.
 */
TEST_CASE_METHOD(LSHTestFixture, "LSHDiffProjectionsTest",
                 "[LSHMainTest][BindingTests]")
{
  arma::mat reference = arma::randu<arma::mat>(5, 100);

  SetInputParam("reference", reference);
  SetInputParam("k", (int) 6);

  FixedRandomSeed();
  RUN_BINDING();

  arma::Mat<size_t> neighbors = params.Get<arma::Mat<size_t>>("neighbors");
  arma::mat distances = params.Get<arma::mat>("distances");

  CleanMemory();
  ResetSettings();

  // Train model using projections equals to 30.

  SetInputParam("reference", std::move(reference));
  SetInputParam("k", (int) 6);
  SetInputParam("projections", (int) 30);

  FixedRandomSeed();
  RUN_BINDING();

  // Check that initial outputs and final outputs using two models are
  // different.
  REQUIRE(arma::accu(neighbors ==
      params.Get<arma::Mat<size_t>>("neighbors")) < neighbors.n_elem);
  REQUIRE(arma::accu(distances ==
      params.Get<arma::mat>("distances")) < distances.n_elem);
}

/**
 * Check learning process using different hash_width.
 */
TEST_CASE_METHOD(LSHTestFixture, "LSHDiffHashWidthTest",
                 "[LSHMainTest][BindingTests]")
{
  arma::mat reference = arma::randu<arma::mat>(5, 100);

  SetInputParam("reference", reference);
  SetInputParam("k", (int) 6);

  FixedRandomSeed();
  RUN_BINDING();

  arma::Mat<size_t> neighbors = params.Get<arma::Mat<size_t>>("neighbors");
  arma::mat distances = params.Get<arma::mat>("distances");

  CleanMemory();
  ResetSettings();

  // Train model using hash_width equals to 0.5.

  SetInputParam("reference", std::move(reference));
  SetInputParam("k", (int) 6);
  SetInputParam("hash_width", (double) 0.5);

  FixedRandomSeed();
  RUN_BINDING();

  // Check that initial outputs and final outputs using two models are
  // different.
  REQUIRE(arma::accu(neighbors ==
      params.Get<arma::Mat<size_t>>("neighbors")) < neighbors.n_elem);
  REQUIRE(arma::accu(distances ==
      params.Get<arma::mat>("distances")) < distances.n_elem);
}

/**
 * Check learning process using different num_probes.
 */
TEST_CASE_METHOD(LSHTestFixture, "LSHDiffNumProbesTest",
                 "[LSHMainTest][BindingTests]")
{
  arma::mat reference = arma::randu<arma::mat>(5, 100);
  arma::mat query = arma::randu<arma::mat>(5, 40);

  SetInputParam("reference", std::move(reference));
  SetInputParam("query", query);
  SetInputParam("k", (int) 6);

  RUN_BINDING();

  arma::Mat<size_t> neighbors = params.Get<arma::Mat<size_t>>("neighbors");
  arma::mat distances = params.Get<arma::mat>("distances");

  LSHSearch<>* m = params.Get<LSHSearch<>*>("output_model");
  params.Get<LSHSearch<>*>("output_model") = NULL;
  CleanMemory();
  ResetSettings();

  // Train model using num_probes equals to 5.

  SetInputParam("input_model", m);
  SetInputParam("query", std::move(query));
  SetInputParam("num_probes", (int) 5);
  SetInputParam("k", (int) 6);

  RUN_BINDING();

  // Check that initial outputs and final outputs using two models are
  // different.
  REQUIRE(arma::accu(neighbors ==
      params.Get<arma::Mat<size_t>>("neighbors")) < neighbors.n_elem);
  REQUIRE(arma::accu(distances ==
      params.Get<arma::mat>("distances")) < distances.n_elem);
}

/**
 * Check learning process using different second_hash_size.
 */
TEST_CASE_METHOD(LSHTestFixture, "LSHDiffSecondHashSizeTest",
                 "[LSHMainTest][BindingTests]")
{
  arma::mat reference = arma::randu<arma::mat>(5, 100);

  SetInputParam("reference", reference);
  SetInputParam("k", (int) 6);

  FixedRandomSeed();
  RUN_BINDING();

  arma::Mat<size_t> neighbors = params.Get<arma::Mat<size_t>>("neighbors");
  arma::mat distances = params.Get<arma::mat>("distances");

  CleanMemory();
  ResetSettings();

  // Train model using second_hash_size equals to 5000.

  SetInputParam("reference", std::move(reference));
  SetInputParam("k", (int) 6);
  SetInputParam("second_hash_size", (int) 5000);

  FixedRandomSeed();
  RUN_BINDING();

  // Check that initial outputs and final outputs using two models are
  // different.
  REQUIRE(arma::accu(neighbors ==
      params.Get<arma::Mat<size_t>>("neighbors")) < neighbors.n_elem);
  REQUIRE(arma::accu(distances ==
      params.Get<arma::mat>("distances")) < distances.n_elem);
}

/**
 * Check learning process using different bucket sizes.
 */
TEST_CASE_METHOD(LSHTestFixture, "LSHDiffBucketSizeTest",
                 "[LSHMainTest][BindingTests]")
{
  arma::mat reference = arma::randu<arma::mat>(5, 100);

  SetInputParam("reference", reference);
  SetInputParam("k", (int) 6);

  FixedRandomSeed();
  RUN_BINDING();

  arma::Mat<size_t> neighbors = params.Get<arma::Mat<size_t>>("neighbors");
  arma::mat distances = params.Get<arma::mat>("distances");

  CleanMemory();
  ResetSettings();

  // Train model using bucket_size equals to 1000.

  SetInputParam("reference", std::move(reference));
  SetInputParam("k", (int) 6);
  SetInputParam("bucket_size", (int) 1);

  FixedRandomSeed();
  RUN_BINDING();

  // Check that initial outputs and final outputs using the two models are
  // different.
  REQUIRE(arma::accu(neighbors ==
      params.Get<arma::Mat<size_t>>("neighbors")) < neighbors.n_elem);
  REQUIRE(arma::accu(distances ==
      params.Get<arma::mat>("distances")) < distances.n_elem);
}

/**
 * Check that saved model can be reused again.
 */
TEST_CASE_METHOD(LSHTestFixture, "LSHModelReuseTest",
                 "[LSHMainTest][BindingTests]")
{
  arma::mat reference = arma::randu<arma::mat>(5, 100);
  arma::mat query = arma::randu<arma::mat>(5, 40);

  SetInputParam("reference", std::move(reference));
  SetInputParam("query", query);
  SetInputParam("k", (int) 6);

  RUN_BINDING();

  arma::Mat<size_t> neighbors = params.Get<arma::Mat<size_t>>("neighbors");
  arma::mat distances = params.Get<arma::mat>("distances");

  LSHSearch<>* m = params.Get<LSHSearch<>*>("output_model");
  params.Get<LSHSearch<>*>("output_model") = NULL;

  CleanMemory();
  ResetSettings();

  SetInputParam("input_model", m);
  SetInputParam("query", std::move(query));
  SetInputParam("k", (int) 6);

  RUN_BINDING();

  // Check that initial query outputs and final outputs using saved model are
  // same.
  CheckMatrices(neighbors, params.Get<arma::Mat<size_t>>("neighbors"));
  CheckMatrices(distances, params.Get<arma::mat>("distances"));
}

/**
 * Make sure true_neighbors have valid dimensions.
 */
TEST_CASE_METHOD(LSHTestFixture, "LSHModelTrueNighborsDimTest",
                 "[LSHMainTest][BindingTests]")
{
  arma::mat reference = arma::randu<arma::mat>(5, 100);

  // Initalize trueNeighbors with invalid dimensions.
  arma::Mat<size_t> trueNeighbors = arma::randu<arma::Mat<size_t>>(7, 100);

  SetInputParam("reference", std::move(reference));
  SetInputParam("true_neighbors", std::move(trueNeighbors));
  SetInputParam("k", (int) 6);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}
