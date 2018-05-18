/**
 * @file lsh_test.cpp
 * @author Manish Kumar
 *
 * Test mlpackMain() of lsh_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <string>

#define BINDING_TYPE BINDING_TYPE_TEST
static const std::string testName = "LSH";

#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "test_helper.hpp"
#include <mlpack/methods/lsh/lsh_main.cpp>

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

struct LSHTestFixture
{
 public:
  LSHTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~LSHTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(LSHMainTest, LSHTestFixture);

/**
 * Check that output neighbors and distances have valid dimensions.
 */
BOOST_AUTO_TEST_CASE(LSHOutputDimensionTest)
{
  arma::mat reference = arma::randu<arma::mat>(5, 100);

  SetInputParam("reference", std::move(reference));
  SetInputParam("k", (int) 6);

  mlpackMain();

  // Check the neighbors matrix has 6 points for each of the 100 input points.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Mat<size_t>>("neighbors").n_rows, 6);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Mat<size_t>>("neighbors").n_cols,
                      100);

  // Check the distances matrix has 6 points for each of the 100 input points.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("distances").n_rows, 6);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("distances").n_cols, 100);
}

/**
 * Ensure that bucket_size, second_hash_size & number of nearest neighbors
 * are always positive.
 */
BOOST_AUTO_TEST_CASE(LSHParamValidityTest)
{
  arma::mat reference = arma::randu<arma::mat>(5, 100);

  // Test for bucket_size.

  SetInputParam("reference", reference);
  SetInputParam("k", (int) 6);
  SetInputParam("bucket_size", (int) -1.0);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  bindings::tests::CleanMemory();

  // Test for second_hash_size.

  SetInputParam("reference", reference);
  SetInputParam("k", (int) 6);
  SetInputParam("second_hash_size", (int) -1.0);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  bindings::tests::CleanMemory();

  // Test for number of nearest neighbors.

  SetInputParam("reference", std::move(reference));
  SetInputParam("k", (int) -2);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Make sure only one of reference data or pre-trained model is passed.
 */
BOOST_AUTO_TEST_CASE(LSHModelValidityTest)
{
  arma::mat reference = arma::randu<arma::mat>(5, 100);

  SetInputParam("reference", std::move(reference));
  SetInputParam("k", (int) 6);

  mlpackMain();

  SetInputParam("input_model", CLI::GetParam<LSHSearch<>*>("output_model"));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Check learning process using different tables.
 */
BOOST_AUTO_TEST_CASE(LSHDiffTablesTest)
{
  arma::mat reference = arma::randu<arma::mat>(5, 100);

  SetInputParam("reference", reference);
  SetInputParam("k", (int) 6);

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  arma::Mat<size_t> neighbors = CLI::GetParam<arma::Mat<size_t>>("neighbors");
  arma::mat distances = CLI::GetParam<arma::mat>("distances");

  bindings::tests::CleanMemory();

  // Train model using tables equals to 40.

  SetInputParam("reference", std::move(reference));
  SetInputParam("k", (int) 6);
  SetInputParam("tables", (int) 40);

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Check that initial outputs and final outputs using two models are
  // different.
  BOOST_REQUIRE_LT(arma::accu(neighbors ==
      CLI::GetParam<arma::Mat<size_t>>("neighbors")), neighbors.n_elem);
  BOOST_REQUIRE_LT(arma::accu(distances ==
      CLI::GetParam<arma::mat>("distances")), distances.n_elem);
}

/**
 * Check learning process using different projections.
 */
BOOST_AUTO_TEST_CASE(LSHDiffProjectionsTest)
{
  arma::mat reference = arma::randu<arma::mat>(5, 100);

  SetInputParam("reference", reference);
  SetInputParam("k", (int) 6);

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  arma::Mat<size_t> neighbors = CLI::GetParam<arma::Mat<size_t>>("neighbors");
  arma::mat distances = CLI::GetParam<arma::mat>("distances");

  bindings::tests::CleanMemory();

  // Train model using projections equals to 30.

  SetInputParam("reference", std::move(reference));
  SetInputParam("k", (int) 6);
  SetInputParam("projections", (int) 30);

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Check that initial outputs and final outputs using two models are
  // different.
  BOOST_REQUIRE_LT(arma::accu(neighbors ==
      CLI::GetParam<arma::Mat<size_t>>("neighbors")), neighbors.n_elem);
  BOOST_REQUIRE_LT(arma::accu(distances ==
      CLI::GetParam<arma::mat>("distances")), distances.n_elem);
}

/**
 * Check learning process using different hash_width.
 */
BOOST_AUTO_TEST_CASE(LSHDiffHashWidthTest)
{
  arma::mat reference = arma::randu<arma::mat>(5, 100);

  SetInputParam("reference", reference);
  SetInputParam("k", (int) 6);

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  arma::Mat<size_t> neighbors = CLI::GetParam<arma::Mat<size_t>>("neighbors");
  arma::mat distances = CLI::GetParam<arma::mat>("distances");

  bindings::tests::CleanMemory();

  // Train model using hash_width equals to 0.5.

  SetInputParam("reference", std::move(reference));
  SetInputParam("k", (int) 6);
  SetInputParam("hash_width", (double) 0.5);

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Check that initial outputs and final outputs using two models are
  // different.
  BOOST_REQUIRE_LT(arma::accu(neighbors ==
      CLI::GetParam<arma::Mat<size_t>>("neighbors")), neighbors.n_elem);
  BOOST_REQUIRE_LT(arma::accu(distances ==
      CLI::GetParam<arma::mat>("distances")), distances.n_elem);
}

/**
 * Check learning process using different num_probes.
 */
BOOST_AUTO_TEST_CASE(LSHDiffNumProbesTest)
{
  arma::mat reference = arma::randu<arma::mat>(5, 100);
  arma::mat query = arma::randu<arma::mat>(5, 40);

  SetInputParam("reference", std::move(reference));
  SetInputParam("query", query);
  SetInputParam("k", (int) 6);

  mlpackMain();

  arma::Mat<size_t> neighbors = CLI::GetParam<arma::Mat<size_t>>("neighbors");
  arma::mat distances = CLI::GetParam<arma::mat>("distances");

  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;

  // Train model using num_probes equals to 5.

  SetInputParam("input_model", CLI::GetParam<LSHSearch<>*>("output_model"));
  SetInputParam("query", std::move(query));
  SetInputParam("num_probes", (int) 5);

  mlpackMain();

  // Check that initial outputs and final outputs using two models are
  // different.
  BOOST_REQUIRE_LT(arma::accu(neighbors ==
      CLI::GetParam<arma::Mat<size_t>>("neighbors")), neighbors.n_elem);
  BOOST_REQUIRE_LT(arma::accu(distances ==
      CLI::GetParam<arma::mat>("distances")), distances.n_elem);
}

/**
 * Check learning process using different second_hash_size.
 */
BOOST_AUTO_TEST_CASE(LSHDiffSecondHashSizeTest)
{
  arma::mat reference = arma::randu<arma::mat>(5, 100);

  SetInputParam("reference", reference);
  SetInputParam("k", (int) 6);

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  arma::Mat<size_t> neighbors = CLI::GetParam<arma::Mat<size_t>>("neighbors");
  arma::mat distances = CLI::GetParam<arma::mat>("distances");

  bindings::tests::CleanMemory();

  // Train model using second_hash_size equals to 5000.

  SetInputParam("reference", std::move(reference));
  SetInputParam("k", (int) 6);
  SetInputParam("second_hash_size", (int) 5000);

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Check that initial outputs and final outputs using two models are
  // different.
  BOOST_REQUIRE_LT(arma::accu(neighbors ==
      CLI::GetParam<arma::Mat<size_t>>("neighbors")), neighbors.n_elem);
  BOOST_REQUIRE_LT(arma::accu(distances ==
      CLI::GetParam<arma::mat>("distances")), distances.n_elem);
}

/**
 * Check learning process using different bucket sizes.
 */
BOOST_AUTO_TEST_CASE(LSHDiffBucketSizeTest)
{
  arma::mat reference = arma::randu<arma::mat>(5, 100);

  SetInputParam("reference", reference);
  SetInputParam("k", (int) 6);

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  arma::Mat<size_t> neighbors = CLI::GetParam<arma::Mat<size_t>>("neighbors");
  arma::mat distances = CLI::GetParam<arma::mat>("distances");

  bindings::tests::CleanMemory();

  // Train model using bucket_size equals to 1000.

  SetInputParam("reference", std::move(reference));
  SetInputParam("k", (int) 6);
  SetInputParam("bucket_size", (int) 1);

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Check that initial outputs and final outputs using the two models are
  // different.
  BOOST_REQUIRE_LT(arma::accu(neighbors ==
      CLI::GetParam<arma::Mat<size_t>>("neighbors")), neighbors.n_elem);
  BOOST_REQUIRE_LT(arma::accu(distances ==
      CLI::GetParam<arma::mat>("distances")), distances.n_elem);
}

/**
 * Check that saved model can be reused again.
 */
BOOST_AUTO_TEST_CASE(LSHModelReuseTest)
{
  arma::mat reference = arma::randu<arma::mat>(5, 100);
  arma::mat query = arma::randu<arma::mat>(5, 40);

  SetInputParam("reference", std::move(reference));
  SetInputParam("query", query);
  SetInputParam("k", (int) 6);

  mlpackMain();

  arma::Mat<size_t> neighbors = CLI::GetParam<arma::Mat<size_t>>("neighbors");
  arma::mat distances = CLI::GetParam<arma::mat>("distances");

  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;

  SetInputParam("input_model", CLI::GetParam<LSHSearch<>*>("output_model"));
  SetInputParam("query", std::move(query));

  mlpackMain();

  // Check that initial query outputs and final outputs using saved model are
  // same.
  CheckMatrices(neighbors, CLI::GetParam<arma::Mat<size_t>>("neighbors"));
  CheckMatrices(distances, CLI::GetParam<arma::mat>("distances"));
}

/**
 * Make sure true_neighbors have valid dimensions.
 */
BOOST_AUTO_TEST_CASE(LSHModelTrueNighborsDimTest)
{
  arma::mat reference = arma::randu<arma::mat>(5, 100);

  // Initalize trueNeighbors with invalid dimensions.
  arma::Mat<size_t> trueNeighbors = arma::randu<arma::Mat<size_t>>(7, 100);

  SetInputParam("reference", std::move(reference));
  SetInputParam("true_neighbors", std::move(trueNeighbors));
  SetInputParam("k", (int) 6);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

BOOST_AUTO_TEST_SUITE_END();
