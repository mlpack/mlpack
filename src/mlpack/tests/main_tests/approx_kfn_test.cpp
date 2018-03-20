/**
 * @file approx_kfn_test.cpp
 * @author Namrata Mukhija
 *
 * Test mlpackMain() of approx_kfn_main.cpp.
 */
#include <string>

#define BINDING_TYPE BINDING_TYPE_TEST
static const std::string testName = "ApproxK-FurthestNeighbors";

#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "test_helper.hpp"
#include <mlpack/methods/approx_kfn/approx_kfn_main.cpp>

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

struct ApproxKFNTestFixture
{
 public:
  ApproxKFNTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~ApproxKFNTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(ApproxKFNMainTest, ApproxKFNTestFixture);

/**
 * Check that we can't specify both a reference set and an input model.
 */
BOOST_AUTO_TEST_CASE(ApproxKFNRefModelTest)
{
  arma::mat referenceData;
  referenceData.randu(2, 80); // 80 points in 2 dimensions.

  // Random input, any k <= reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);

  // Input pre-trained model.
  SetInputParam("input_model", std::move(CLI::GetParam<ApproxKFNModel*>
    ("output_model")));
  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Check that we can't specify an invalid k.
 */
BOOST_AUTO_TEST_CASE(ApproxKFNInvalidKTest)
{
  arma::mat referenceData;
  referenceData.randu(2, 80); // 80 points in 2 dimensions.

  SetInputParam("reference", std::move(referenceData));
  // Random input, k > reference points.
  SetInputParam("k", (int) 81); // Invalid.

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::invalid_argument);
  Log::Fatal.ignoreInput = false;
}

/**
 * Make sure that the dimensions of neighbors and distances is correct given a
   value of k.
 */
BOOST_AUTO_TEST_CASE(ApproxKFNOutputDimensionTest)
{
  arma::mat referenceData;
  referenceData.randu(2, 80); // 80 points in 2 dimensions.

  // Random input, any k <= reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);

  mlpackMain();

  // Check the neighbors matrix has 10 points for each of the 80 input points.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Mat<size_t>>("neighbors").n_rows, 10);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Mat<size_t>>("neighbors").n_cols, 80);

  // Check the distances matrix has 10 points for each of the 80 input points.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("distances").n_rows, 10);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("distances").n_cols, 80);
}

/**
 * Check that we can't specify an invalid algorithm.
 */
BOOST_AUTO_TEST_CASE(ApproxKFNInvalidAlgorithmTest)
{
  arma::mat referenceData;
  referenceData.randu(2, 80); // 80 points in 2 dimensions.

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("algorithm", (string) "any_algo"); // Invalid.

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Check that we can't specify num_projections as zero.
 */
BOOST_AUTO_TEST_CASE(ApproxKFNZeroNumProjTest)
{
  arma::mat referenceData;
  referenceData.randu(2, 80); // 80 points in 2 dimensions.

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);
  SetInputParam("num_projections", (int) 0);//Invalid

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Check that we can't specify num_projections as negative.
 */
BOOST_AUTO_TEST_CASE(ApproxKFNNegativeNumProjTest)
{
  arma::mat referenceData;
  referenceData.randu(2, 80); // 80 points in 2 dimensions.

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);
  SetInputParam("num_projections", (int) -5); // Invalid.

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Check that we can't specify num_tables as zero.
 */
BOOST_AUTO_TEST_CASE(ApproxKFNZeroNumTablesTest)
{
  arma::mat referenceData;
  referenceData.randu(2, 80); // 80 points in 2 dimensions.

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);
  SetInputParam("num_tables", (int) 0); // Invalid.

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Check that we can't specify num_tables as negative.
 */
BOOST_AUTO_TEST_CASE(ApproxKFNNegativeNumTablesTest)
{
  arma::mat referenceData;
  referenceData.randu(2, 80); // 80 points in 2 dimensions.
 
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);
  SetInputParam("num_tables", (int) -5); // Invalid.

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensuring that a saved model can be loaded and used again correctly.
 */
BOOST_AUTO_TEST_CASE(ApproxKFNModelReuseTest)
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
  neighbors = std::move(CLI::GetParam<arma::Mat<size_t>>("neighbors"));
  distances = std::move(CLI::GetParam<arma::mat>("distances"));

  // Reset passed parameters.
  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  CLI::GetSingleton().Parameters()["query"].wasPassed = false;

  // Input saved model, pass the same query and keep k unchanged.
  SetInputParam("input_model",
      std::move(CLI::GetParam<ApproxKFNModel*>("output_model")));
  SetInputParam("query", queryData);

  mlpackMain();

  // Check that initial output matrices and the output matrices using
  // saved model are equal.
  CheckMatrices(neighbors, CLI::GetParam<arma::Mat<size_t>>("neighbors"));
  CheckMatrices(distances, CLI::GetParam<arma::mat>("distances"));
}

/**
 * Make sure that the dimensions of the exact distances matrix are correct.
 */
BOOST_AUTO_TEST_CASE(ApproxKFNExactDistDimensionTest)
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
  exactDistances.randu(9, 90); // Wrong size (should be (10, 80)).
  SetInputParam("exact_distances", std::move(exactDistances));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

BOOST_AUTO_TEST_SUITE_END();