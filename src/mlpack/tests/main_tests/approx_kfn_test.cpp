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
  SetInputParam("input_model", std::move(CLI::GetParam<ApproxKFNModel*>("output_model")));
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
  SetInputParam("k", (int) 81); //Invalid

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::invalid_argument);
  Log::Fatal.ignoreInput = false;
}

/**
 * Make sure that the dimensions of neighbors and distances is correct given a value of k.
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
  SetInputParam("algorithm", (string) "any_algo"); //Invalid

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

BOOST_AUTO_TEST_SUITE_END();