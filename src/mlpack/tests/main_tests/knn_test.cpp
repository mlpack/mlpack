/**
 * @file knn_test.cpp
 * @author Atharva Khandait
 *
 * Test mlpackMain() of knn_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <string>

#define BINDING_TYPE BINDING_TYPE_TEST
static const std::string testName = "K-NearestNeighborsSearch";

#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "test_helper.hpp"
#include <mlpack/methods/neighbor_search/knn_main.cpp>

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

struct KNNTestFixture
{
 public:
  KNNTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~KNNTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(KNNMainTest, KNNTestFixture);

/*
 * Check that we can't provide reference and query matrices
 * with different dimensions.
 */
BOOST_AUTO_TEST_CASE(KNNEqualDimensionTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 10); // 10 points in 3 dimensions.

  /* Now we specify an invalid dimension(2) for the query data.
   * Note that the number of points in query and reference matrices
   * are allowed to be different
   */
  arma::mat queryData;
  queryData.randu(2, 10);

  // Random input, some k <= number of reference points 
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("query", std::move(queryData));
  SetInputParam("k", (int) 4);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/*
 * Check that we can't specify an invalid k when only reference
 * matrix is given. 
 */
BOOST_AUTO_TEST_CASE(KNNInvalidKTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 10); // 10 points in 3 dimensions.

  // Random input, some k > number of reference points
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 11);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/*
 * Check that we can't specify an invalid k when both reference
 * and query matrices are given. 
 */
BOOST_AUTO_TEST_CASE(KNNInvalidKQueryDataTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 10); // 10 points in 3 dimensions.

  arma::mat queryData;
  queryData.randu(3, 10); // 10 points in 3 dimensions.

  // Random input, some k > number of reference points
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("query", std::move(queryData));
  SetInputParam("k", (int) 11);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Check that we can't specify a negative leaf size.
 */
BOOST_AUTO_TEST_CASE(KNNLeafSizeTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 10); // 10 points in 3 dimensions.
  
  // Random input, negative leaf size.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("leaf_size", (int) -1); // Invalid.

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/*
 * Check that we can't pass both input_model and reference matrix.
 */
BOOST_AUTO_TEST_CASE(KNNRefModelTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 10); // 10 points in 3 dimensions.

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 4);

  mlpackMain();

  // Input pre-trained model.
  SetInputParam("input_model", 
      std::move(CLI::GetParam<KNNModel*>("output_model")));
  
  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Make sure that dimensions of the neighbors and distances
 * matrices are correct given a value of k.  
 */
BOOST_AUTO_TEST_CASE(KNNOutputDimensionTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 10); // 10 points in 3 dimensions.
  
  // Random input, k = 3.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 3);
  
  mlpackMain();

  // Check the neighbors matrix has 3 points for each input point.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Mat<size_t>>("neighbors").n_rows, 3);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Mat<size_t>>("neighbors").n_cols, 10);

  // Check the distances matrix has 3 points for each input point.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("distances").n_rows, 3);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("distances").n_cols, 10);
}

/**
 * Ensure that saved model can be used again.
 */
BOOST_AUTO_TEST_CASE(KNNModelReuseTest)
{
  arma::mat referenceData;
  if (!data::Load("test_data_3_1000.csv", referenceData))
    BOOST_FAIL("Cannot load labels for test_data_3_1000.csv");

  arma::mat queryData;
  if (!data::Load("rann_test_r_3_900.csv", queryData))
    BOOST_FAIL("Cannot load labels for rann_test_r_3_900.csv");

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("query", std::move(queryData));
  SetInputParam("k", (int) 4);

  mlpackMain();

  arma::Mat<size_t> neighbors;
  arma::mat distances;
  neighbors = std::move(CLI::GetParam<arma::Mat<size_t>>("neighbors"));
  distances = std::move(CLI::GetParam<arma::mat>("distances"));

  // Reset passed parameters. 
  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  CLI::GetSingleton().Parameters()["query"].wasPassed = false;

  // Input saved model, keep query and k unchanged.
  SetInputParam("input_model", 
      std::move(CLI::GetParam<KNNModel*>("output_model")));
  if (!data::Load("rann_test_r_3_900.csv", queryData))
    BOOST_FAIL("Cannot load labels for rann_test_r_3_900.csv");
  SetInputParam("query", std::move(queryData));
  
  mlpackMain();

  // Check that initial output matrices and the output matrices using saved model are equal.
  CheckMatrices(neighbors, CLI::GetParam<arma::Mat<size_t>>("neighbors"));
  CheckMatrices(distances, CLI::GetParam<arma::mat>("distances"));
}

BOOST_AUTO_TEST_SUITE_END();