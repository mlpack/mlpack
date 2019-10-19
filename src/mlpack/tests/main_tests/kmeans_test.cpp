/**
 * @file kmeans_test.cpp
 * @author Prabhat Sharma
 *
 * Test mlpackMain() of kmeans_main.cpp
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <string>

#define BINDING_TYPE BINDING_TYPE_TEST
static const std::string testName = "Kmeans";

#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "test_helper.hpp"
#include <mlpack/methods/kmeans/kmeans_main.cpp>

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

struct KmTestFixture
{
 public:
  KmTestFixture()
  {
      // Cache in the options for this program.
      CLI::RestoreSettings(testName);
  }

  ~KmTestFixture()
  {
      // Clear the settings.
      CLI::ClearSettings();
  }
};

void ResetKmSettings()
{
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);
}

BOOST_FIXTURE_TEST_SUITE(KmeansMainTest, KmTestFixture);

/**
 * Checking that number of Clusters are non negative
 */
BOOST_AUTO_TEST_CASE(NonNegativeClustersTest)
{
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    BOOST_FAIL("Unable to load train dataset vc2.csv!");

  SetInputParam("input", std::move(inputData));
  SetInputParam("clusters", (int) -1); // Invalid

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}


/**
 * Checking that initial centroids are provided if clusters are to be auto detected
 */
BOOST_AUTO_TEST_CASE(AutoDetectClusterTest)
{
  constexpr int N = 10;
  constexpr int D = 4;

  arma::mat inputData = arma::randu<arma::mat>(N, D);

  SetInputParam("input", std::move(inputData));
  SetInputParam("clusters", (int) 0); // Invalid

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}


/**
 * Checking that percentage is between 0 and 1 when --refined_start is specified
*/
BOOST_AUTO_TEST_CASE(RefinedStartPercentageTest)
{
  int c = 2;
  double P = 2.0;
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    BOOST_FAIL("Unable to load train dataset vc2.csv!");

  SetInputParam("input", std::move(inputData));
  SetInputParam("refined_start", true);
  SetInputParam("clusters", c);
  SetInputParam("percentage", std::move(P));     // Invalid

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}


/**
 * Checking percentage is non-negative when --refined_start is specified
 */
BOOST_AUTO_TEST_CASE(NonNegativePercentageTest)
{
  int c = 2;
  double P = -1.0;
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    BOOST_FAIL("Unable to load train dataset vc2.csv!");

  SetInputParam("input", std::move(inputData));
  SetInputParam("refined_start", true);
  SetInputParam("clusters", c);
  SetInputParam("percentage", P);     // Invalid

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}


/**
 * Checking that size and dimensionality of prediction is correct.
 */
BOOST_AUTO_TEST_CASE(KmClusteringSizeCheck)
{
  int c = 2;
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    BOOST_FAIL("Unable to load train dataset vc2.csv!");

  size_t col = inputData.n_cols;
  size_t row = inputData.n_rows;

  SetInputParam("input", std::move(inputData));
  SetInputParam("clusters", c);

  mlpackMain();

  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_rows, row+1);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_cols, col);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("centroid").n_rows, row);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("centroid").n_cols, c);
}

/**
 * Checking that size and dimensionality of prediction is correct when --labels_only is specified
 */
BOOST_AUTO_TEST_CASE(KmClusteringSizeCheckLabelOnly)
{
  int c = 2;

  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    BOOST_FAIL("Unable to load train dataset vc2.csv!");
  size_t col = inputData.n_cols;
  size_t row = inputData.n_rows;

  SetInputParam("input", std::move(inputData));
  SetInputParam("clusters", c);
  SetInputParam("labels_only", true);

  mlpackMain();

  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_rows, 1);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_cols, col);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("centroid").n_rows, row);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("centroid").n_cols, c);
}


/**
 * Checking that predictions are not same when --allow_empty_clusters or kill_empty_clusters are specified
 */
BOOST_AUTO_TEST_CASE(KmClusteringEmptyClustersCheck)
{
  int c = 400;
  int iterations = 100;

  arma::mat inputData;
  if (!data::Load("test_data_3_1000.csv", inputData))
    BOOST_FAIL("Unable to load train dataset test_data_3_1000.csv!");
  arma::mat initCentroid = arma::randu<arma::mat>(inputData.n_rows, c);

  SetInputParam("input", inputData);
  SetInputParam("clusters", c);
  SetInputParam("labels_only", true);
  SetInputParam("max_iterations", iterations);
  SetInputParam("initial_centroids", initCentroid);

  mlpackMain();

  arma::mat normalOutput;
  normalOutput = std::move(CLI::GetParam<arma::mat>("centroid"));

  ResetKmSettings();

  SetInputParam("input", inputData);
  SetInputParam("clusters", c);
  SetInputParam("labels_only", true);
  SetInputParam("allow_empty_clusters", true);
  SetInputParam("max_iterations", iterations);
  SetInputParam("initial_centroids", initCentroid);

  mlpackMain();

  arma::mat allowEmptyOutput;
  allowEmptyOutput = std::move(CLI::GetParam<arma::mat>("centroid"));

  ResetKmSettings();

  SetInputParam("input", inputData);
  SetInputParam("clusters", c);
  SetInputParam("labels_only", true);
  SetInputParam("kill_empty_clusters", true);
  SetInputParam("max_iterations", iterations);
  SetInputParam("initial_centroids", initCentroid);

  mlpackMain();

  arma::mat killEmptyOutput;
  killEmptyOutput = std::move(CLI::GetParam<arma::mat>("centroid"));

  ResetKmSettings();

  if (killEmptyOutput.n_elem == allowEmptyOutput.n_elem)
  {
    BOOST_REQUIRE_GT(arma::accu(killEmptyOutput != allowEmptyOutput), 1);
    BOOST_REQUIRE_GT(arma::accu(killEmptyOutput != normalOutput), 1);
  }
  BOOST_REQUIRE_GT(arma::accu(normalOutput != allowEmptyOutput), 1);
}

/**
 * Checking that that size and dimensionality of Final Input File is correct
 * when flag --in_place is specified
 */
BOOST_AUTO_TEST_CASE(KmClusteringResultSizeCheck)
{
  int c = 2;

  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    BOOST_FAIL("Unable to load train dataset vc2.csv!");

  size_t row = inputData.n_rows;
  size_t col = inputData.n_cols;

  SetInputParam("input", inputData);
  SetInputParam("clusters", c);
  SetInputParam("in_place", true);

  mlpackMain();
  arma::mat processedInput = CLI::GetParam<arma::mat>("output");
  // here input is actually accessed through output
  // due to a little trick in kmeans_main

  BOOST_REQUIRE_EQUAL(processedInput.n_cols, col);
  BOOST_REQUIRE_EQUAL(processedInput.n_rows, row+1);
}

/**
 * Ensuring that absence of Number of Clusters is checked.
 */
BOOST_AUTO_TEST_CASE(KmClustersNotDefined)
{
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    BOOST_FAIL("Unable to load train dataset vc2.csv!");

  SetInputParam("input", std::move(inputData));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Checking that all the algorithms yield same results
 */
BOOST_AUTO_TEST_CASE(AlgorithmsSimilarTest)
{
  int c = 5;
  arma::mat inputData(10, 1000);
  inputData.randu();

  arma::mat initCentroids(10, 5);
  initCentroids.randu();

  arma::mat initCentroid = arma::randu<arma::mat>(inputData.n_rows, c);
  std::string algo = "naive";

  SetInputParam("input", inputData);
  SetInputParam("clusters", c);
  SetInputParam("algorithm", std::move(algo));
  SetInputParam("labels_only", true);
  SetInputParam("initial_centroids", initCentroid);

  mlpackMain();

  arma::mat naiveOutput;
  arma::mat naiveCentroid;
  naiveOutput = std::move(CLI::GetParam<arma::mat>("output"));
  naiveCentroid = std::move(CLI::GetParam<arma::mat>("centroid"));

  ResetKmSettings();

  algo = "elkan";

  SetInputParam("input", inputData);
  SetInputParam("clusters", c);
  SetInputParam("algorithm", std::move(algo));
  SetInputParam("labels_only", true);
  SetInputParam("initial_centroids", initCentroid);

  mlpackMain();

  arma::mat elkanOutput;
  arma::mat elkanCentroid;
  elkanOutput = std::move(CLI::GetParam<arma::mat>("output"));
  elkanCentroid = std::move(CLI::GetParam<arma::mat>("centroid"));

  ResetKmSettings();

  algo = "hamerly";

  SetInputParam("input", inputData);
  SetInputParam("clusters", c);
  SetInputParam("algorithm", std::move(algo));
  SetInputParam("labels_only", true);
  SetInputParam("initial_centroids", initCentroid);

  mlpackMain();

  arma::mat hamerlyOutput;
  arma::mat hamerlyCentroid;
  hamerlyOutput = std::move(CLI::GetParam<arma::mat>("output"));
  hamerlyCentroid = std::move(CLI::GetParam<arma::mat>("centroid"));

  ResetKmSettings();

  algo = "dualtree";

  SetInputParam("input", inputData);
  SetInputParam("clusters", c);
  SetInputParam("algorithm", std::move(algo));
  SetInputParam("labels_only", true);
  SetInputParam("initial_centroids", initCentroid);

  mlpackMain();

  arma::mat dualTreeOutput;
  arma::mat dualTreeCentroid;
  dualTreeOutput = std::move(CLI::GetParam<arma::mat>("output"));
  dualTreeCentroid = std::move(CLI::GetParam<arma::mat>("centroid"));

  ResetKmSettings();

  algo = "dualtree-covertree";

  SetInputParam("input", std::move(inputData));
  SetInputParam("clusters", c);
  SetInputParam("algorithm", std::move(algo));
  SetInputParam("labels_only", true);
  SetInputParam("initial_centroids", std::move(initCentroid));

  mlpackMain();

  arma::mat dualCoverTreeOutput;
  arma::mat dualCoverTreeCentroid;
  dualCoverTreeOutput = std::move(CLI::GetParam<arma::mat>("output"));
  dualCoverTreeCentroid = std::move(CLI::GetParam<arma::mat>("centroid"));

  // Checking all the algorithms return same assignments
  CheckMatrices(naiveOutput, hamerlyOutput);
  CheckMatrices(naiveOutput, elkanOutput);
  CheckMatrices(naiveOutput, dualTreeOutput);
  CheckMatrices(naiveOutput, dualCoverTreeOutput);

  // Checking all the algorithms return almost same centroid
  CheckMatrices(naiveCentroid, hamerlyCentroid);
  CheckMatrices(naiveCentroid, elkanCentroid);
  CheckMatrices(naiveCentroid, dualTreeCentroid);
  CheckMatrices(naiveCentroid, dualCoverTreeCentroid);
}

BOOST_AUTO_TEST_SUITE_END();
