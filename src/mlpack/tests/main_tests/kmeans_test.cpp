/**
 * @file kmeans_test.cpp
 * @author Prabhat Sharma
 *
 * Test mlpackMain() of kmeans_main.cpp
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
 * Checking that number of Clusters is less than number of points to be clustered
 */
BOOST_AUTO_TEST_CASE(PointsLessThanClustersTest)
{
  constexpr int N = 10;
  constexpr int D = 4;

  arma::mat inputData = arma::randu<arma::mat>(N, D);

  SetInputParam("input", std::move(inputData));
  SetInputParam("clusters", (int) 11); // Invalid

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}


/**
 * Checking that percentage is between 0 and 1 when --refined_start is specified
*/
BOOST_AUTO_TEST_CASE(RefinedStartPercentageTest)
{
  int C = 2;
  double P = 2.0;
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    BOOST_FAIL("Unable to load train dataset vc2.csv!");

  SetInputParam("input", std::move(inputData));
  SetInputParam("refined_start", true);
  SetInputParam("clusters", std::move(C));
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
  int C = 2;
  double P = -1.0;
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    BOOST_FAIL("Unable to load train dataset vc2.csv!");

  SetInputParam("input", std::move(inputData));
  SetInputParam("refined_start", true);
  SetInputParam("clusters", std::move(C));
  SetInputParam("percentage", std::move(P));     // Invalid

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}


/**
 * Checking that size and dimensionality of prediction is correct.
 */
    BOOST_AUTO_TEST_CASE(KmClusteringSizeCheck)
{
  int C = 2;
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    BOOST_FAIL("Unable to load train dataset vc2.csv!");

  size_t col = inputData.n_cols;
  size_t row = inputData.n_rows;

  SetInputParam("input", std::move(inputData));
  SetInputParam("clusters", std::move(C));
  SetInputParam("allow_empty_clusters", false);

  mlpackMain();

  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_rows, row+1);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_cols, col);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("centroid").n_rows, row);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("centroid").n_cols, C);
}

/**
 * Checking that size and dimensionality of prediction is correct when --labels_only is specified
 */
BOOST_AUTO_TEST_CASE(KmClusteringSizeCheckLabelOnly)
{
  int C = 2;

  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
      BOOST_FAIL("Unable to load train dataset vc2.csv!");
  size_t col = inputData.n_cols;
  size_t row = inputData.n_rows;

  SetInputParam("input", std::move(inputData));
  SetInputParam("clusters", std::move(C));
  SetInputParam("labels_only", true);
  SetInputParam("allow_empty_clusters", false);


  mlpackMain();

  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_rows, 1);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_cols, col);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("centroid").n_rows, row);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("centroid").n_cols, C);
}


/**
 * Checking that predictions are not same when --allow_empty_clusters or kill_empty_clusters are specified
 */
BOOST_AUTO_TEST_CASE(KmClusteringEmptyClustersCheck)
{
  int C = 400;
  int iterations = 100;

  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
      BOOST_FAIL("Unable to load train dataset vc2.csv!");


  SetInputParam("input", inputData);
  SetInputParam("clusters", C);
  SetInputParam("labels_only", true);
  SetInputParam("max_iterations", iterations);
  SetInputParam("allow_empty_clusters", false);

  mlpackMain();

  arma::mat normalOutput;
  normalOutput = std::move(CLI::GetParam<arma::mat>("output"));

  ResetKmSettings();

  SetInputParam("input", inputData);
  SetInputParam("clusters", C);
  SetInputParam("labels_only", true);
  SetInputParam("allow_empty_clusters", true);
  SetInputParam("max_iterations", iterations);

  mlpackMain();

  arma::mat allowEmptyOutput;
  allowEmptyOutput = std::move(CLI::GetParam<arma::mat>("output"));

  ResetKmSettings();

  SetInputParam("input", inputData);
  SetInputParam("clusters", std::move(C));
  SetInputParam("labels_only", true);
  SetInputParam("kill_empty_clusters", true);
  SetInputParam("max_iterations", std::move(iterations));

  mlpackMain();

  arma::mat killEmptyOutput;
  killEmptyOutput = std::move(CLI::GetParam<arma::mat>("output"));

  ResetKmSettings();

  BOOST_REQUIRE_GT(arma::accu(killEmptyOutput != allowEmptyOutput), 1);
  BOOST_REQUIRE_GT(arma::accu(normalOutput != allowEmptyOutput), 1);
}

/**
 * Ensuring that absence of Input is checked.
 */
BOOST_AUTO_TEST_CASE(KmNoInputData)
{
  int C = 2;

  SetInputParam("clusters", std::move(C));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Checking that that size and dimensionality of Final Input File is correct
 * when flag --in_place is specified
 */
BOOST_AUTO_TEST_CASE(KmClusteringResultSizeCheck)
{
  int C = 2;

  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
      BOOST_FAIL("Unable to load train dataset vc2.csv!");

  size_t row = inputData.n_rows;
  size_t col = inputData.n_cols;

  SetInputParam("input", inputData);
  SetInputParam("clusters", std::move(C));
  SetInputParam("in_place", true);
  SetInputParam("allow_empty_clusters", false);

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
  SetInputParam("allow_empty_clusters", false);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Checking that all the algorithms yield same results
 */
BOOST_AUTO_TEST_CASE(AlgorithmsSimilarTest)
{
  int C = 5;
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
      BOOST_FAIL("Unable to load train dataset vc2.csv!");

  std::string algo = "naive";

  SetInputParam("input", inputData);
  SetInputParam("clusters", C);
  SetInputParam("algorithm", std::move(algo));
  SetInputParam("allow_empty_clusters", false);
  SetInputParam("seed", 1);

  mlpackMain();

  arma::mat naiveOutput;
  arma::mat naiveCentroid;
  naiveOutput = std::move(CLI::GetParam<arma::mat>("output"));
  naiveCentroid = std::move(CLI::GetParam<arma::mat>("centroid"));

  ResetKmSettings();

  algo = "elkan";

  SetInputParam("input", inputData);
  SetInputParam("clusters", C);
  SetInputParam("algorithm", std::move(algo));
  SetInputParam("allow_empty_clusters", false);
  SetInputParam("seed", 1);

  mlpackMain();

  arma::mat elkanOutput;
  arma::mat elkanCentroid;
  elkanOutput = std::move(CLI::GetParam<arma::mat>("output"));
  elkanCentroid = std::move(CLI::GetParam<arma::mat>("centroid"));

  ResetKmSettings();

  algo = "hamerly";

  SetInputParam("input", inputData);
  SetInputParam("clusters", C);
  SetInputParam("algorithm", std::move(algo));
  SetInputParam("allow_empty_clusters", false);
  SetInputParam("seed", 1);

  mlpackMain();

  arma::mat hamerlyOutput;
  arma::mat hamerlyCentroid;
  hamerlyOutput = std::move(CLI::GetParam<arma::mat>("output"));
  hamerlyCentroid = std::move(CLI::GetParam<arma::mat>("centroid"));

  ResetKmSettings();

  algo = "dualtree";

  SetInputParam("input", inputData);
  SetInputParam("clusters", C);
  SetInputParam("algorithm", std::move(algo));
  SetInputParam("allow_empty_clusters", false);
  SetInputParam("seed", 1);

  mlpackMain();

  arma::mat dualTreeOutput;
  arma::mat dualTreeCentroid;
  dualTreeOutput = std::move(CLI::GetParam<arma::mat>("output"));
  dualTreeCentroid = std::move(CLI::GetParam<arma::mat>("centroid"));

  ResetKmSettings();

  algo = "dualtree-covertree";

  SetInputParam("input", std::move(inputData));
  SetInputParam("clusters", std::move(C));
  SetInputParam("algorithm", std::move(algo));
  SetInputParam("allow_empty_clusters", false);
  SetInputParam("seed", 1);

  mlpackMain();

  arma::mat dualCoverTreeOutput;
  arma::mat dualCoverTreeCentroid;
  dualCoverTreeOutput = std::move(CLI::GetParam<arma::mat>("output"));
  dualCoverTreeCentroid = std::move(CLI::GetParam<arma::mat>("centroid"));

  // Check That all the algorithms yield the same clusters
  CheckMatrices(naiveOutput, elkanOutput);
  CheckMatrices(elkanOutput, hamerlyOutput);
  CheckMatrices(hamerlyOutput, dualTreeOutput);
  CheckMatrices(dualTreeOutput, dualCoverTreeOutput);

  CheckMatrices(naiveCentroid, elkanCentroid);
  CheckMatrices(elkanCentroid, hamerlyCentroid);
  CheckMatrices(hamerlyCentroid, dualTreeCentroid);
  CheckMatrices(dualTreeCentroid, dualCoverTreeCentroid);
}


BOOST_AUTO_TEST_SUITE_END();
