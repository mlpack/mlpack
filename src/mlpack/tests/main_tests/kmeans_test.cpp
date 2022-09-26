/**
 * @file tests/main_tests/kmeans_test.cpp
 * @author Prabhat Sharma
 *
 * Test RUN_BINDING() of kmeans_main.cpp
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/kmeans/kmeans_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "main_test_fixture.hpp"

#include "../catch.hpp"
#include "../test_catch_tools.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(KmTestFixture);

/**
 * Checking that number of Clusters are non negative
 */
TEST_CASE_METHOD(KmTestFixture, "NonNegativeClustersTest",
                 "[KmeansMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    FAIL("Unable to load train dataset vc2.csv!");

  SetInputParam("input", std::move(inputData));
  SetInputParam("clusters", (int) -1); // Invalid

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}


/**
 * Checking that initial centroids are provided if clusters are to be auto detected
 */
TEST_CASE_METHOD(KmTestFixture, "AutoDetectClusterTest",
                 "[KmeansMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 4;

  arma::mat inputData = arma::randu<arma::mat>(N, D);

  SetInputParam("input", std::move(inputData));
  SetInputParam("clusters", (int) 0); // Invalid

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}


/**
 * Checking that percentage is between 0 and 1 when --refined_start is specified
*/
TEST_CASE_METHOD(KmTestFixture, "RefinedStartPercentageTest",
                 "[KmeansMainTest][BindingTests]")
{
  int c = 2;
  double P = 2.0;
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    FAIL("Unable to load train dataset vc2.csv!");

  SetInputParam("input", std::move(inputData));
  SetInputParam("refined_start", true);
  SetInputParam("clusters", c);
  SetInputParam("percentage", std::move(P));     // Invalid

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}


/**
 * Checking percentage is non-negative when --refined_start is specified
 */
TEST_CASE_METHOD(KmTestFixture, "NonNegativePercentageTest",
                 "[KmeansMainTest][BindingTests]")
{
  int c = 2;
  double P = -1.0;
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    FAIL("Unable to load train dataset vc2.csv!");

  SetInputParam("input", std::move(inputData));
  SetInputParam("refined_start", true);
  SetInputParam("clusters", c);
  SetInputParam("percentage", P);     // Invalid

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}


/**
 * Checking that size and dimensionality of prediction is correct.
 */
TEST_CASE_METHOD(KmTestFixture, "KmClusteringSizeCheck",
                 "[KmeansMainTest][BindingTests]")
{
  int c = 2;
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    FAIL("Unable to load train dataset vc2.csv!");

  size_t col = inputData.n_cols;
  size_t row = inputData.n_rows;

  SetInputParam("input", std::move(inputData));
  SetInputParam("clusters", c);

  RUN_BINDING();

  REQUIRE(params.Get<arma::mat>("output").n_rows == row+1);
  REQUIRE(params.Get<arma::mat>("output").n_cols == col);
  REQUIRE(params.Get<arma::mat>("centroid").n_rows == row);
  REQUIRE(params.Get<arma::mat>("centroid").n_cols == (arma::uword) c);
}

/**
 * Checking that size and dimensionality of prediction is correct when --labels_only is specified
 */
TEST_CASE_METHOD(KmTestFixture, "KmClusteringSizeCheckLabelOnly",
                 "[KmeansMainTest][BindingTests]")
{
  int c = 2;

  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    FAIL("Unable to load train dataset vc2.csv!");
  size_t col = inputData.n_cols;
  size_t row = inputData.n_rows;

  SetInputParam("input", std::move(inputData));
  SetInputParam("clusters", c);
  SetInputParam("labels_only", true);

  RUN_BINDING();

  REQUIRE(params.Get<arma::mat>("output").n_rows == 1);
  REQUIRE(params.Get<arma::mat>("output").n_cols == col);
  REQUIRE(params.Get<arma::mat>("centroid").n_rows == row);
  REQUIRE(params.Get<arma::mat>("centroid").n_cols == (arma::uword) c);
}


/**
 * Checking that predictions are not same when --allow_empty_clusters or kill_empty_clusters are specified
 */
TEST_CASE_METHOD(KmTestFixture, "KmClusteringEmptyClustersCheck",
                 "[KmeansMainTest][BindingTests]")
{
  int c = 400;
  int iterations = 100;

  arma::mat inputData;
  if (!data::Load("test_data_3_1000.csv", inputData))
    FAIL("Unable to load train dataset test_data_3_1000.csv!");
  arma::mat initCentroid = arma::randu<arma::mat>(inputData.n_rows, c);

  SetInputParam("input", inputData);
  SetInputParam("clusters", c);
  SetInputParam("labels_only", true);
  SetInputParam("max_iterations", iterations);
  SetInputParam("initial_centroids", initCentroid);

  RUN_BINDING();

  arma::mat normalOutput;
  normalOutput = std::move(params.Get<arma::mat>("centroid"));

  CleanMemory();
  ResetSettings();

  SetInputParam("input", inputData);
  SetInputParam("clusters", c);
  SetInputParam("labels_only", true);
  SetInputParam("allow_empty_clusters", true);
  SetInputParam("max_iterations", iterations);
  SetInputParam("initial_centroids", initCentroid);

  RUN_BINDING();

  arma::mat allowEmptyOutput;
  allowEmptyOutput = std::move(params.Get<arma::mat>("centroid"));

  CleanMemory();
  ResetSettings();

  SetInputParam("input", inputData);
  SetInputParam("clusters", c);
  SetInputParam("labels_only", true);
  SetInputParam("kill_empty_clusters", true);
  SetInputParam("max_iterations", iterations);
  SetInputParam("initial_centroids", initCentroid);

  RUN_BINDING();

  arma::mat killEmptyOutput;
  killEmptyOutput = std::move(params.Get<arma::mat>("centroid"));

  CleanMemory();
  ResetSettings();

  if (killEmptyOutput.n_elem == allowEmptyOutput.n_elem)
  {
    REQUIRE(arma::accu(killEmptyOutput != allowEmptyOutput) > 1);
    REQUIRE(arma::accu(killEmptyOutput != normalOutput) > 1);
  }
  REQUIRE(arma::accu(normalOutput != allowEmptyOutput) > 1);
}

/**
 * Checking that that size and dimensionality of Final Input File is correct
 * when flag --in_place is specified
 */
TEST_CASE_METHOD(KmTestFixture, "KmClusteringResultSizeCheck",
                 "[KmeansMainTest][BindingTests]")
{
  int c = 2;

  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    FAIL("Unable to load train dataset vc2.csv!");

  size_t row = inputData.n_rows;
  size_t col = inputData.n_cols;

  SetInputParam("input", inputData);
  SetInputParam("clusters", c);
  SetInputParam("in_place", true);

  RUN_BINDING();
  arma::mat processedInput = params.Get<arma::mat>("output");
  // here input is actually accessed through output
  // due to a little trick in kmeans_main

  REQUIRE(processedInput.n_cols == col);
  REQUIRE(processedInput.n_rows == row+1);
}

/**
 * Ensuring that absence of Number of Clusters is checked.
 */
TEST_CASE_METHOD(KmTestFixture, "KmClustersNotDefined",
                 "[KmeansMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    FAIL("Unable to load train dataset vc2.csv!");

  SetInputParam("input", std::move(inputData));

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Checking that all the algorithms yield same results
 */
TEST_CASE_METHOD(KmTestFixture, "AlgorithmsSimilarTest",
                 "[KmeansMainTest][BindingTests]")
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

  RUN_BINDING();

  arma::mat naiveOutput;
  arma::mat naiveCentroid;
  naiveOutput = std::move(params.Get<arma::mat>("output"));
  naiveCentroid = std::move(params.Get<arma::mat>("centroid"));

  CleanMemory();
  ResetSettings();

  algo = "elkan";

  SetInputParam("input", inputData);
  SetInputParam("clusters", c);
  SetInputParam("algorithm", std::move(algo));
  SetInputParam("labels_only", true);
  SetInputParam("initial_centroids", initCentroid);

  RUN_BINDING();

  arma::mat elkanOutput;
  arma::mat elkanCentroid;
  elkanOutput = std::move(params.Get<arma::mat>("output"));
  elkanCentroid = std::move(params.Get<arma::mat>("centroid"));

  CleanMemory();
  ResetSettings();

  algo = "hamerly";

  SetInputParam("input", inputData);
  SetInputParam("clusters", c);
  SetInputParam("algorithm", std::move(algo));
  SetInputParam("labels_only", true);
  SetInputParam("initial_centroids", initCentroid);

  RUN_BINDING();

  arma::mat hamerlyOutput;
  arma::mat hamerlyCentroid;
  hamerlyOutput = std::move(params.Get<arma::mat>("output"));
  hamerlyCentroid = std::move(params.Get<arma::mat>("centroid"));

  CleanMemory();
  ResetSettings();

  algo = "dualtree";

  SetInputParam("input", inputData);
  SetInputParam("clusters", c);
  SetInputParam("algorithm", std::move(algo));
  SetInputParam("labels_only", true);
  SetInputParam("initial_centroids", initCentroid);

  RUN_BINDING();

  arma::mat dualTreeOutput;
  arma::mat dualTreeCentroid;
  dualTreeOutput = std::move(params.Get<arma::mat>("output"));
  dualTreeCentroid = std::move(params.Get<arma::mat>("centroid"));

  CleanMemory();
  ResetSettings();

  algo = "dualtree-covertree";

  SetInputParam("input", std::move(inputData));
  SetInputParam("clusters", c);
  SetInputParam("algorithm", std::move(algo));
  SetInputParam("labels_only", true);
  SetInputParam("initial_centroids", std::move(initCentroid));

  RUN_BINDING();

  arma::mat dualCoverTreeOutput;
  arma::mat dualCoverTreeCentroid;
  dualCoverTreeOutput = std::move(params.Get<arma::mat>("output"));
  dualCoverTreeCentroid = std::move(params.Get<arma::mat>("centroid"));

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
