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
BOOST_AUTO_TEST_CAES(KNNEqualDimensionTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Now we specify an invalid dimension(2) for the query data.
  // Note that the number of points  in query and referenceDatae matrices
  // are allowed to be different
  arma::mat queryData;
  queryData.randu(2, 90); // 90 points in 2 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("query", std::move(queryData));
  SetInputParam("k", (int) 10);

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
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k > number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 101);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);

  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  CLI::GetSingleton().Parameters()["k"].wasPassed = false;

  // SetInputParam("reference", referenceData);
  // SetInputParam("k", (int) 0); // Invalid. // Gives memory access violation

  // BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);

  // CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  // CLI::GetSingleton().Parmaeters()["k"].wasPassed = false;

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) -1); // Invalid.

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
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  arma::mat queryData;
  queryData.randu(3, 90); // 90 points in 3 dimensions.

  // Random input, some k > number of reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("query", std::move(queryData));
  SetInputParam("k", (int) 101);

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
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

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
{`
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);

  mlpackMain();

  // Input pre-trained model.
  SetInputParam("input_model",
      std::move(CLI::GetParam<KNNModel*>("output_model")));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/*
 * Check that we can't pass an invalid tree type.
 */
BOOST_AUTO_TEST_CASE(KNNInvalidTreeTypeTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);
  SetInputParam("tree_type", (string) "min-rp"); // Invalid.

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/*
 * Check that we can't pass an invalid algorithm.
 */
BOOST_AUTO_TEST_CASE(KNNInvalidAlgoTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);
  SetInputParam("algorithm", (string) "triple_tree"); // Invalid.

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/*
 * Check that we can't pass an invalid value of epsilon.
 */
BOOST_AUTO_TEST_CASE(KNNInvalidEpsilonTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);
  SetInputParam("epsilon", (double) -1); // Invalid.

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/*
 * Check that we can't pass an invalid value of tau.
 */
BOOST_AUTO_TEST_CASE(KNNInvalidTauTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);
  SetInputParam("tau", (double) -1); // Invalid.

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/*
 * Check that we can't pass an invalid value of rho.
 */
BOOST_AUTO_TEST_CASE(KNNInvalidRhoTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // some k <= number of reference points.
  SetInputParam("k", (int) 10);

  // Random input.
  SetInputParam("reference", referenceData);
  SetInputParam("rho", (double) -1); // Invalid.

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);

  // Reset passed parameters.
  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  CLI::GetSingleton().Parameters()["rho"].wasPassed = false;

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("rho", (double) 1.5); // Invalid.

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
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);

  mlpackMain();

  // Check the neighbors matrix has 10 points for each input point.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Mat<size_t>>
      ("neighbors").n_rows, 10);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Mat<size_t>>
      ("neighbors").n_cols, 100);

  // Check the distances matrix has 10 points for each input point.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("distances").n_rows, 10);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("distances").n_cols, 100);
}

/**
 * Ensure that saved model can be used again.
 */
BOOST_AUTO_TEST_CASE(KNNModelReuseTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  arma::mat queryData;
  queryData.randu(3, 90); // 90 points in 3 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("query", queryData);
  SetInputParam("k", (int) 10);

  mlpackMain();

  arma::Mat<size_t> neighbors;
  arma::mat distances;
  KNNModel* output_model;
  neighbors = std::move(CLI::GetParam<arma::Mat<size_t>>("neighbors"));
  distances = std::move(CLI::GetParam<arma::mat>("distances"));
  output_model = std::move(CLI::GetParam<KNNModel*>("output_model"));

  // Reset passed parameters.
  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  CLI::GetSingleton().Parameters()["query"].wasPassed = false;

  // Input saved model, pass the same query and keep k unchanged.
  SetInputParam("input_model", output_model);
  SetInputParam("query", queryData);

  mlpackMain();

  // Check that initial output matrices and the output matrices using
  // saved model are equal.
  CheckMatrices(neighbors, CLI::GetParam<arma::Mat<size_t>>("neighbors"));
  CheckMatrices(distances, CLI::GetParam<arma::mat>("distances"));
}

/*
 * Ensure that changing the value of tau gives us different greedy
 * spill tree results.
 */
BOOST_AUTO_TEST_CASE(KNNDifferentTauTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 1000); // 1000 points in 3 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 10);
  SetInputParam("tree_type", (string) "spill");
  SetInputParam("tau", (double) 0.2);
  SetInputParam("algorithm", (string) "greedy");

  mlpackMain();

  arma::Mat<size_t> neighbors;
  arma::mat distances;
  neighbors = std::move(CLI::GetParam<arma::Mat<size_t>>("neighbors"));
  distances = std::move(CLI::GetParam<arma::mat>("distances"));

  bindings::tests::CleanMemory();

  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  CLI::GetSingleton().Parameters()["tau"].wasPassed = false;

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("tau", (double) 0.8);

  mlpackMain();

  CheckMatricesNotEqual(neighbors,
      CLI::GetParam<arma::Mat<size_t>>("neighbors"));
  CheckMatricesNotEqual(distances,
      CLI::GetParam<arma::mat>("distances"));
}

/*
 * Ensure that changing the value of rho gives us different greedy
 * spill tree results.
 */
BOOST_AUTO_TEST_CASE(KNNDifferentRhoTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 1000); // 1000 points in 3 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 10);
  SetInputParam("tree_type", (string) "spill");
  SetInputParam("tau", (double) 0.3);
  SetInputParam("rho", (double) 0.2);
  SetInputParam("algorithm", (string) "greedy");

  mlpackMain();

  arma::Mat<size_t> neighbors;
  arma::mat distances;
  neighbors = std::move(CLI::GetParam<arma::Mat<size_t>>("neighbors"));
  distances = std::move(CLI::GetParam<arma::mat>("distances"));

  bindings::tests::CleanMemory();

  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  CLI::GetSingleton().Parameters()["rho"].wasPassed = false;

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("rho", (double) 0.8);

  mlpackMain();

  CheckMatricesNotEqual(neighbors,
      CLI::GetParam<arma::Mat<size_t>>("neighbors"));
  CheckMatricesNotEqual(distances,
      CLI::GetParam<arma::mat>("distances"));
}

/*
 * Ensure that changing the value of epslion gives us different
 * approximate KNN results.
 */
BOOST_AUTO_TEST_CASE(KNNDifferentEpsilonTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 1000); // 1000 points in 3 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 10);
  SetInputParam("epsilon", (double) 0.2);

  mlpackMain();

  arma::Mat<size_t> neighbors;
  arma::mat distances;
  neighbors = std::move(CLI::GetParam<arma::Mat<size_t>>("neighbors"));
  distances = std::move(CLI::GetParam<arma::mat>("distances"));

  bindings::tests::CleanMemory();

  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  CLI::GetSingleton().Parameters()["epsilon"].wasPassed = false;

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("epsilon", (double) 0.8);

  mlpackMain();

  CheckMatricesNotEqual(neighbors,
      CLI::GetParam<arma::Mat<size_t>>("neighbors"));
  CheckMatricesNotEqual(distances,
      CLI::GetParam<arma::mat>("distances"));
}

/*
 * Ensure that we get different results on running twice in greedy
 * search mode when random_basis is specified.
 */
BOOST_AUTO_TEST_CASE(KNNRandomBasisTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 1000); // 1000 points in 3 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 10);
  SetInputParam("algorithm", (string) "greedy");
  CLI::SetPassed("random_basis");

  mlpackMain();

  arma::Mat<size_t> neighbors;
  arma::mat distances;
  neighbors = std::move(CLI::GetParam<arma::Mat<size_t>>("neighbors"));
  distances = std::move(CLI::GetParam<arma::mat>("distances"));

  bindings::tests::CleanMemory();

  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;

  SetInputParam("reference", std::move(referenceData));

  mlpackMain();

  CheckMatricesNotEqual(neighbors,
      CLI::GetParam<arma::Mat<size_t>>("neighbors"));
  CheckMatricesNotEqual(distances,
      CLI::GetParam<arma::mat>("distances"));
}

/*
 * Ensure that the program runs successfully when we pass true_neighbors
 * and/or true_distances and fails when those matrices have the wrong shape.
 */
BOOST_AUTO_TEST_CASE(KNNTrueNeighborDistanceTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 10);

  mlpackMain();

  arma::Mat<size_t> neighbors;
  arma::mat distances;
  neighbors = std::move(CLI::GetParam<arma::Mat<size_t>>("neighbors"));
  distances = std::move(CLI::GetParam<arma::mat>("distances"));

  bindings::tests::CleanMemory();

  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;

  SetInputParam("reference", referenceData);
  SetInputParam("true_neighbors", neighbors);
  SetInputParam("true_distances", distances);
  SetInputParam("epsilon", (double) 0.5);

  BOOST_REQUIRE_NO_THROW(mlpackMain());

  // True output matrices have incorrect shape.
  arma::Mat<size_t> dummy_neighbors;
  arma::mat dummy_distances;
  dummy_neighbors.randu(100, 20);
  dummy_distances.randu(100, 20);

  // bindings::tests::CleanMemory();

  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  CLI::GetSingleton().Parameters()["true_neighbors"].wasPassed = false;
  CLI::GetSingleton().Parameters()["true_distances"].wasPassed = false;

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("true_neighbors", std::move(dummy_neighbors));
  SetInputParam("true_distances", std::move(dummy_distances));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/*
 * Ensure that different search algorithms give same result.
 * We do not consider greedy because it is an approximate algorithm.
 */
BOOST_AUTO_TEST_CASE(KNNAllAlgorithmsTest)
{
  string algorithms[] = {"dual_tree", "naive", "single_tree"};
  const int nofalgorithms = 3;

  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  arma::mat queryData;
  queryData.randu(3, 90); // 90 points in 3 dimensions.

  // Keep some k <= number of reference points same over all.
  SetInputParam("k", (int) 10);

  arma::Mat<size_t> neighbors_compare;
  arma::mat distances_compare;

  arma::Mat<size_t> neighbors;
  arma::mat distances;

  // Looping over all the algorithms and storing their outputs.
  for (int i = 0; i < nofalgorithms; i++)
  {
    // Same random inputs, different algorithms.
    SetInputParam("reference", referenceData);
    SetInputParam("query", queryData);
    SetInputParam("algorithm", algorithms[i]);

    mlpackMain();

    if (i == 0)
    {
      neighbors_compare = std::move(CLI::GetParam<arma::Mat<size_t>>
        ("neighbors"));
      distances_compare = std::move(CLI::GetParam<arma::mat>("distances"));
    }
    else
    {
      neighbors = std::move(CLI::GetParam<arma::Mat<size_t>>("neighbors"));
      distances = std::move(CLI::GetParam<arma::mat>("distances"));

      CheckMatrices(neighbors_compare, neighbors);
      CheckMatrices(distances_compare, distances);
    }

    // Reset passed parameters.
    CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
    CLI::GetSingleton().Parameters()["query"].wasPassed = false;
    CLI::GetSingleton().Parameters()["algorithm"].wasPassed = false;
  }
}

/*
 * Ensure that different tree types give same result.
 */
BOOST_AUTO_TEST_CASE(KNNAllTreeTypesTest)
{
  // Not including spill for now.
  string treetypes[] = {"kd", "vp", "rp", "max-rp", "ub", "cover", "r",
      "r-star", "x", "ball", "hilbert-r", "r-plus", "r-plus-plus",
      "oct"};
  const int noftreetypes = 14; // 15 including spill.

  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  arma::mat queryData;
  queryData.randu(3, 90); // 90 points in 3 dimensions.

  // Keep some k <= number of reference points same over all.
  SetInputParam("k", (int) 10);

  arma::Mat<size_t> neighbors_compare;
  arma::mat distances_compare;

  arma::Mat<size_t> neighbors;
  arma::mat distances;

  // Looping over all the algorithms and storing their outputs.
  for (int i = 0; i < noftreetypes; i++)
  {
    // Same random inputs, different algorithms.
    SetInputParam("reference", referenceData);
    SetInputParam("query", queryData);
    SetInputParam("tree_type", treetypes[i]);

    mlpackMain();

    if (i == 0)
    {
      neighbors_compare = std::move(CLI::GetParam<arma::Mat<size_t>>
        ("neighbors"));
      distances_compare = std::move(CLI::GetParam<arma::mat>("distances"));
    }
    else
    {
      neighbors = std::move(CLI::GetParam<arma::Mat<size_t>>("neighbors"));
      distances = std::move(CLI::GetParam<arma::mat>("distances"));

      CheckMatrices(neighbors_compare, neighbors);
      CheckMatrices(distances_compare, distances);
    }

    // Reset passed parameters.
    CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
    CLI::GetSingleton().Parameters()["query"].wasPassed = false;
    CLI::GetSingleton().Parameters()["tree_type"].wasPassed = false;
  }
}

BOOST_AUTO_TEST_SUITE_END();
