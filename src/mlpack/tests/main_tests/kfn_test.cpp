/**
 * @file kfn_test.cpp
 * @author Atharva Khandait
 * @author Heet Sankesara
 *
 * Test mlpackMain() of kfn_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <string>

#define BINDING_TYPE BINDING_TYPE_TEST
static const std::string testName = "K-FurthestNeighborsSearch";

#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "test_helper.hpp"
#include <mlpack/methods/neighbor_search/kfn_main.cpp>

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

struct KFNTestFixture
{
 public:
  KFNTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~KFNTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(KFNMainTest, KFNTestFixture);

/*
 * Check that we can't provide reference and query matrices
 * with different dimensions.
 */
BOOST_AUTO_TEST_CASE(KFNEqualDimensionTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Now we specify an invalid dimension(2) for the query data.
  // Note that the number of points in query and reference matrices
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
BOOST_AUTO_TEST_CASE(KFNInvalidKTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k > number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 101);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);

  delete CLI::GetParam<KFNModel*>("output_model");
  CLI::GetParam<KFNModel*>("output_model") = NULL;

  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  CLI::GetSingleton().Parameters()["k"].wasPassed = false;

  // SetInputParam("reference", referenceData);
  // SetInputParam("k", (int) 0); // Invalid.

  // BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);

  // CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  // CLI::GetSingleton().Parameters()["k"].wasPassed = false;

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) -1); // Invalid.

  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/*
 * Check that we can't specify an invalid k when both reference
 * and query matrices are given.
 */
BOOST_AUTO_TEST_CASE(KFNInvalidKQueryDataTest)
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
BOOST_AUTO_TEST_CASE(KFNLeafSizeTest)
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
BOOST_AUTO_TEST_CASE(KFNRefModelTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);

  mlpackMain();

  // Input pre-trained model.
  SetInputParam("input_model",
      std::move(CLI::GetParam<KFNModel*>("output_model")));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/*
 * Check that we can't pass an invalid tree type.
 */
BOOST_AUTO_TEST_CASE(KFNInvalidTreeTypeTest)
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
BOOST_AUTO_TEST_CASE(KFNInvalidAlgoTest)
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
BOOST_AUTO_TEST_CASE(KFNInvalidEpsilonTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 10);
  SetInputParam("epsilon", (double) -1); // Invalid.

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);

  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  CLI::GetSingleton().Parameters()["epsilon"].wasPassed = false;

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("epsilon", (double) 2); // Invalid.

  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);

  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  CLI::GetSingleton().Parameters()["epsilon"].wasPassed = false;

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("epsilon", (double) 1); // Invalid.

  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/*
 * Check that we can't pass an invalid value of percentage.
 */
BOOST_AUTO_TEST_CASE(KFNInvalidPercentageTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 10);
  SetInputParam("percentage", (double) -1); // Invalid.

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);

  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  CLI::GetSingleton().Parameters()["percentage"].wasPassed = false;

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("percentage", (double) 0); // Invalid.

  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);

  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  CLI::GetSingleton().Parameters()["epsilon"].wasPassed = false;

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("percentage", (double) 2); // Invalid.

  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Make sure that dimensions of the neighbors and distances
 * matrices are correct given a value of k.
 */
BOOST_AUTO_TEST_CASE(KFNOutputDimensionTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);

  mlpackMain();

  // Check the neighbors matrix has 4 points for each input point.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Mat<size_t>>
      ("neighbors").n_rows, 10);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Mat<size_t>>
      ("neighbors").n_cols, 100);

  // Check the distances matrix has 4 points for each input point.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("distances").n_rows, 10);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("distances").n_cols, 100);
}

/**
 * Ensure that saved model can be used again.
 */
BOOST_AUTO_TEST_CASE(KFNModelReuseTest)
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
  neighbors = std::move(CLI::GetParam<arma::Mat<size_t>>("neighbors"));
  distances = std::move(CLI::GetParam<arma::mat>("distances"));

  // bindings::tests::CleanMemory();

  // Reset passed parameters.
  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  CLI::GetSingleton().Parameters()["query"].wasPassed = false;

  // Input saved model, pass the same query and keep k unchanged.
  SetInputParam("input_model",
      std::move(CLI::GetParam<KFNModel*>("output_model")));
  SetInputParam("query", queryData);

  mlpackMain();

  // Check that initial output matrices and the output matrices using
  // saved model are equal.
  CheckMatrices(neighbors, CLI::GetParam<arma::Mat<size_t>>("neighbors"));
  CheckMatrices(distances, CLI::GetParam<arma::mat>("distances"));
}

/*
 * Ensure that changing the value of epsilon gives us different
 * approximate KFN results.
 */
BOOST_AUTO_TEST_CASE(KFNDifferentEpsilonTest)
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
 * Ensure that changing the value of percentage gives us different
 * approximate KFN results.
 */
BOOST_AUTO_TEST_CASE(KFNDifferentPercentageTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 1000); // 1000 points in 3 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 10);
  SetInputParam("percentage", (double) 0.2);

  mlpackMain();

  arma::Mat<size_t> neighbors;
  arma::mat distances;
  neighbors = std::move(CLI::GetParam<arma::Mat<size_t>>("neighbors"));
  distances = std::move(CLI::GetParam<arma::mat>("distances"));

  bindings::tests::CleanMemory();

  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  CLI::GetSingleton().Parameters()["percentage"].wasPassed = false;

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("percentage", (double) 0.8);

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
BOOST_AUTO_TEST_CASE(KFNRandomBasisTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 1000); // 1000 points in 3 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 10);
  CLI::SetPassed("random_basis");

  mlpackMain();

  arma::Mat<size_t> neighbors;
  arma::mat distances;
  neighbors = std::move(CLI::GetParam<arma::Mat<size_t>>("neighbors"));
  distances = std::move(CLI::GetParam<arma::mat>("distances"));
  BOOST_REQUIRE_EQUAL(CLI::GetParam<KFNModel*>("output_model")->RandomBasis(),
      true);

  bindings::tests::CleanMemory();

  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  CLI::GetSingleton().Parameters()["random_basis"].wasPassed = false;

  SetInputParam("reference", std::move(referenceData));

  mlpackMain();

  CheckMatrices(neighbors, CLI::GetParam<arma::Mat<size_t>>("neighbors"));
  CheckMatrices(distances, CLI::GetParam<arma::mat>("distances"));
  BOOST_REQUIRE_EQUAL(CLI::GetParam<KFNModel*>("output_model")->RandomBasis(),
      false);
}

/*
 * Ensure that the program runs successfully when we pass true_neighbors
 * and/or true_distances and fails when those matrices have the wrong shape.
 */
BOOST_AUTO_TEST_CASE(KFNTrueNeighborDistanceTest)
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

  delete CLI::GetParam<KFNModel*>("output_model");
  CLI::GetParam<KFNModel*>("output_model") = NULL;

  SetInputParam("reference", referenceData);
  SetInputParam("true_neighbors", neighbors);
  SetInputParam("true_distances", distances);
  SetInputParam("epsilon", (double) 0.5);

  BOOST_REQUIRE_NO_THROW(mlpackMain());

  // True output matrices have incorrect shape.
  arma::Mat<size_t> dummyNeighbors;
  arma::mat dummyDistances;
  dummyNeighbors.randu(20, 100);
  dummyDistances.randu(20, 100);

  delete CLI::GetParam<KFNModel*>("output_model");
  CLI::GetParam<KFNModel*>("output_model") = NULL;

  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  CLI::GetSingleton().Parameters()["true_neighbors"].wasPassed = false;
  CLI::GetSingleton().Parameters()["true_distances"].wasPassed = false;

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("true_neighbors", std::move(dummyNeighbors));
  SetInputParam("true_distances", std::move(dummyDistances));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/*
 * Ensure that different search algorithms give same result.
 * We do not consider greedy because it is an approximate algorithm.
 */
BOOST_AUTO_TEST_CASE(KFNAllAlgorithmsTest)
{
  string algorithms[] = {"dual_tree", "naive", "single_tree"};
  const int nofalgorithms = 3;

  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  arma::mat queryData;
  queryData.randu(3, 90); // 90 points in 3 dimensions.

  // Keep some k <= number of reference points same over all.
  SetInputParam("k", (int) 10);

  arma::Mat<size_t> neighborsCompare;
  arma::mat distancesCompare;

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
      neighborsCompare = std::move
          (CLI::GetParam<arma::Mat<size_t>>("neighbors"));
      distancesCompare = std::move(CLI::GetParam<arma::mat>("distances"));
    }
    else
    {
      neighbors = std::move(CLI::GetParam<arma::Mat<size_t>>("neighbors"));
      distances = std::move(CLI::GetParam<arma::mat>("distances"));

      CheckMatrices(neighborsCompare, neighbors);
      CheckMatrices(distancesCompare, distances);
    }

    delete CLI::GetParam<KFNModel*>("output_model");
    CLI::GetParam<KFNModel*>("output_model") = NULL;

    // Reset passed parameters.
    CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
    CLI::GetSingleton().Parameters()["query"].wasPassed = false;
    CLI::GetSingleton().Parameters()["algorithm"].wasPassed = false;
  }
}

/*
 * Ensure that different tree types give same result.
 */
BOOST_AUTO_TEST_CASE(KFNAllTreeTypesTest)
{
  string treetypes[] = {"kd", "vp", "rp", "max-rp", "ub", "cover", "r",
      "r-star", "x", "ball", "hilbert-r", "r-plus", "r-plus-plus",
      "oct"};
  const int noftreetypes = 14;

  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  arma::mat queryData;
  queryData.randu(3, 90); // 90 points in 3 dimensions.

  // Keep some k <= number of reference points same over all.
  SetInputParam("k", (int) 10);

  arma::Mat<size_t> neighborsCompare;
  arma::mat distancesCompare;

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
      neighborsCompare = std::move(
          CLI::GetParam<arma::Mat<size_t>>("neighbors"));
      distancesCompare = std::move(CLI::GetParam<arma::mat>("distances"));
    }
    else
    {
      neighbors = std::move(CLI::GetParam<arma::Mat<size_t>>("neighbors"));
      distances = std::move(CLI::GetParam<arma::mat>("distances"));

      CheckMatrices(neighborsCompare, neighbors);
      CheckMatrices(distancesCompare, distances);
    }

    delete CLI::GetParam<KFNModel*>("output_model");
    CLI::GetParam<KFNModel*>("output_model") = NULL;

    // Reset passed parameters.
    CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
    CLI::GetSingleton().Parameters()["query"].wasPassed = false;
    CLI::GetSingleton().Parameters()["tree_type"].wasPassed = false;
  }
}

/**
  * Ensure that different leaf sizes give different results.
 */
BOOST_AUTO_TEST_CASE(KFNDifferentLeafSizes)
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 10);
  SetInputParam("leaf_size", (int) 1);

  mlpackMain();

  BOOST_CHECK_EQUAL(CLI::GetParam<KFNModel*>("output_model")->LeafSize(),
      (int) 1);

  bindings::tests::CleanMemory();

  // Reset passed parameters.
  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;

  // Input saved model, pass the same query and keep k unchanged.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);
  SetInputParam("leaf_size", (int) 10);

  mlpackMain();

  // Check that initial output matrices and the output matrices using
  // saved model are equal.
  BOOST_CHECK_EQUAL(CLI::GetParam<KFNModel*>("output_model")->LeafSize(),
      (int) 10);
}

BOOST_AUTO_TEST_SUITE_END();
