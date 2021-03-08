/**
 * @file tests/main_tests/kfn_test.cpp
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

#include "../test_catch_tools.hpp"
#include "../catch.hpp"

using namespace mlpack;

struct KFNTestFixture
{
 public:
  KFNTestFixture()
  {
    // Cache in the options for this program.
    IO::RestoreSettings(testName);
  }

  ~KFNTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    IO::ClearSettings();
  }
};

/*
 * Check that we can't provide reference and query matrices
 * with different dimensions.
 */
TEST_CASE_METHOD(KFNTestFixture, "KFNEqualDimensionTest",
                 "[KFNMainTest][BindingTests]")
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
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/*
 * Check that we can't specify an invalid k when only reference
 * matrix is given.
 */
TEST_CASE_METHOD(KFNTestFixture, "KFNInvalidKTest",
                 "[KFNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k > number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 101);

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);

  delete IO::GetParam<KFNModel*>("output_model");
  IO::GetParam<KFNModel*>("output_model") = NULL;

  IO::GetSingleton().Parameters()["reference"].wasPassed = false;
  IO::GetSingleton().Parameters()["k"].wasPassed = false;

  // SetInputParam("reference", referenceData);
  // SetInputParam("k", (int) 0); // Invalid.

  // REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);

  // IO::GetSingleton().Parameters()["reference"].wasPassed = false;
  // IO::GetSingleton().Parameters()["k"].wasPassed = false;

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) -1); // Invalid.

  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/*
 * Check that we can't specify an invalid k when both reference
 * and query matrices are given.
 */
TEST_CASE_METHOD(KFNTestFixture, "KFNInvalidKQueryDataTest",
                 "[KFNMainTest][BindingTests]")
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
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Check that we can't specify a negative leaf size.
 */
TEST_CASE_METHOD(KFNTestFixture, "KFNLeafSizeTest",
                 "[KFNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, negative leaf size.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("leaf_size", (int) -1); // Invalid.

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/*
 * Check that we can't pass both input_model and reference matrix.
 */
TEST_CASE_METHOD(KFNTestFixture, "KFNRefModelTest",
                 "[KFNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);

  mlpackMain();

  // Input pre-trained model.
  SetInputParam("input_model",
      std::move(IO::GetParam<KFNModel*>("output_model")));

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/*
 * Check that we can't pass an invalid tree type.
 */
TEST_CASE_METHOD(KFNTestFixture, "KFNInvalidTreeTypeTest",
                 "[KFNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);
  SetInputParam("tree_type", (string) "min-rp"); // Invalid.

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/*
 * Check that we can't pass an invalid algorithm.
 */
TEST_CASE_METHOD(KFNTestFixture, "KFNInvalidAlgoTest",
                 "[KFNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);
  SetInputParam("algorithm", (string) "triple_tree"); // Invalid.

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/*
 * Check that we can't pass an invalid value of epsilon.
 */
TEST_CASE_METHOD(KFNTestFixture, "KFNInvalidEpsilonTest",
                 "[KFNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 10);
  SetInputParam("epsilon", (double) -1); // Invalid.

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);

  IO::GetSingleton().Parameters()["reference"].wasPassed = false;
  IO::GetSingleton().Parameters()["epsilon"].wasPassed = false;

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("epsilon", (double) 2); // Invalid.

  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);

  IO::GetSingleton().Parameters()["reference"].wasPassed = false;
  IO::GetSingleton().Parameters()["epsilon"].wasPassed = false;

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("epsilon", (double) 1); // Invalid.

  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/*
 * Check that we can't pass an invalid value of percentage.
 */
TEST_CASE_METHOD(KFNTestFixture, "KFNInvalidPercentageTest",
                 "[KFNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 10);
  SetInputParam("percentage", (double) -1); // Invalid.

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);

  IO::GetSingleton().Parameters()["reference"].wasPassed = false;
  IO::GetSingleton().Parameters()["percentage"].wasPassed = false;

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("percentage", (double) 0); // Invalid.

  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);

  IO::GetSingleton().Parameters()["reference"].wasPassed = false;
  IO::GetSingleton().Parameters()["epsilon"].wasPassed = false;

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("percentage", (double) 2); // Invalid.

  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Make sure that dimensions of the neighbors and distances
 * matrices are correct given a value of k.
 */
TEST_CASE_METHOD(KFNTestFixture, "KFNOutputDimensionTest",
                 "[KFNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);

  mlpackMain();

  // Check the neighbors matrix has 4 points for each input point.
  REQUIRE(IO::GetParam<arma::Mat<size_t>>("neighbors").n_rows == 10);
  REQUIRE(IO::GetParam<arma::Mat<size_t>>("neighbors").n_cols == 100);

  // Check the distances matrix has 4 points for each input point.
  REQUIRE(IO::GetParam<arma::mat>("distances").n_rows == 10);
  REQUIRE(IO::GetParam<arma::mat>("distances").n_cols == 100);
}

/**
 * Ensure that saved model can be used again.
 */
TEST_CASE_METHOD(KFNTestFixture, "KFNModelReuseTest",
                 "[KFNMainTest][BindingTests]")
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
  neighbors = std::move(IO::GetParam<arma::Mat<size_t>>("neighbors"));
  distances = std::move(IO::GetParam<arma::mat>("distances"));

  // bindings::tests::CleanMemory();

  // Reset passed parameters.
  IO::GetSingleton().Parameters()["reference"].wasPassed = false;
  IO::GetSingleton().Parameters()["query"].wasPassed = false;

  // Input saved model, pass the same query and keep k unchanged.
  SetInputParam("input_model",
      std::move(IO::GetParam<KFNModel*>("output_model")));
  SetInputParam("query", queryData);

  mlpackMain();

  // Check that initial output matrices and the output matrices using
  // saved model are equal.
  CheckMatrices(neighbors, IO::GetParam<arma::Mat<size_t>>("neighbors"));
  CheckMatrices(distances, IO::GetParam<arma::mat>("distances"));
}

/*
 * Ensure that changing the value of epsilon gives us different
 * approximate KFN results.
 */
TEST_CASE_METHOD(KFNTestFixture, "KFNDifferentEpsilonTest",
                 "[KFNMainTest][BindingTests]")
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
  neighbors = std::move(IO::GetParam<arma::Mat<size_t>>("neighbors"));
  distances = std::move(IO::GetParam<arma::mat>("distances"));

  bindings::tests::CleanMemory();

  IO::GetSingleton().Parameters()["reference"].wasPassed = false;
  IO::GetSingleton().Parameters()["epsilon"].wasPassed = false;

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("epsilon", (double) 0.8);

  mlpackMain();

  CheckMatricesNotEqual(neighbors,
      IO::GetParam<arma::Mat<size_t>>("neighbors"));
  CheckMatricesNotEqual(distances,
      IO::GetParam<arma::mat>("distances"));
}

/*
 * Ensure that changing the value of percentage gives us different
 * approximate KFN results.
 */
TEST_CASE_METHOD(KFNTestFixture, "KFNDifferentPercentageTest",
                 "[KFNMainTest][BindingTests]")
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
  neighbors = std::move(IO::GetParam<arma::Mat<size_t>>("neighbors"));
  distances = std::move(IO::GetParam<arma::mat>("distances"));

  bindings::tests::CleanMemory();

  IO::GetSingleton().Parameters()["reference"].wasPassed = false;
  IO::GetSingleton().Parameters()["percentage"].wasPassed = false;

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("percentage", (double) 0.8);

  mlpackMain();

  CheckMatricesNotEqual(neighbors,
      IO::GetParam<arma::Mat<size_t>>("neighbors"));
  CheckMatricesNotEqual(distances,
      IO::GetParam<arma::mat>("distances"));
}

/*
 * Ensure that we get different results on running twice in greedy
 * search mode when random_basis is specified.
 */
TEST_CASE_METHOD(KFNTestFixture, "KFNRandomBasisTest",
                 "[KFNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(3, 1000); // 1000 points in 3 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 10);
  IO::SetPassed("random_basis");

  mlpackMain();

  arma::Mat<size_t> neighbors;
  arma::mat distances;
  neighbors = std::move(IO::GetParam<arma::Mat<size_t>>("neighbors"));
  distances = std::move(IO::GetParam<arma::mat>("distances"));
  REQUIRE(IO::GetParam<KFNModel*>("output_model")->RandomBasis() == true);

  bindings::tests::CleanMemory();

  IO::GetSingleton().Parameters()["reference"].wasPassed = false;
  IO::GetSingleton().Parameters()["random_basis"].wasPassed = false;

  SetInputParam("reference", std::move(referenceData));

  mlpackMain();

  CheckMatrices(neighbors, IO::GetParam<arma::Mat<size_t>>("neighbors"));
  CheckMatrices(distances, IO::GetParam<arma::mat>("distances"));
  REQUIRE(IO::GetParam<KFNModel*>("output_model")->RandomBasis() == false);
}

/*
 * Ensure that the program runs successfully when we pass true_neighbors
 * and/or true_distances and fails when those matrices have the wrong shape.
 */
TEST_CASE_METHOD(KFNTestFixture, "KFNTrueNeighborDistanceTest",
                 "[KFNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 10);

  mlpackMain();

  arma::Mat<size_t> neighbors;
  arma::mat distances;
  neighbors = std::move(IO::GetParam<arma::Mat<size_t>>("neighbors"));
  distances = std::move(IO::GetParam<arma::mat>("distances"));

  delete IO::GetParam<KFNModel*>("output_model");
  IO::GetParam<KFNModel*>("output_model") = NULL;

  SetInputParam("reference", referenceData);
  SetInputParam("true_neighbors", neighbors);
  SetInputParam("true_distances", distances);
  SetInputParam("epsilon", (double) 0.5);

  REQUIRE_NOTHROW(mlpackMain());

  // True output matrices have incorrect shape.
  arma::Mat<size_t> dummyNeighbors;
  arma::mat dummyDistances;
  dummyNeighbors.randu(20, 100);
  dummyDistances.randu(20, 100);

  delete IO::GetParam<KFNModel*>("output_model");
  IO::GetParam<KFNModel*>("output_model") = NULL;

  IO::GetSingleton().Parameters()["reference"].wasPassed = false;
  IO::GetSingleton().Parameters()["true_neighbors"].wasPassed = false;
  IO::GetSingleton().Parameters()["true_distances"].wasPassed = false;

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("true_neighbors", std::move(dummyNeighbors));
  SetInputParam("true_distances", std::move(dummyDistances));

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/*
 * Ensure that different search algorithms give same result.
 * We do not consider greedy because it is an approximate algorithm.
 */
TEST_CASE_METHOD(KFNTestFixture, "KFNAllAlgorithmsTest",
                 "[KFNMainTest][BindingTests]")
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
  for (int i = 0; i < nofalgorithms; ++i)
  {
    // Same random inputs, different algorithms.
    SetInputParam("reference", referenceData);
    SetInputParam("query", queryData);
    SetInputParam("algorithm", algorithms[i]);

    mlpackMain();

    if (i == 0)
    {
      neighborsCompare = std::move
          (IO::GetParam<arma::Mat<size_t>>("neighbors"));
      distancesCompare = std::move(IO::GetParam<arma::mat>("distances"));
    }
    else
    {
      neighbors = std::move(IO::GetParam<arma::Mat<size_t>>("neighbors"));
      distances = std::move(IO::GetParam<arma::mat>("distances"));

      CheckMatrices(neighborsCompare, neighbors);
      CheckMatrices(distancesCompare, distances);
    }

    delete IO::GetParam<KFNModel*>("output_model");
    IO::GetParam<KFNModel*>("output_model") = NULL;

    // Reset passed parameters.
    IO::GetSingleton().Parameters()["reference"].wasPassed = false;
    IO::GetSingleton().Parameters()["query"].wasPassed = false;
    IO::GetSingleton().Parameters()["algorithm"].wasPassed = false;
  }
}

/*
 * Ensure that different tree types give same result.
 */
TEST_CASE_METHOD(KFNTestFixture, "KFNAllTreeTypesTest",
                 "[KFNMainTest][BindingTests]")
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
  for (int i = 0; i < noftreetypes; ++i)
  {
    // Same random inputs, different algorithms.
    SetInputParam("reference", referenceData);
    SetInputParam("query", queryData);
    SetInputParam("tree_type", treetypes[i]);

    mlpackMain();

    if (i == 0)
    {
      neighborsCompare = std::move(
          IO::GetParam<arma::Mat<size_t>>("neighbors"));
      distancesCompare = std::move(IO::GetParam<arma::mat>("distances"));
    }
    else
    {
      neighbors = std::move(IO::GetParam<arma::Mat<size_t>>("neighbors"));
      distances = std::move(IO::GetParam<arma::mat>("distances"));

      CheckMatrices(neighborsCompare, neighbors);
      CheckMatrices(distancesCompare, distances);
    }

    delete IO::GetParam<KFNModel*>("output_model");
    IO::GetParam<KFNModel*>("output_model") = NULL;

    // Reset passed parameters.
    IO::GetSingleton().Parameters()["reference"].wasPassed = false;
    IO::GetSingleton().Parameters()["query"].wasPassed = false;
    IO::GetSingleton().Parameters()["tree_type"].wasPassed = false;
  }
}

/**
  * Ensure that different leaf sizes give different results.
 */
TEST_CASE_METHOD(KFNTestFixture, "KFNDifferentLeafSizes",
                 "[KFNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 10);
  SetInputParam("leaf_size", (int) 1);

  mlpackMain();

  REQUIRE(IO::GetParam<KFNModel*>("output_model")->LeafSize() == (int) 1);

  bindings::tests::CleanMemory();

  // Reset passed parameters.
  IO::GetSingleton().Parameters()["reference"].wasPassed = false;

  // Input saved model, pass the same query and keep k unchanged.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);
  SetInputParam("leaf_size", (int) 10);

  mlpackMain();

  // Check that initial output matrices and the output matrices using
  // saved model are equal.
  REQUIRE(IO::GetParam<KFNModel*>("output_model")->LeafSize() == (int) 10);
}
