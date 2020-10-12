/**
 * @file tests/main_tests/knn_test.cpp
 * @author Atharva Khandait
 * @author Heet Sankesara
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

#include "../test_catch_tools.hpp"
#include "../catch.hpp"

using namespace mlpack;

struct KNNTestFixture
{
 public:
  KNNTestFixture()
  {
    // Cache in the options for this program.
    IO::RestoreSettings(testName);
  }

  ~KNNTestFixture()
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
TEST_CASE_METHOD(KNNTestFixture, "KNNEqualDimensionTest",
                 "[KNNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Now we specify an invalid dimension(2) for the query data.
  // Note that the number of points in queryData and referenceData matrices
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
TEST_CASE_METHOD(KNNTestFixture, "KNNInvalidKTest",
                 "[KNNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k > number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 101);

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);

  IO::GetSingleton().Parameters()["reference"].wasPassed = false;
  IO::GetSingleton().Parameters()["k"].wasPassed = false;

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) -1); // Invalid.

  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/*
 * Check that we can't specify an invalid k when both reference
 * and query matrices are given.
 */
TEST_CASE_METHOD(KNNTestFixture, "KNNInvalidKQueryDataTest",
                 "[KNNMainTest][BindingTests]")
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
TEST_CASE_METHOD(KNNTestFixture, "KNNLeafSizeTest",
                 "[KNNMainTest][BindingTests]")
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
TEST_CASE_METHOD(KNNTestFixture, "KNNRefModelTest",
                 "[KNNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);

  mlpackMain();

  // Input pre-trained model.
  SetInputParam("input_model",
      std::move(IO::GetParam<KNNModel*>("output_model")));

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/*
 * Check that we can't pass an invalid tree type.
 */
TEST_CASE_METHOD(KNNTestFixture, "KNNInvalidTreeTypeTest",
                 "[KNNMainTest][BindingTests]")
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
TEST_CASE_METHOD(KNNTestFixture, "KNNInvalidAlgoTest",
                 "[KNNMainTest][BindingTests]")
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
TEST_CASE_METHOD(KNNTestFixture, "KNNInvalidEpsilonTest",
                 "[KNNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);
  SetInputParam("epsilon", (double) -1); // Invalid.

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/*
 * Check that we can't pass an invalid value of tau.
 */
TEST_CASE_METHOD(KNNTestFixture, "KNNInvalidTauTest",
                 "[KNNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);
  SetInputParam("tau", (double) -1); // Invalid.

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/*
 * Check that we can't pass an invalid value of rho.
 */
TEST_CASE_METHOD(KNNTestFixture, "KNNInvalidRhoTest",
                 "[KNNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // some k <= number of reference points.
  SetInputParam("k", (int) 10);

  // Random input.
  SetInputParam("reference", referenceData);
  SetInputParam("rho", (double) -1); // Invalid.

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);

  // Reset passed parameters.
  IO::GetSingleton().Parameters()["reference"].wasPassed = false;
  IO::GetSingleton().Parameters()["rho"].wasPassed = false;

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("rho", (double) 1.5); // Invalid.

  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Make sure that dimensions of the neighbors and distances matrices are correct
 * given a value of k.
 */
TEST_CASE_METHOD(KNNTestFixture, "KNNOutputDimensionTest",
                 "[KNNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);

  mlpackMain();

  // Check the neighbors matrix has 10 points for each input point.
  REQUIRE(IO::GetParam<arma::Mat<size_t>>("neighbors").n_rows == 10);
  REQUIRE(IO::GetParam<arma::Mat<size_t>>("neighbors").n_cols == 100);

  // Check the distances matrix has 10 points for each input point.
  REQUIRE(IO::GetParam<arma::mat>("distances").n_rows == 10);
  REQUIRE(IO::GetParam<arma::mat>("distances").n_cols == 100);
}

/**
 * Ensure that saved model can be used again.
 */
TEST_CASE_METHOD(KNNTestFixture, "KNNModelReuseTest",
                 "[KNNMainTest][BindingTests]")
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
  neighbors = std::move(IO::GetParam<arma::Mat<size_t>>("neighbors"));
  distances = std::move(IO::GetParam<arma::mat>("distances"));
  output_model = std::move(IO::GetParam<KNNModel*>("output_model"));

  // Reset passed parameters.
  IO::GetSingleton().Parameters()["reference"].wasPassed = false;
  IO::GetSingleton().Parameters()["query"].wasPassed = false;

  // Input saved model, pass the same query and keep k unchanged.
  SetInputParam("input_model", output_model);
  SetInputParam("query", queryData);

  mlpackMain();

  // Check that initial output matrices and the output matrices using
  // saved model are equal.
  CheckMatrices(neighbors, IO::GetParam<arma::Mat<size_t>>("neighbors"));
  CheckMatrices(distances, IO::GetParam<arma::mat>("distances"));
}

/*
 * Ensure that changing the value of tau gives us different greedy
 * spill tree results.
 */
TEST_CASE_METHOD(KNNTestFixture, "KNNDifferentTauTest",
                 "[KNNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(6, 1000); // 1000 points in 6 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 4);
  SetInputParam("tree_type", (string) "spill");
  SetInputParam("tau", (double) 0.2);
  SetInputParam("algorithm", (string) "greedy");

  mlpackMain();

  arma::Mat<size_t> neighbors;
  arma::mat distances;
  neighbors = std::move(IO::GetParam<arma::Mat<size_t>>("neighbors"));
  distances = std::move(IO::GetParam<arma::mat>("distances"));

  bindings::tests::CleanMemory();

  IO::GetSingleton().Parameters()["reference"].wasPassed = false;
  IO::GetSingleton().Parameters()["tau"].wasPassed = false;

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("tau", (double) 0.8);

  mlpackMain();

  CheckMatricesNotEqual(neighbors,
      IO::GetParam<arma::Mat<size_t>>("neighbors"));
  CheckMatricesNotEqual(distances,
      IO::GetParam<arma::mat>("distances"));
}

/*
 * Ensure that changing the value of rho gives us different greedy
 * spill tree results.
 */
TEST_CASE_METHOD(KNNTestFixture, "KNNDifferentRhoTest",
                 "[KNNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(3, 1000); // 1000 points in 3 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 10);
  SetInputParam("tree_type", (string) "spill");
  SetInputParam("tau", (double) 0.3);
  SetInputParam("rho", (double) 0.01);
  SetInputParam("algorithm", (string) "greedy");

  mlpackMain();

  arma::Mat<size_t> neighbors;
  arma::mat distances;
  neighbors = std::move(IO::GetParam<arma::Mat<size_t>>("neighbors"));
  distances = std::move(IO::GetParam<arma::mat>("distances"));

  bindings::tests::CleanMemory();

  IO::GetSingleton().Parameters()["reference"].wasPassed = false;
  IO::GetSingleton().Parameters()["rho"].wasPassed = false;

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("rho", (double) 0.99);

  mlpackMain();

  CheckMatricesNotEqual(neighbors,
      IO::GetParam<arma::Mat<size_t>>("neighbors"));
  CheckMatricesNotEqual(distances,
      IO::GetParam<arma::mat>("distances"));
}

/*
 * Ensure that changing the value of epslion gives us different
 * approximate KNN results.
 */
TEST_CASE_METHOD(KNNTestFixture, "KNNDifferentEpsilonTest",
                 "[KNNMainTest][BindingTests]")
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
 * Ensure that we get same results on running twice in dual-tree mode
 * search mode when random_basis is specified.
 */
TEST_CASE_METHOD(KNNTestFixture, "KNNRandomBasisTest",
                 "[KNNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(3, 1000); // 1000 points in 3 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 10);
  SetInputParam("algorithm", (string) "dual_tree");
  IO::SetPassed("random_basis");

  mlpackMain();

  arma::Mat<size_t> neighbors;
  arma::mat distances;
  neighbors = std::move(IO::GetParam<arma::Mat<size_t>>("neighbors"));
  distances = std::move(IO::GetParam<arma::mat>("distances"));
  REQUIRE(IO::GetParam<KNNModel*>("output_model")->RandomBasis() == true);

  bindings::tests::CleanMemory();

  IO::GetSingleton().Parameters()["reference"].wasPassed = false;
  IO::GetSingleton().Parameters()["random_basis"].wasPassed = false;

  SetInputParam("reference", std::move(referenceData));

  mlpackMain();

  CheckMatrices(neighbors, IO::GetParam<arma::Mat<size_t>>("neighbors"));
  CheckMatrices(distances, IO::GetParam<arma::mat>("distances"));
  REQUIRE(IO::GetParam<KNNModel*>("output_model")->RandomBasis() == false);
}

/*
 * Ensure that the program runs successfully when we pass true_neighbors
 * and/or true_distances and fails when those matrices have the wrong shape.
 */
TEST_CASE_METHOD(KNNTestFixture, "KNNTrueNeighborDistanceTest",
                 "[KNNMainTest][BindingTests]")
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

  bindings::tests::CleanMemory();

  IO::GetSingleton().Parameters()["reference"].wasPassed = false;

  SetInputParam("reference", referenceData);
  SetInputParam("true_neighbors", neighbors);
  SetInputParam("true_distances", distances);
  SetInputParam("epsilon", (double) 0.5);

  REQUIRE_NOTHROW(mlpackMain());

  // True output matrices have incorrect shape.
  arma::Mat<size_t> dummyNeighbors;
  arma::mat dummyDistances;
  dummyNeighbors.randu(100, 20);
  dummyDistances.randu(100, 20);

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
TEST_CASE_METHOD(KNNTestFixture, "KNNAllAlgorithmsTest",
                 "[KNNMainTest][BindingTests]")
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

    delete IO::GetParam<KNNModel*>("output_model");
    IO::GetParam<KNNModel*>("output_model") = NULL;

    // Reset passed parameters.
    IO::GetSingleton().Parameters()["reference"].wasPassed = false;
    IO::GetSingleton().Parameters()["query"].wasPassed = false;
    IO::GetSingleton().Parameters()["algorithm"].wasPassed = false;
  }
}

/*
 * Ensure that different tree types give same result.
 */
TEST_CASE_METHOD(KNNTestFixture, "KNNAllTreeTypesTest",
                 "[KNNMainTest][BindingTests]")
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
  SetInputParam("k", (int) 15);

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

    delete IO::GetParam<KNNModel*>("output_model");
    IO::GetParam<KNNModel*>("output_model") = NULL;

    // Reset passed parameters.
    IO::GetSingleton().Parameters()["reference"].wasPassed = false;
    IO::GetSingleton().Parameters()["query"].wasPassed = false;
    IO::GetSingleton().Parameters()["tree_type"].wasPassed = false;
  }
}

/**
  * Ensure that different leaf sizes give different results.
 */
TEST_CASE_METHOD(KNNTestFixture, "KNNDifferentLeafSizes",
                 "[KNNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 10);
  SetInputParam("leaf_size", (int) 1);

  mlpackMain();

  KNNModel* output_model;
  output_model = std::move(IO::GetParam<KNNModel*>("output_model"));

  // Reset passed parameters.
  IO::GetSingleton().Parameters()["reference"].wasPassed = false;

  // Input saved model, pass the same query and keep k unchanged.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);
  SetInputParam("leaf_size", (int) 10);

  mlpackMain();

  // Check that initial output matrices and the output matrices using
  // saved model are equal.
  REQUIRE(output_model->LeafSize() == (int) 1);
  REQUIRE(IO::GetParam<KNNModel*>("output_model")->LeafSize() == (int) 10);
  delete output_model;
}
