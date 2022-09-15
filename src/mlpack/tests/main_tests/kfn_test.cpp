/**
 * @file tests/main_tests/kfn_test.cpp
 * @author Atharva Khandait
 * @author Heet Sankesara
 *
 * Test RUN_BINDING() of kfn_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/neighbor_search/kfn_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "main_test_fixture.hpp"

#include "../test_catch_tools.hpp"
#include "../catch.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(KFNTestFixture);

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

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
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

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);

  delete params.Get<KFNModel*>("output_model");
  params.Get<KFNModel*>("output_model") = NULL;

  ResetSettings();

  ResetSettings();

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) -1); // Invalid.

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
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

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
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

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
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

  RUN_BINDING();

  // Input pre-trained model.
  SetInputParam("input_model",
      std::move(params.Get<KFNModel*>("output_model")));

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
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

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
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

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
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

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);

  ResetSettings();

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("epsilon", (double) 2); // Invalid.

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);

  ResetSettings();

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("epsilon", (double) 1); // Invalid.

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
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

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);

  ResetSettings();

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("percentage", (double) 0); // Invalid.

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);

  ResetSettings();

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("percentage", (double) 2); // Invalid.

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
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

  RUN_BINDING();

  // Check the neighbors matrix has 4 points for each input point.
  REQUIRE(params.Get<arma::Mat<size_t>>("neighbors").n_rows == 10);
  REQUIRE(params.Get<arma::Mat<size_t>>("neighbors").n_cols == 100);

  // Check the distances matrix has 4 points for each input point.
  REQUIRE(params.Get<arma::mat>("distances").n_rows == 10);
  REQUIRE(params.Get<arma::mat>("distances").n_cols == 100);
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

  RUN_BINDING();

  arma::Mat<size_t> neighbors;
  arma::mat distances;
  neighbors = std::move(params.Get<arma::Mat<size_t>>("neighbors"));
  distances = std::move(params.Get<arma::mat>("distances"));

  // Reset passed parameters.
  KFNModel* m = params.Get<KFNModel*>("output_model");
  params.Get<KFNModel*>("output_model") = NULL;
  CleanMemory();
  ResetSettings();

  // Input saved model, pass the same query and keep k unchanged.
  SetInputParam("input_model", m);
  SetInputParam("query", queryData);
  SetInputParam("k", (int) 10);

  RUN_BINDING();

  // Check that initial output matrices and the output matrices using
  // saved model are equal.
  CheckMatrices(neighbors, params.Get<arma::Mat<size_t>>("neighbors"));
  CheckMatrices(distances, params.Get<arma::mat>("distances"));
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

  RUN_BINDING();

  arma::Mat<size_t> neighbors;
  arma::mat distances;
  neighbors = std::move(params.Get<arma::Mat<size_t>>("neighbors"));
  distances = std::move(params.Get<arma::mat>("distances"));

  CleanMemory();
  ResetSettings();

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("epsilon", (double) 0.8);

  RUN_BINDING();

  CheckMatricesNotEqual(neighbors,
      params.Get<arma::Mat<size_t>>("neighbors"));
  CheckMatricesNotEqual(distances,
      params.Get<arma::mat>("distances"));
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

  RUN_BINDING();

  arma::Mat<size_t> neighbors;
  arma::mat distances;
  neighbors = std::move(params.Get<arma::Mat<size_t>>("neighbors"));
  distances = std::move(params.Get<arma::mat>("distances"));

  CleanMemory();
  ResetSettings();

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("percentage", (double) 0.8);

  RUN_BINDING();

  CheckMatricesNotEqual(neighbors,
      params.Get<arma::Mat<size_t>>("neighbors"));
  CheckMatricesNotEqual(distances,
      params.Get<arma::mat>("distances"));
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
  SetInputParam("random_basis", true);

  RUN_BINDING();

  arma::Mat<size_t> neighbors;
  arma::mat distances;
  neighbors = std::move(params.Get<arma::Mat<size_t>>("neighbors"));
  distances = std::move(params.Get<arma::mat>("distances"));
  REQUIRE(params.Get<KFNModel*>("output_model")->RandomBasis() == true);

  CleanMemory();
  ResetSettings();

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);

  RUN_BINDING();

  CheckMatrices(neighbors, params.Get<arma::Mat<size_t>>("neighbors"));
  CheckMatrices(distances, params.Get<arma::mat>("distances"));
  REQUIRE(params.Get<KFNModel*>("output_model")->RandomBasis() == false);
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

  RUN_BINDING();

  arma::Mat<size_t> neighbors;
  arma::mat distances;
  neighbors = std::move(params.Get<arma::Mat<size_t>>("neighbors"));
  distances = std::move(params.Get<arma::mat>("distances"));

  delete params.Get<KFNModel*>("output_model");
  params.Get<KFNModel*>("output_model") = NULL;

  SetInputParam("reference", referenceData);
  SetInputParam("true_neighbors", neighbors);
  SetInputParam("true_distances", distances);
  SetInputParam("epsilon", (double) 0.5);
  SetInputParam("k", (int) 10);

  REQUIRE_NOTHROW(RUN_BINDING());

  // True output matrices have incorrect shape.
  arma::Mat<size_t> dummyNeighbors;
  arma::mat dummyDistances;
  dummyNeighbors.randu(20, 100);
  dummyDistances.randu(20, 100);

  delete params.Get<KFNModel*>("output_model");
  params.Get<KFNModel*>("output_model") = NULL;

  ResetSettings();

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("true_neighbors", std::move(dummyNeighbors));
  SetInputParam("true_distances", std::move(dummyDistances));
  SetInputParam("k", (int) 10);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
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
    SetInputParam("k", (int) 10);

    RUN_BINDING();

    if (i == 0)
    {
      neighborsCompare = std::move
          (params.Get<arma::Mat<size_t>>("neighbors"));
      distancesCompare = std::move(params.Get<arma::mat>("distances"));
    }
    else
    {
      neighbors = std::move(params.Get<arma::Mat<size_t>>("neighbors"));
      distances = std::move(params.Get<arma::mat>("distances"));

      CheckMatrices(neighborsCompare, neighbors);
      CheckMatrices(distancesCompare, distances);
    }

    delete params.Get<KFNModel*>("output_model");
    params.Get<KFNModel*>("output_model") = NULL;

    // Reset passed parameters.
    ResetSettings();
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
    SetInputParam("k", (int) 10);

    RUN_BINDING();

    if (i == 0)
    {
      neighborsCompare = std::move(
          params.Get<arma::Mat<size_t>>("neighbors"));
      distancesCompare = std::move(params.Get<arma::mat>("distances"));
    }
    else
    {
      neighbors = std::move(params.Get<arma::Mat<size_t>>("neighbors"));
      distances = std::move(params.Get<arma::mat>("distances"));

      CheckMatrices(neighborsCompare, neighbors);
      CheckMatrices(distancesCompare, distances);
    }

    delete params.Get<KFNModel*>("output_model");
    params.Get<KFNModel*>("output_model") = NULL;

    // Reset passed parameters.
    ResetSettings();
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

  RUN_BINDING();

  REQUIRE(params.Get<KFNModel*>("output_model")->LeafSize() == (int) 1);

  // Reset passed parameters.
  CleanMemory();
  ResetSettings();

  // Input saved model, pass the same query and keep k unchanged.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);
  SetInputParam("leaf_size", (int) 10);

  RUN_BINDING();

  // Check that initial output matrices and the output matrices using
  // saved model are equal.
  REQUIRE(params.Get<KFNModel*>("output_model")->LeafSize() == (int) 10);
}
