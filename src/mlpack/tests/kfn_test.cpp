/**
 * @file tests/kfn_test.cpp
 *
 * Tests for KFN (k-furthest-neighbors).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/neighbor_search.hpp>
#include "test_catch_tools.hpp"
#include "catch.hpp"

using namespace mlpack;

/**
 * Simple furthest-neighbors test with small, synthetic dataset.  This is an
 * exhaustive test, which checks that each method for performing the calculation
 * (dual-tree, single-tree, naive) produces the correct results.  An
 * eleven-point dataset and the ten furthest neighbors are taken.  The dataset
 * is in one dimension for simplicity -- the correct functionality of distance
 * functions is not tested here.
 */
TEST_CASE("KFNExhaustiveSyntheticTest", "[KFNTest]")
{
  // Set up our data.
  arma::mat data(1, 11);
  data[0] = 0.05; // Row addressing is unnecessary (they are all 0).
  data[1] = 0.35;
  data[2] = 0.15;
  data[3] = 1.25;
  data[4] = 5.05;
  data[5] = -0.22;
  data[6] = -2.00;
  data[7] = -1.30;
  data[8] = 0.45;
  data[9] = 0.90;
  data[10] = 1.00;

  typedef BinarySpaceTree<EuclideanDistance,
      NeighborSearchStat<FurthestNeighborSort>, arma::mat> TreeType;

  // We will loop through three times, one for each method of performing the
  // calculation.  We'll always use 10 neighbors, so set that parameter.
  std::vector<size_t> oldFromNew;
  std::vector<size_t> newFromOld;
  TreeType tree(data, oldFromNew, newFromOld, 1);
  KFN kfn(std::move(tree));

  for (int i = 0; i < 3; ++i)
  {
    switch (i)
    {
      case 0: // Use the dual-tree method.
        kfn.SearchMode() = DUAL_TREE_MODE;
        break;
      case 1: // Use the single-tree method.
        kfn.SearchMode() = SINGLE_TREE_MODE;
        break;
      case 2: // Use the naive method.
        kfn.SearchMode() = NAIVE_MODE;
        break;
    }

    // Now perform the actual calculation.
    arma::Mat<size_t> neighbors;
    arma::mat distances;
    kfn.Search(10, neighbors, distances);

    // Now the exhaustive check for correctness.  This will be long.  We must
    // also remember that the distances returned are squared distances.  As a
    // result, distance comparisons are written out as (distance * distance) for
    // readability.

    // Neighbors of point 0.
    REQUIRE(neighbors(9, newFromOld[0]) == newFromOld[2]);
    REQUIRE(distances(9, newFromOld[0]) == Approx(0.10).epsilon(1e-7));
    REQUIRE(neighbors(8, newFromOld[0]) == newFromOld[5]);
    REQUIRE(distances(8, newFromOld[0]) == Approx(0.27).epsilon(1e-7));
    REQUIRE(neighbors(7, newFromOld[0]) == newFromOld[1]);
    REQUIRE(distances(7, newFromOld[0]) == Approx(0.30).epsilon(1e-7));
    REQUIRE(neighbors(6, newFromOld[0]) == newFromOld[8]);
    REQUIRE(distances(6, newFromOld[0]) == Approx(0.40).epsilon(1e-7));
    REQUIRE(neighbors(5, newFromOld[0]) == newFromOld[9]);
    REQUIRE(distances(5, newFromOld[0]) == Approx(0.85).epsilon(1e-7));
    REQUIRE(neighbors(4, newFromOld[0]) == newFromOld[10]);
    REQUIRE(distances(4, newFromOld[0]) == Approx(0.95).epsilon(1e-7));
    REQUIRE(neighbors(3, newFromOld[0]) == newFromOld[3]);
    REQUIRE(distances(3, newFromOld[0]) == Approx(1.20).epsilon(1e-7));
    REQUIRE(neighbors(2, newFromOld[0]) == newFromOld[7]);
    REQUIRE(distances(2, newFromOld[0]) == Approx(1.35).epsilon(1e-7));
    REQUIRE(neighbors(1, newFromOld[0]) == newFromOld[6]);
    REQUIRE(distances(1, newFromOld[0]) == Approx(2.05).epsilon(1e-7));
    REQUIRE(neighbors(0, newFromOld[0]) == newFromOld[4]);
    REQUIRE(distances(0, newFromOld[0]) == Approx(5.00).epsilon(1e-7));

    // Neighbors of point 1.
    REQUIRE(neighbors(9, newFromOld[1]) == newFromOld[8]);
    REQUIRE(distances(9, newFromOld[1]) == Approx(0.10).epsilon(1e-7));
    REQUIRE(neighbors(8, newFromOld[1]) == newFromOld[2]);
    REQUIRE(distances(8, newFromOld[1]) == Approx(0.20).epsilon(1e-7));
    REQUIRE(neighbors(7, newFromOld[1]) == newFromOld[0]);
    REQUIRE(distances(7, newFromOld[1]) == Approx(0.30).epsilon(1e-7));
    REQUIRE(neighbors(6, newFromOld[1]) == newFromOld[9]);
    REQUIRE(distances(6, newFromOld[1]) == Approx(0.55).epsilon(1e-7));
    REQUIRE(neighbors(5, newFromOld[1]) == newFromOld[5]);
    REQUIRE(distances(5, newFromOld[1]) == Approx(0.57).epsilon(1e-7));
    REQUIRE(neighbors(4, newFromOld[1]) == newFromOld[10]);
    REQUIRE(distances(4, newFromOld[1]) == Approx(0.65).epsilon(1e-7));
    REQUIRE(neighbors(3, newFromOld[1]) == newFromOld[3]);
    REQUIRE(distances(3, newFromOld[1]) == Approx(0.90).epsilon(1e-7));
    REQUIRE(neighbors(2, newFromOld[1]) == newFromOld[7]);
    REQUIRE(distances(2, newFromOld[1]) == Approx(1.65).epsilon(1e-7));
    REQUIRE(neighbors(1, newFromOld[1]) == newFromOld[6]);
    REQUIRE(distances(1, newFromOld[1]) == Approx(2.35).epsilon(1e-7));
    REQUIRE(neighbors(0, newFromOld[1]) == newFromOld[4]);
    REQUIRE(distances(0, newFromOld[1]) == Approx(4.70).epsilon(1e-7));

    // Neighbors of point 2.
    REQUIRE(neighbors(9, newFromOld[2]) == newFromOld[0]);
    REQUIRE(distances(9, newFromOld[2]) == Approx(0.10).epsilon(1e-7));
    REQUIRE(neighbors(8, newFromOld[2]) == newFromOld[1]);
    REQUIRE(distances(8, newFromOld[2]) == Approx(0.20).epsilon(1e-7));
    REQUIRE(neighbors(7, newFromOld[2]) == newFromOld[8]);
    REQUIRE(distances(7, newFromOld[2]) == Approx(0.30).epsilon(1e-7));
    REQUIRE(neighbors(6, newFromOld[2]) == newFromOld[5]);
    REQUIRE(distances(6, newFromOld[2]) == Approx(0.37).epsilon(1e-7));
    REQUIRE(neighbors(5, newFromOld[2]) == newFromOld[9]);
    REQUIRE(distances(5, newFromOld[2]) == Approx(0.75).epsilon(1e-7));
    REQUIRE(neighbors(4, newFromOld[2]) == newFromOld[10]);
    REQUIRE(distances(4, newFromOld[2]) == Approx(0.85).epsilon(1e-7));
    REQUIRE(neighbors(3, newFromOld[2]) == newFromOld[3]);
    REQUIRE(distances(3, newFromOld[2]) == Approx(1.10).epsilon(1e-7));
    REQUIRE(neighbors(2, newFromOld[2]) == newFromOld[7]);
    REQUIRE(distances(2, newFromOld[2]) == Approx(1.45).epsilon(1e-7));
    REQUIRE(neighbors(1, newFromOld[2]) == newFromOld[6]);
    REQUIRE(distances(1, newFromOld[2]) == Approx(2.15).epsilon(1e-7));
    REQUIRE(neighbors(0, newFromOld[2]) == newFromOld[4]);
    REQUIRE(distances(0, newFromOld[2]) == Approx(4.90).epsilon(1e-7));

    // Neighbors of point 3.
    REQUIRE(neighbors(9, newFromOld[3]) == newFromOld[10]);
    REQUIRE(distances(9, newFromOld[3]) == Approx(0.25).epsilon(1e-7));
    REQUIRE(neighbors(8, newFromOld[3]) == newFromOld[9]);
    REQUIRE(distances(8, newFromOld[3]) == Approx(0.35).epsilon(1e-7));
    REQUIRE(neighbors(7, newFromOld[3]) == newFromOld[8]);
    REQUIRE(distances(7, newFromOld[3]) == Approx(0.80).epsilon(1e-7));
    REQUIRE(neighbors(6, newFromOld[3]) == newFromOld[1]);
    REQUIRE(distances(6, newFromOld[3]) == Approx(0.90).epsilon(1e-7));
    REQUIRE(neighbors(5, newFromOld[3]) == newFromOld[2]);
    REQUIRE(distances(5, newFromOld[3]) == Approx(1.10).epsilon(1e-7));
    REQUIRE(neighbors(4, newFromOld[3]) == newFromOld[0]);
    REQUIRE(distances(4, newFromOld[3]) == Approx(1.20).epsilon(1e-7));
    REQUIRE(neighbors(3, newFromOld[3]) == newFromOld[5]);
    REQUIRE(distances(3, newFromOld[3]) == Approx(1.47).epsilon(1e-7));
    REQUIRE(neighbors(2, newFromOld[3]) == newFromOld[7]);
    REQUIRE(distances(2, newFromOld[3]) == Approx(2.55).epsilon(1e-7));
    REQUIRE(neighbors(1, newFromOld[3]) == newFromOld[6]);
    REQUIRE(distances(1, newFromOld[3]) == Approx(3.25).epsilon(1e-7));
    REQUIRE(neighbors(0, newFromOld[3]) == newFromOld[4]);
    REQUIRE(distances(0, newFromOld[3]) == Approx(3.80).epsilon(1e-7));

    // Neighbors of point 4.
    REQUIRE(neighbors(9, newFromOld[4]) == newFromOld[3]);
    REQUIRE(distances(9, newFromOld[4]) == Approx(3.80).epsilon(1e-7));
    REQUIRE(neighbors(8, newFromOld[4]) == newFromOld[10]);
    REQUIRE(distances(8, newFromOld[4]) == Approx(4.05).epsilon(1e-7));
    REQUIRE(neighbors(7, newFromOld[4]) == newFromOld[9]);
    REQUIRE(distances(7, newFromOld[4]) == Approx(4.15).epsilon(1e-7));
    REQUIRE(neighbors(6, newFromOld[4]) == newFromOld[8]);
    REQUIRE(distances(6, newFromOld[4]) == Approx(4.60).epsilon(1e-7));
    REQUIRE(neighbors(5, newFromOld[4]) == newFromOld[1]);
    REQUIRE(distances(5, newFromOld[4]) == Approx(4.70).epsilon(1e-7));
    REQUIRE(neighbors(4, newFromOld[4]) == newFromOld[2]);
    REQUIRE(distances(4, newFromOld[4]) == Approx(4.90).epsilon(1e-7));
    REQUIRE(neighbors(3, newFromOld[4]) == newFromOld[0]);
    REQUIRE(distances(3, newFromOld[4]) == Approx(5.00).epsilon(1e-7));
    REQUIRE(neighbors(2, newFromOld[4]) == newFromOld[5]);
    REQUIRE(distances(2, newFromOld[4]) == Approx(5.27).epsilon(1e-7));
    REQUIRE(neighbors(1, newFromOld[4]) == newFromOld[7]);
    REQUIRE(distances(1, newFromOld[4]) == Approx(6.35).epsilon(1e-7));
    REQUIRE(neighbors(0, newFromOld[4]) == newFromOld[6]);
    REQUIRE(distances(0, newFromOld[4]) == Approx(7.05).epsilon(1e-7));

    // Neighbors of point 5.
    REQUIRE(neighbors(9, newFromOld[5]) == newFromOld[0]);
    REQUIRE(distances(9, newFromOld[5]) == Approx(0.27).epsilon(1e-7));
    REQUIRE(neighbors(8, newFromOld[5]) == newFromOld[2]);
    REQUIRE(distances(8, newFromOld[5]) == Approx(0.37).epsilon(1e-7));
    REQUIRE(neighbors(7, newFromOld[5]) == newFromOld[1]);
    REQUIRE(distances(7, newFromOld[5]) == Approx(0.57).epsilon(1e-7));
    REQUIRE(neighbors(6, newFromOld[5]) == newFromOld[8]);
    REQUIRE(distances(6, newFromOld[5]) == Approx(0.67).epsilon(1e-7));
    REQUIRE(neighbors(5, newFromOld[5]) == newFromOld[7]);
    REQUIRE(distances(5, newFromOld[5]) == Approx(1.08).epsilon(1e-7));
    REQUIRE(neighbors(4, newFromOld[5]) == newFromOld[9]);
    REQUIRE(distances(4, newFromOld[5]) == Approx(1.12).epsilon(1e-7));
    REQUIRE(neighbors(3, newFromOld[5]) == newFromOld[10]);
    REQUIRE(distances(3, newFromOld[5]) == Approx(1.22).epsilon(1e-7));
    REQUIRE(neighbors(2, newFromOld[5]) == newFromOld[3]);
    REQUIRE(distances(2, newFromOld[5]) == Approx(1.47).epsilon(1e-7));
    REQUIRE(neighbors(1, newFromOld[5]) == newFromOld[6]);
    REQUIRE(distances(1, newFromOld[5]) == Approx(1.78).epsilon(1e-7));
    REQUIRE(neighbors(0, newFromOld[5]) == newFromOld[4]);
    REQUIRE(distances(0, newFromOld[5]) == Approx(5.27).epsilon(1e-7));

    // Neighbors of point 6.
    REQUIRE(neighbors(9, newFromOld[6]) == newFromOld[7]);
    REQUIRE(distances(9, newFromOld[6]) == Approx(0.70).epsilon(1e-7));
    REQUIRE(neighbors(8, newFromOld[6]) == newFromOld[5]);
    REQUIRE(distances(8, newFromOld[6]) == Approx(1.78).epsilon(1e-7));
    REQUIRE(neighbors(7, newFromOld[6]) == newFromOld[0]);
    REQUIRE(distances(7, newFromOld[6]) == Approx(2.05).epsilon(1e-7));
    REQUIRE(neighbors(6, newFromOld[6]) == newFromOld[2]);
    REQUIRE(distances(6, newFromOld[6]) == Approx(2.15).epsilon(1e-7));
    REQUIRE(neighbors(5, newFromOld[6]) == newFromOld[1]);
    REQUIRE(distances(5, newFromOld[6]) == Approx(2.35).epsilon(1e-7));
    REQUIRE(neighbors(4, newFromOld[6]) == newFromOld[8]);
    REQUIRE(distances(4, newFromOld[6]) == Approx(2.45).epsilon(1e-7));
    REQUIRE(neighbors(3, newFromOld[6]) == newFromOld[9]);
    REQUIRE(distances(3, newFromOld[6]) == Approx(2.90).epsilon(1e-7));
    REQUIRE(neighbors(2, newFromOld[6]) == newFromOld[10]);
    REQUIRE(distances(2, newFromOld[6]) == Approx(3.00).epsilon(1e-7));
    REQUIRE(neighbors(1, newFromOld[6]) == newFromOld[3]);
    REQUIRE(distances(1, newFromOld[6]) == Approx(3.25).epsilon(1e-7));
    REQUIRE(neighbors(0, newFromOld[6]) == newFromOld[4]);
    REQUIRE(distances(0, newFromOld[6]) == Approx(7.05).epsilon(1e-7));

    // Neighbors of point 7.
    REQUIRE(neighbors(9, newFromOld[7]) == newFromOld[6]);
    REQUIRE(distances(9, newFromOld[7]) == Approx(0.70).epsilon(1e-7));
    REQUIRE(neighbors(8, newFromOld[7]) == newFromOld[5]);
    REQUIRE(distances(8, newFromOld[7]) == Approx(1.08).epsilon(1e-7));
    REQUIRE(neighbors(7, newFromOld[7]) == newFromOld[0]);
    REQUIRE(distances(7, newFromOld[7]) == Approx(1.35).epsilon(1e-7));
    REQUIRE(neighbors(6, newFromOld[7]) == newFromOld[2]);
    REQUIRE(distances(6, newFromOld[7]) == Approx(1.45).epsilon(1e-7));
    REQUIRE(neighbors(5, newFromOld[7]) == newFromOld[1]);
    REQUIRE(distances(5, newFromOld[7]) == Approx(1.65).epsilon(1e-7));
    REQUIRE(neighbors(4, newFromOld[7]) == newFromOld[8]);
    REQUIRE(distances(4, newFromOld[7]) == Approx(1.75).epsilon(1e-7));
    REQUIRE(neighbors(3, newFromOld[7]) == newFromOld[9]);
    REQUIRE(distances(3, newFromOld[7]) == Approx(2.20).epsilon(1e-7));
    REQUIRE(neighbors(2, newFromOld[7]) == newFromOld[10]);
    REQUIRE(distances(2, newFromOld[7]) == Approx(2.30).epsilon(1e-7));
    REQUIRE(neighbors(1, newFromOld[7]) == newFromOld[3]);
    REQUIRE(distances(1, newFromOld[7]) == Approx(2.55).epsilon(1e-7));
    REQUIRE(neighbors(0, newFromOld[7]) == newFromOld[4]);
    REQUIRE(distances(0, newFromOld[7]) == Approx(6.35).epsilon(1e-7));

    // Neighbors of point 8.
    REQUIRE(neighbors(9, newFromOld[8]) == newFromOld[1]);
    REQUIRE(distances(9, newFromOld[8]) == Approx(0.10).epsilon(1e-7));
    REQUIRE(neighbors(8, newFromOld[8]) == newFromOld[2]);
    REQUIRE(distances(8, newFromOld[8]) == Approx(0.30).epsilon(1e-7));
    REQUIRE(neighbors(7, newFromOld[8]) == newFromOld[0]);
    REQUIRE(distances(7, newFromOld[8]) == Approx(0.40).epsilon(1e-7));
    REQUIRE(neighbors(6, newFromOld[8]) == newFromOld[9]);
    REQUIRE(distances(6, newFromOld[8]) == Approx(0.45).epsilon(1e-7));
    REQUIRE(neighbors(5, newFromOld[8]) == newFromOld[10]);
    REQUIRE(distances(5, newFromOld[8]) == Approx(0.55).epsilon(1e-7));
    REQUIRE(neighbors(4, newFromOld[8]) == newFromOld[5]);
    REQUIRE(distances(4, newFromOld[8]) == Approx(0.67).epsilon(1e-7));
    REQUIRE(neighbors(3, newFromOld[8]) == newFromOld[3]);
    REQUIRE(distances(3, newFromOld[8]) == Approx(0.80).epsilon(1e-7));
    REQUIRE(neighbors(2, newFromOld[8]) == newFromOld[7]);
    REQUIRE(distances(2, newFromOld[8]) == Approx(1.75).epsilon(1e-7));
    REQUIRE(neighbors(1, newFromOld[8]) == newFromOld[6]);
    REQUIRE(distances(1, newFromOld[8]) == Approx(2.45).epsilon(1e-7));
    REQUIRE(neighbors(0, newFromOld[8]) == newFromOld[4]);
    REQUIRE(distances(0, newFromOld[8]) == Approx(4.60).epsilon(1e-7));

    // Neighbors of point 9.
    REQUIRE(neighbors(9, newFromOld[9]) == newFromOld[10]);
    REQUIRE(distances(9, newFromOld[9]) == Approx(0.10).epsilon(1e-7));
    REQUIRE(neighbors(8, newFromOld[9]) == newFromOld[3]);
    REQUIRE(distances(8, newFromOld[9]) == Approx(0.35).epsilon(1e-7));
    REQUIRE(neighbors(7, newFromOld[9]) == newFromOld[8]);
    REQUIRE(distances(7, newFromOld[9]) == Approx(0.45).epsilon(1e-7));
    REQUIRE(neighbors(6, newFromOld[9]) == newFromOld[1]);
    REQUIRE(distances(6, newFromOld[9]) == Approx(0.55).epsilon(1e-7));
    REQUIRE(neighbors(5, newFromOld[9]) == newFromOld[2]);
    REQUIRE(distances(5, newFromOld[9]) == Approx(0.75).epsilon(1e-7));
    REQUIRE(neighbors(4, newFromOld[9]) == newFromOld[0]);
    REQUIRE(distances(4, newFromOld[9]) == Approx(0.85).epsilon(1e-7));
    REQUIRE(neighbors(3, newFromOld[9]) == newFromOld[5]);
    REQUIRE(distances(3, newFromOld[9]) == Approx(1.12).epsilon(1e-7));
    REQUIRE(neighbors(2, newFromOld[9]) == newFromOld[7]);
    REQUIRE(distances(2, newFromOld[9]) == Approx(2.20).epsilon(1e-7));
    REQUIRE(neighbors(1, newFromOld[9]) == newFromOld[6]);
    REQUIRE(distances(1, newFromOld[9]) == Approx(2.90).epsilon(1e-7));
    REQUIRE(neighbors(0, newFromOld[9]) == newFromOld[4]);
    REQUIRE(distances(0, newFromOld[9]) == Approx(4.15).epsilon(1e-7));

    // Neighbors of point 10.
    REQUIRE(neighbors(9, newFromOld[10]) == newFromOld[9]);
    REQUIRE(distances(9, newFromOld[10]) == Approx(0.10).epsilon(1e-7));
    REQUIRE(neighbors(8, newFromOld[10]) == newFromOld[3]);
    REQUIRE(distances(8, newFromOld[10]) == Approx(0.25).epsilon(1e-7));
    REQUIRE(neighbors(7, newFromOld[10]) == newFromOld[8]);
    REQUIRE(distances(7, newFromOld[10]) == Approx(0.55).epsilon(1e-7));
    REQUIRE(neighbors(6, newFromOld[10]) == newFromOld[1]);
    REQUIRE(distances(6, newFromOld[10]) == Approx(0.65).epsilon(1e-7));
    REQUIRE(neighbors(5, newFromOld[10]) == newFromOld[2]);
    REQUIRE(distances(5, newFromOld[10]) == Approx(0.85).epsilon(1e-7));
    REQUIRE(neighbors(4, newFromOld[10]) == newFromOld[0]);
    REQUIRE(distances(4, newFromOld[10]) == Approx(0.95).epsilon(1e-7));
    REQUIRE(neighbors(3, newFromOld[10]) == newFromOld[5]);
    REQUIRE(distances(3, newFromOld[10]) == Approx(1.22).epsilon(1e-7));
    REQUIRE(neighbors(2, newFromOld[10]) == newFromOld[7]);
    REQUIRE(distances(2, newFromOld[10]) == Approx(2.30).epsilon(1e-7));
    REQUIRE(neighbors(1, newFromOld[10]) == newFromOld[6]);
    REQUIRE(distances(1, newFromOld[10]) == Approx(3.00).epsilon(1e-7));
    REQUIRE(neighbors(0, newFromOld[10]) == newFromOld[4]);
    REQUIRE(distances(0, newFromOld[10]) == Approx(4.05).epsilon(1e-7));
  }
}

/**
 * Test the dual-tree furthest-neighbors method with the naive method.  This
 * uses both a query and reference dataset.
 *
 * Errors are produced if the results are not identical.
 */
TEST_CASE("KFNDualTreeVsNaive1", "[KFNTest]")
{
  arma::mat dataset;

  // Hard-coded filename: bad?
  if (!data::Load("test_data_3_1000.csv", dataset))
    FAIL("Cannot load test dataset test_data_3_1000.csv");

  KFN kfn(dataset);

  KFN naive(dataset, NAIVE_MODE);

  arma::Mat<size_t> neighborsTree;
  arma::mat distancesTree;
  kfn.Search(dataset, 15, neighborsTree, distancesTree);

  arma::Mat<size_t> neighborsNaive;
  arma::mat distancesNaive;
  naive.Search(dataset, 15, neighborsNaive, distancesNaive);

  for (size_t i = 0; i < neighborsTree.n_elem; ++i)
  {
    REQUIRE(neighborsTree[i] == neighborsNaive[i]);
    REQUIRE(distancesTree[i] == Approx(distancesNaive[i]).epsilon(1e-7));
  }
}

/**
 * Test the dual-tree furthest-neighbors method with the naive method.  This
 * uses only a reference dataset.
 *
 * Errors are produced if the results are not identical.
 */
TEST_CASE("KFNDualTreeVsNaive2", "[KFNTest]")
{
  arma::mat dataset;

  // Hard-coded filename: bad?
  // Code duplication: also bad!
  if (!data::Load("test_data_3_1000.csv", dataset))
    FAIL("Cannot load test dataset test_data_3_1000.csv");

  KFN kfn(dataset);

  KFN naive(dataset, NAIVE_MODE);

  arma::Mat<size_t> neighborsTree;
  arma::mat distancesTree;
  kfn.Search(15, neighborsTree, distancesTree);

  arma::Mat<size_t> neighborsNaive;
  arma::mat distancesNaive;
  naive.Search(15, neighborsNaive, distancesNaive);

  for (size_t i = 0; i < neighborsTree.n_elem; ++i)
  {
    REQUIRE(neighborsTree[i] == neighborsNaive[i]);
    REQUIRE(distancesTree[i] == Approx(distancesNaive[i]).epsilon(1e-7));
  }
}

/**
 * Test the single-tree furthest-neighbors method with the naive method.  This
 * uses only a reference dataset.
 *
 * Errors are produced if the results are not identical.
 */
TEST_CASE("KFNSingleTreeVsNaive", "[KFNTest]")
{
  arma::mat dataset;

  // Hard-coded filename: bad!
  // Code duplication: also bad!
  if (!data::Load("test_data_3_1000.csv", dataset))
    FAIL("Cannot load test dataset test_data_3_1000.csv");

  KFN kfn(dataset, SINGLE_TREE_MODE);

  KFN naive(dataset, NAIVE_MODE);

  arma::Mat<size_t> neighborsTree;
  arma::mat distancesTree;
  kfn.Search(15, neighborsTree, distancesTree);

  arma::Mat<size_t> neighborsNaive;
  arma::mat distancesNaive;
  naive.Search(15, neighborsNaive, distancesNaive);

  for (size_t i = 0; i < neighborsTree.n_elem; ++i)
  {
    REQUIRE(neighborsTree[i] == neighborsNaive[i]);
    REQUIRE(distancesTree[i] == Approx(distancesNaive[i]).epsilon(1e-7));
  }
}

/**
 * Test the cover tree single-tree furthest-neighbors method against the naive
 * method.  This uses only a random reference dataset.
 *
 * Errors are produced if the results are not identical.
 */
TEST_CASE("KFNSingleCoverTreeTest", "[KFNTest]")
{
  arma::mat data;
  data.randu(75, 1000); // 75 dimensional, 1000 points.

  // This depends on the cover tree not mapping points.
  CoverTree<LMetric<2>, NeighborSearchStat<FurthestNeighborSort>, arma::mat,
      FirstPointIsRoot> tree(data);

  NeighborSearch<FurthestNeighborSort, LMetric<2>, arma::mat, StandardCoverTree>
      coverTreeSearch(std::move(tree), SINGLE_TREE_MODE);

  KFN naive(data, NAIVE_MODE);

  arma::Mat<size_t> coverTreeNeighbors;
  arma::mat coverTreeDistances;
  coverTreeSearch.Search(data, 15, coverTreeNeighbors, coverTreeDistances);

  arma::Mat<size_t> naiveNeighbors;
  arma::mat naiveDistances;
  naive.Search(data, 15, naiveNeighbors, naiveDistances);

  for (size_t i = 0; i < coverTreeNeighbors.n_elem; ++i)
  {
    REQUIRE(coverTreeNeighbors[i] == naiveNeighbors[i]);
    REQUIRE(coverTreeDistances[i] == Approx(naiveDistances[i]).epsilon(1e-7));
  }
}

/**
 * Test the cover tree dual-tree furthest neighbors method against the naive
 * method.
 */
TEST_CASE("KFNDualCoverTreeTest", "[KFNTest]")
{
  arma::mat dataset;
  if (!data::Load("test_data_3_1000.csv", dataset))
    FAIL("Cannot load test dataset test_data_3_1000.csv");

  KFN tree(dataset);

  arma::Mat<size_t> kdNeighbors;
  arma::mat kdDistances;
  tree.Search(dataset, 5, kdNeighbors, kdDistances);

  typedef CoverTree<LMetric<2, true>, NeighborSearchStat<FurthestNeighborSort>,
      arma::mat, FirstPointIsRoot> TreeType;

  TreeType referenceTree(dataset);

  NeighborSearch<FurthestNeighborSort, LMetric<2, true>, arma::mat,
      StandardCoverTree> coverTreeSearch(std::move(referenceTree));

  arma::Mat<size_t> coverNeighbors;
  arma::mat coverDistances;
  coverTreeSearch.Search(dataset, 5, coverNeighbors, coverDistances);

  for (size_t i = 0; i < coverNeighbors.n_elem; ++i)
  {
    REQUIRE(coverNeighbors(i) == kdNeighbors(i));
    REQUIRE(coverDistances(i) == Approx(kdDistances(i)).epsilon(1e-7));
  }
}

/**
 * Test the ball tree single-tree furthest-neighbors method against the naive
 * method.  This uses only a random reference dataset.
 *
 * Errors are produced if the results are not identical.
 */
TEST_CASE("KFNSingleBallTreeTest", "[KFNTest]")
{
  arma::mat data;
  data.randu(75, 1000); // 75 dimensional, 1000 points.

  typedef BallTree<EuclideanDistance, NeighborSearchStat<FurthestNeighborSort>,
      arma::mat> TreeType;
  TreeType tree(data);

  KFN naive(tree.Dataset(), NAIVE_MODE);

  // BinarySpaceTree modifies data. Use modified data to maintain the
  // correspondence between points in the dataset for both methods. The order of
  // query points in both methods should be same.
  NeighborSearch<FurthestNeighborSort, LMetric<2>, arma::mat, BallTree>
      ballTreeSearch(std::move(tree), SINGLE_TREE_MODE);

  arma::Mat<size_t> ballTreeNeighbors;
  arma::mat ballTreeDistances;
  ballTreeSearch.Search(15, ballTreeNeighbors, ballTreeDistances);

  arma::Mat<size_t> naiveNeighbors;
  arma::mat naiveDistances;
  naive.Search(15, naiveNeighbors, naiveDistances);

  for (size_t i = 0; i < ballTreeNeighbors.n_elem; ++i)
  {
    REQUIRE(ballTreeNeighbors[i] == naiveNeighbors[i]);
    REQUIRE(ballTreeDistances[i] == Approx(naiveDistances[i]).epsilon(1e-7));
  }
}

/**
 * Test the ball tree dual-tree furthest neighbors method against the naive
 * method.
 */
TEST_CASE("KFNDualBallTreeTest", "[KFNTest]")
{
  arma::mat dataset;
  if (!data::Load("test_data_3_1000.csv", dataset))
    FAIL("Cannot load test dataset test_data_3_1000.csv");

  KFN tree(dataset);

  arma::Mat<size_t> kdNeighbors;
  arma::mat kdDistances;
  tree.Search(5, kdNeighbors, kdDistances);

  NeighborSearch<FurthestNeighborSort, LMetric<2, true>, arma::mat, BallTree>
      ballTreeSearch(dataset);

  arma::Mat<size_t> ballNeighbors;
  arma::mat ballDistances;
  ballTreeSearch.Search(5, ballNeighbors, ballDistances);

  for (size_t i = 0; i < ballNeighbors.n_elem; ++i)
  {
    REQUIRE(ballNeighbors(i) == kdNeighbors(i));
    REQUIRE(ballDistances(i) == Approx(kdDistances(i)).epsilon(1e-7));
  }
}
