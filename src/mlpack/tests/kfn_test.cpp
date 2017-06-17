/**
 * @file kfn_test.cpp
 *
 * Tests for KFN (k-furthest-neighbors).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <mlpack/core/tree/cover_tree.hpp>
#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::neighbor;
using namespace mlpack::tree;
using namespace mlpack::metric;
using namespace mlpack::bound;

BOOST_AUTO_TEST_SUITE(KFNTest);

/**
 * Simple furthest-neighbors test with small, synthetic dataset.  This is an
 * exhaustive test, which checks that each method for performing the calculation
 * (dual-tree, single-tree, naive) produces the correct results.  An
 * eleven-point dataset and the ten furthest neighbors are taken.  The dataset
 * is in one dimension for simplicity -- the correct functionality of distance
 * functions is not tested here.
 */
BOOST_AUTO_TEST_CASE(ExhaustiveSyntheticTest)
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

  for (int i = 0; i < 3; i++)
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
    BOOST_REQUIRE_EQUAL(neighbors(9, newFromOld[0]), newFromOld[2]);
    BOOST_REQUIRE_CLOSE(distances(9, newFromOld[0]), 0.10, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(8, newFromOld[0]), newFromOld[5]);
    BOOST_REQUIRE_CLOSE(distances(8, newFromOld[0]), 0.27, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(7, newFromOld[0]), newFromOld[1]);
    BOOST_REQUIRE_CLOSE(distances(7, newFromOld[0]), 0.30, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(6, newFromOld[0]), newFromOld[8]);
    BOOST_REQUIRE_CLOSE(distances(6, newFromOld[0]), 0.40, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(5, newFromOld[0]), newFromOld[9]);
    BOOST_REQUIRE_CLOSE(distances(5, newFromOld[0]), 0.85, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(4, newFromOld[0]), newFromOld[10]);
    BOOST_REQUIRE_CLOSE(distances(4, newFromOld[0]), 0.95, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(3, newFromOld[0]), newFromOld[3]);
    BOOST_REQUIRE_CLOSE(distances(3, newFromOld[0]), 1.20, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(2, newFromOld[0]), newFromOld[7]);
    BOOST_REQUIRE_CLOSE(distances(2, newFromOld[0]), 1.35, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(1, newFromOld[0]), newFromOld[6]);
    BOOST_REQUIRE_CLOSE(distances(1, newFromOld[0]), 2.05, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(0, newFromOld[0]), newFromOld[4]);
    BOOST_REQUIRE_CLOSE(distances(0, newFromOld[0]), 5.00, 1e-5);

    // Neighbors of point 1.
    BOOST_REQUIRE_EQUAL(neighbors(9, newFromOld[1]), newFromOld[8]);
    BOOST_REQUIRE_CLOSE(distances(9, newFromOld[1]), 0.10, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(8, newFromOld[1]), newFromOld[2]);
    BOOST_REQUIRE_CLOSE(distances(8, newFromOld[1]), 0.20, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(7, newFromOld[1]), newFromOld[0]);
    BOOST_REQUIRE_CLOSE(distances(7, newFromOld[1]), 0.30, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(6, newFromOld[1]), newFromOld[9]);
    BOOST_REQUIRE_CLOSE(distances(6, newFromOld[1]), 0.55, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(5, newFromOld[1]), newFromOld[5]);
    BOOST_REQUIRE_CLOSE(distances(5, newFromOld[1]), 0.57, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(4, newFromOld[1]), newFromOld[10]);
    BOOST_REQUIRE_CLOSE(distances(4, newFromOld[1]), 0.65, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(3, newFromOld[1]), newFromOld[3]);
    BOOST_REQUIRE_CLOSE(distances(3, newFromOld[1]), 0.90, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(2, newFromOld[1]), newFromOld[7]);
    BOOST_REQUIRE_CLOSE(distances(2, newFromOld[1]), 1.65, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(1, newFromOld[1]), newFromOld[6]);
    BOOST_REQUIRE_CLOSE(distances(1, newFromOld[1]), 2.35, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(0, newFromOld[1]), newFromOld[4]);
    BOOST_REQUIRE_CLOSE(distances(0, newFromOld[1]), 4.70, 1e-5);

    // Neighbors of point 2.
    BOOST_REQUIRE_EQUAL(neighbors(9, newFromOld[2]), newFromOld[0]);
    BOOST_REQUIRE_CLOSE(distances(9, newFromOld[2]), 0.10, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(8, newFromOld[2]), newFromOld[1]);
    BOOST_REQUIRE_CLOSE(distances(8, newFromOld[2]), 0.20, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(7, newFromOld[2]), newFromOld[8]);
    BOOST_REQUIRE_CLOSE(distances(7, newFromOld[2]), 0.30, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(6, newFromOld[2]), newFromOld[5]);
    BOOST_REQUIRE_CLOSE(distances(6, newFromOld[2]), 0.37, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(5, newFromOld[2]), newFromOld[9]);
    BOOST_REQUIRE_CLOSE(distances(5, newFromOld[2]), 0.75, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(4, newFromOld[2]), newFromOld[10]);
    BOOST_REQUIRE_CLOSE(distances(4, newFromOld[2]), 0.85, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(3, newFromOld[2]), newFromOld[3]);
    BOOST_REQUIRE_CLOSE(distances(3, newFromOld[2]), 1.10, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(2, newFromOld[2]), newFromOld[7]);
    BOOST_REQUIRE_CLOSE(distances(2, newFromOld[2]), 1.45, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(1, newFromOld[2]), newFromOld[6]);
    BOOST_REQUIRE_CLOSE(distances(1, newFromOld[2]), 2.15, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(0, newFromOld[2]), newFromOld[4]);
    BOOST_REQUIRE_CLOSE(distances(0, newFromOld[2]), 4.90, 1e-5);

    // Neighbors of point 3.
    BOOST_REQUIRE_EQUAL(neighbors(9, newFromOld[3]), newFromOld[10]);
    BOOST_REQUIRE_CLOSE(distances(9, newFromOld[3]), 0.25, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(8, newFromOld[3]), newFromOld[9]);
    BOOST_REQUIRE_CLOSE(distances(8, newFromOld[3]), 0.35, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(7, newFromOld[3]), newFromOld[8]);
    BOOST_REQUIRE_CLOSE(distances(7, newFromOld[3]), 0.80, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(6, newFromOld[3]), newFromOld[1]);
    BOOST_REQUIRE_CLOSE(distances(6, newFromOld[3]), 0.90, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(5, newFromOld[3]), newFromOld[2]);
    BOOST_REQUIRE_CLOSE(distances(5, newFromOld[3]), 1.10, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(4, newFromOld[3]), newFromOld[0]);
    BOOST_REQUIRE_CLOSE(distances(4, newFromOld[3]), 1.20, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(3, newFromOld[3]), newFromOld[5]);
    BOOST_REQUIRE_CLOSE(distances(3, newFromOld[3]), 1.47, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(2, newFromOld[3]), newFromOld[7]);
    BOOST_REQUIRE_CLOSE(distances(2, newFromOld[3]), 2.55, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(1, newFromOld[3]), newFromOld[6]);
    BOOST_REQUIRE_CLOSE(distances(1, newFromOld[3]), 3.25, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(0, newFromOld[3]), newFromOld[4]);
    BOOST_REQUIRE_CLOSE(distances(0, newFromOld[3]), 3.80, 1e-5);

    // Neighbors of point 4.
    BOOST_REQUIRE_EQUAL(neighbors(9, newFromOld[4]), newFromOld[3]);
    BOOST_REQUIRE_CLOSE(distances(9, newFromOld[4]), 3.80, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(8, newFromOld[4]), newFromOld[10]);
    BOOST_REQUIRE_CLOSE(distances(8, newFromOld[4]), 4.05, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(7, newFromOld[4]), newFromOld[9]);
    BOOST_REQUIRE_CLOSE(distances(7, newFromOld[4]), 4.15, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(6, newFromOld[4]), newFromOld[8]);
    BOOST_REQUIRE_CLOSE(distances(6, newFromOld[4]), 4.60, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(5, newFromOld[4]), newFromOld[1]);
    BOOST_REQUIRE_CLOSE(distances(5, newFromOld[4]), 4.70, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(4, newFromOld[4]), newFromOld[2]);
    BOOST_REQUIRE_CLOSE(distances(4, newFromOld[4]), 4.90, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(3, newFromOld[4]), newFromOld[0]);
    BOOST_REQUIRE_CLOSE(distances(3, newFromOld[4]), 5.00, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(2, newFromOld[4]), newFromOld[5]);
    BOOST_REQUIRE_CLOSE(distances(2, newFromOld[4]), 5.27, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(1, newFromOld[4]), newFromOld[7]);
    BOOST_REQUIRE_CLOSE(distances(1, newFromOld[4]), 6.35, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(0, newFromOld[4]), newFromOld[6]);
    BOOST_REQUIRE_CLOSE(distances(0, newFromOld[4]), 7.05, 1e-5);

    // Neighbors of point 5.
    BOOST_REQUIRE_EQUAL(neighbors(9, newFromOld[5]), newFromOld[0]);
    BOOST_REQUIRE_CLOSE(distances(9, newFromOld[5]), 0.27, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(8, newFromOld[5]), newFromOld[2]);
    BOOST_REQUIRE_CLOSE(distances(8, newFromOld[5]), 0.37, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(7, newFromOld[5]), newFromOld[1]);
    BOOST_REQUIRE_CLOSE(distances(7, newFromOld[5]), 0.57, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(6, newFromOld[5]), newFromOld[8]);
    BOOST_REQUIRE_CLOSE(distances(6, newFromOld[5]), 0.67, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(5, newFromOld[5]), newFromOld[7]);
    BOOST_REQUIRE_CLOSE(distances(5, newFromOld[5]), 1.08, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(4, newFromOld[5]), newFromOld[9]);
    BOOST_REQUIRE_CLOSE(distances(4, newFromOld[5]), 1.12, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(3, newFromOld[5]), newFromOld[10]);
    BOOST_REQUIRE_CLOSE(distances(3, newFromOld[5]), 1.22, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(2, newFromOld[5]), newFromOld[3]);
    BOOST_REQUIRE_CLOSE(distances(2, newFromOld[5]), 1.47, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(1, newFromOld[5]), newFromOld[6]);
    BOOST_REQUIRE_CLOSE(distances(1, newFromOld[5]), 1.78, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(0, newFromOld[5]), newFromOld[4]);
    BOOST_REQUIRE_CLOSE(distances(0, newFromOld[5]), 5.27, 1e-5);

    // Neighbors of point 6.
    BOOST_REQUIRE_EQUAL(neighbors(9, newFromOld[6]), newFromOld[7]);
    BOOST_REQUIRE_CLOSE(distances(9, newFromOld[6]), 0.70, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(8, newFromOld[6]), newFromOld[5]);
    BOOST_REQUIRE_CLOSE(distances(8, newFromOld[6]), 1.78, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(7, newFromOld[6]), newFromOld[0]);
    BOOST_REQUIRE_CLOSE(distances(7, newFromOld[6]), 2.05, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(6, newFromOld[6]), newFromOld[2]);
    BOOST_REQUIRE_CLOSE(distances(6, newFromOld[6]), 2.15, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(5, newFromOld[6]), newFromOld[1]);
    BOOST_REQUIRE_CLOSE(distances(5, newFromOld[6]), 2.35, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(4, newFromOld[6]), newFromOld[8]);
    BOOST_REQUIRE_CLOSE(distances(4, newFromOld[6]), 2.45, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(3, newFromOld[6]), newFromOld[9]);
    BOOST_REQUIRE_CLOSE(distances(3, newFromOld[6]), 2.90, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(2, newFromOld[6]), newFromOld[10]);
    BOOST_REQUIRE_CLOSE(distances(2, newFromOld[6]), 3.00, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(1, newFromOld[6]), newFromOld[3]);
    BOOST_REQUIRE_CLOSE(distances(1, newFromOld[6]), 3.25, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(0, newFromOld[6]), newFromOld[4]);
    BOOST_REQUIRE_CLOSE(distances(0, newFromOld[6]), 7.05, 1e-5);

    // Neighbors of point 7.
    BOOST_REQUIRE_EQUAL(neighbors(9, newFromOld[7]), newFromOld[6]);
    BOOST_REQUIRE_CLOSE(distances(9, newFromOld[7]), 0.70, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(8, newFromOld[7]), newFromOld[5]);
    BOOST_REQUIRE_CLOSE(distances(8, newFromOld[7]), 1.08, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(7, newFromOld[7]), newFromOld[0]);
    BOOST_REQUIRE_CLOSE(distances(7, newFromOld[7]), 1.35, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(6, newFromOld[7]), newFromOld[2]);
    BOOST_REQUIRE_CLOSE(distances(6, newFromOld[7]), 1.45, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(5, newFromOld[7]), newFromOld[1]);
    BOOST_REQUIRE_CLOSE(distances(5, newFromOld[7]), 1.65, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(4, newFromOld[7]), newFromOld[8]);
    BOOST_REQUIRE_CLOSE(distances(4, newFromOld[7]), 1.75, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(3, newFromOld[7]), newFromOld[9]);
    BOOST_REQUIRE_CLOSE(distances(3, newFromOld[7]), 2.20, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(2, newFromOld[7]), newFromOld[10]);
    BOOST_REQUIRE_CLOSE(distances(2, newFromOld[7]), 2.30, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(1, newFromOld[7]), newFromOld[3]);
    BOOST_REQUIRE_CLOSE(distances(1, newFromOld[7]), 2.55, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(0, newFromOld[7]), newFromOld[4]);
    BOOST_REQUIRE_CLOSE(distances(0, newFromOld[7]), 6.35, 1e-5);

    // Neighbors of point 8.
    BOOST_REQUIRE_EQUAL(neighbors(9, newFromOld[8]), newFromOld[1]);
    BOOST_REQUIRE_CLOSE(distances(9, newFromOld[8]), 0.10, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(8, newFromOld[8]), newFromOld[2]);
    BOOST_REQUIRE_CLOSE(distances(8, newFromOld[8]), 0.30, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(7, newFromOld[8]), newFromOld[0]);
    BOOST_REQUIRE_CLOSE(distances(7, newFromOld[8]), 0.40, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(6, newFromOld[8]), newFromOld[9]);
    BOOST_REQUIRE_CLOSE(distances(6, newFromOld[8]), 0.45, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(5, newFromOld[8]), newFromOld[10]);
    BOOST_REQUIRE_CLOSE(distances(5, newFromOld[8]), 0.55, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(4, newFromOld[8]), newFromOld[5]);
    BOOST_REQUIRE_CLOSE(distances(4, newFromOld[8]), 0.67, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(3, newFromOld[8]), newFromOld[3]);
    BOOST_REQUIRE_CLOSE(distances(3, newFromOld[8]), 0.80, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(2, newFromOld[8]), newFromOld[7]);
    BOOST_REQUIRE_CLOSE(distances(2, newFromOld[8]), 1.75, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(1, newFromOld[8]), newFromOld[6]);
    BOOST_REQUIRE_CLOSE(distances(1, newFromOld[8]), 2.45, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(0, newFromOld[8]), newFromOld[4]);
    BOOST_REQUIRE_CLOSE(distances(0, newFromOld[8]), 4.60, 1e-5);

    // Neighbors of point 9.
    BOOST_REQUIRE_EQUAL(neighbors(9, newFromOld[9]), newFromOld[10]);
    BOOST_REQUIRE_CLOSE(distances(9, newFromOld[9]), 0.10, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(8, newFromOld[9]), newFromOld[3]);
    BOOST_REQUIRE_CLOSE(distances(8, newFromOld[9]), 0.35, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(7, newFromOld[9]), newFromOld[8]);
    BOOST_REQUIRE_CLOSE(distances(7, newFromOld[9]), 0.45, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(6, newFromOld[9]), newFromOld[1]);
    BOOST_REQUIRE_CLOSE(distances(6, newFromOld[9]), 0.55, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(5, newFromOld[9]), newFromOld[2]);
    BOOST_REQUIRE_CLOSE(distances(5, newFromOld[9]), 0.75, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(4, newFromOld[9]), newFromOld[0]);
    BOOST_REQUIRE_CLOSE(distances(4, newFromOld[9]), 0.85, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(3, newFromOld[9]), newFromOld[5]);
    BOOST_REQUIRE_CLOSE(distances(3, newFromOld[9]), 1.12, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(2, newFromOld[9]), newFromOld[7]);
    BOOST_REQUIRE_CLOSE(distances(2, newFromOld[9]), 2.20, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(1, newFromOld[9]), newFromOld[6]);
    BOOST_REQUIRE_CLOSE(distances(1, newFromOld[9]), 2.90, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(0, newFromOld[9]), newFromOld[4]);
    BOOST_REQUIRE_CLOSE(distances(0, newFromOld[9]), 4.15, 1e-5);

    // Neighbors of point 10.
    BOOST_REQUIRE_EQUAL(neighbors(9, newFromOld[10]), newFromOld[9]);
    BOOST_REQUIRE_CLOSE(distances(9, newFromOld[10]), 0.10, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(8, newFromOld[10]), newFromOld[3]);
    BOOST_REQUIRE_CLOSE(distances(8, newFromOld[10]), 0.25, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(7, newFromOld[10]), newFromOld[8]);
    BOOST_REQUIRE_CLOSE(distances(7, newFromOld[10]), 0.55, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(6, newFromOld[10]), newFromOld[1]);
    BOOST_REQUIRE_CLOSE(distances(6, newFromOld[10]), 0.65, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(5, newFromOld[10]), newFromOld[2]);
    BOOST_REQUIRE_CLOSE(distances(5, newFromOld[10]), 0.85, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(4, newFromOld[10]), newFromOld[0]);
    BOOST_REQUIRE_CLOSE(distances(4, newFromOld[10]), 0.95, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(3, newFromOld[10]), newFromOld[5]);
    BOOST_REQUIRE_CLOSE(distances(3, newFromOld[10]), 1.22, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(2, newFromOld[10]), newFromOld[7]);
    BOOST_REQUIRE_CLOSE(distances(2, newFromOld[10]), 2.30, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(1, newFromOld[10]), newFromOld[6]);
    BOOST_REQUIRE_CLOSE(distances(1, newFromOld[10]), 3.00, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(0, newFromOld[10]), newFromOld[4]);
    BOOST_REQUIRE_CLOSE(distances(0, newFromOld[10]), 4.05, 1e-5);
  }
}

/**
 * Test the dual-tree furthest-neighbors method with the naive method.  This
 * uses both a query and reference dataset.
 *
 * Errors are produced if the results are not identical.
 */
BOOST_AUTO_TEST_CASE(DualTreeVsNaive1)
{
  arma::mat dataset;

  // Hard-coded filename: bad?
  if (!data::Load("test_data_3_1000.csv", dataset))
    BOOST_FAIL("Cannot load test dataset test_data_3_1000.csv!");

  KFN kfn(dataset);

  KFN naive(dataset, NAIVE_MODE);

  arma::Mat<size_t> neighborsTree;
  arma::mat distancesTree;
  kfn.Search(dataset, 15, neighborsTree, distancesTree);

  arma::Mat<size_t> neighborsNaive;
  arma::mat distancesNaive;
  naive.Search(dataset, 15, neighborsNaive, distancesNaive);

  for (size_t i = 0; i < neighborsTree.n_elem; i++)
  {
    BOOST_REQUIRE(neighborsTree[i] == neighborsNaive[i]);
    BOOST_REQUIRE_CLOSE(distancesTree[i], distancesNaive[i], 1e-5);
  }
}

/**
 * Test the dual-tree furthest-neighbors method with the naive method.  This
 * uses only a reference dataset.
 *
 * Errors are produced if the results are not identical.
 */
BOOST_AUTO_TEST_CASE(DualTreeVsNaive2)
{
  arma::mat dataset;

  // Hard-coded filename: bad?
  // Code duplication: also bad!
  if (!data::Load("test_data_3_1000.csv", dataset))
    BOOST_FAIL("Cannot load test dataset test_data_3_1000.csv!");

  KFN kfn(dataset);

  KFN naive(dataset, NAIVE_MODE);

  arma::Mat<size_t> neighborsTree;
  arma::mat distancesTree;
  kfn.Search(15, neighborsTree, distancesTree);

  arma::Mat<size_t> neighborsNaive;
  arma::mat distancesNaive;
  naive.Search(15, neighborsNaive, distancesNaive);

  for (size_t i = 0; i < neighborsTree.n_elem; i++)
  {
    BOOST_REQUIRE_EQUAL(neighborsTree[i], neighborsNaive[i]);
    BOOST_REQUIRE_CLOSE(distancesTree[i], distancesNaive[i], 1e-5);
  }
}

/**
 * Test the single-tree furthest-neighbors method with the naive method.  This
 * uses only a reference dataset.
 *
 * Errors are produced if the results are not identical.
 */
BOOST_AUTO_TEST_CASE(SingleTreeVsNaive)
{
  arma::mat dataset;

  // Hard-coded filename: bad!
  // Code duplication: also bad!
  if (!data::Load("test_data_3_1000.csv", dataset))
    BOOST_FAIL("Cannot load test dataset test_data_3_1000.csv!");

  KFN kfn(dataset, SINGLE_TREE_MODE);

  KFN naive(dataset, NAIVE_MODE);

  arma::Mat<size_t> neighborsTree;
  arma::mat distancesTree;
  kfn.Search(15, neighborsTree, distancesTree);

  arma::Mat<size_t> neighborsNaive;
  arma::mat distancesNaive;
  naive.Search(15, neighborsNaive, distancesNaive);

  for (size_t i = 0; i < neighborsTree.n_elem; i++)
  {
    BOOST_REQUIRE_EQUAL(neighborsTree[i], neighborsNaive[i]);
    BOOST_REQUIRE_CLOSE(distancesTree[i], distancesNaive[i], 1e-5);
  }
}

/**
 * Test the cover tree single-tree furthest-neighbors method against the naive
 * method.  This uses only a random reference dataset.
 *
 * Errors are produced if the results are not identical.
 */
BOOST_AUTO_TEST_CASE(SingleCoverTreeTest)
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
    BOOST_REQUIRE_EQUAL(coverTreeNeighbors[i], naiveNeighbors[i]);
    BOOST_REQUIRE_CLOSE(coverTreeDistances[i], naiveDistances[i], 1e-5);
  }
}

/**
 * Test the cover tree dual-tree furthest neighbors method against the naive
 * method.
 */
BOOST_AUTO_TEST_CASE(DualCoverTreeTest)
{
  arma::mat dataset;
  data::Load("test_data_3_1000.csv", dataset);

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
    BOOST_REQUIRE_EQUAL(coverNeighbors(i), kdNeighbors(i));
    BOOST_REQUIRE_CLOSE(coverDistances(i), kdDistances(i), 1e-5);
  }
}

/**
 * Test the ball tree single-tree furthest-neighbors method against the naive
 * method.  This uses only a random reference dataset.
 *
 * Errors are produced if the results are not identical.
 */
BOOST_AUTO_TEST_CASE(SingleBallTreeTest)
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
    BOOST_REQUIRE_EQUAL(ballTreeNeighbors[i], naiveNeighbors[i]);
    BOOST_REQUIRE_CLOSE(ballTreeDistances[i], naiveDistances[i], 1e-5);
  }
}

/**
 * Test the ball tree dual-tree furthest neighbors method against the naive
 * method.
 */
BOOST_AUTO_TEST_CASE(DualBallTreeTest)
{
  arma::mat dataset;
  data::Load("test_data_3_1000.csv", dataset);

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
    BOOST_REQUIRE_EQUAL(ballNeighbors(i), kdNeighbors(i));
    BOOST_REQUIRE_CLOSE(ballDistances(i), kdDistances(i), 1e-5);
  }
}

BOOST_AUTO_TEST_SUITE_END();
