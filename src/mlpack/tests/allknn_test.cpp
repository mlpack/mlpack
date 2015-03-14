/**
 * @file allknn_test.cpp
 *
 * Test file for AllkNN class.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <mlpack/methods/neighbor_search/unmap.hpp>
#include <mlpack/core/tree/cover_tree.hpp>
#include <mlpack/core/tree/example_tree.hpp>
#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace mlpack::neighbor;
using namespace mlpack::tree;
using namespace mlpack::metric;
using namespace mlpack::bound;

BOOST_AUTO_TEST_SUITE(AllkNNTest);

/**
 * Test that Unmap() works in the dual-tree case (see unmap.hpp).
 */
BOOST_AUTO_TEST_CASE(DualTreeUnmapTest)
{
  std::vector<size_t> refMap;
  refMap.push_back(3);
  refMap.push_back(4);
  refMap.push_back(1);
  refMap.push_back(2);
  refMap.push_back(0);

  std::vector<size_t> queryMap;
  queryMap.push_back(2);
  queryMap.push_back(0);
  queryMap.push_back(4);
  queryMap.push_back(3);
  queryMap.push_back(1);
  queryMap.push_back(5);

  // Now generate some results.  6 queries, 5 references.
  arma::Mat<size_t> neighbors("3 1 2 0 4;"
                              "1 0 2 3 4;"
                              "0 1 2 3 4;"
                              "4 1 0 3 2;"
                              "3 0 4 1 2;"
                              "3 0 4 1 2;");
  neighbors = neighbors.t();

  // Integer distances will work fine here.
  arma::mat distances("3 1 2 0 4;"
                      "1 0 2 3 4;"
                      "0 1 2 3 4;"
                      "4 1 0 3 2;"
                      "3 0 4 1 2;"
                      "3 0 4 1 2;");
  distances = distances.t();

  // This is what the results should be when they are unmapped.
  arma::Mat<size_t> correctNeighbors("4 3 1 2 0;"
                                     "2 3 0 4 1;"
                                     "2 4 1 3 0;"
                                     "0 4 3 2 1;"
                                     "3 4 1 2 0;"
                                     "2 3 0 4 1;");
  correctNeighbors = correctNeighbors.t();

  arma::mat correctDistances("1 0 2 3 4;"
                             "3 0 4 1 2;"
                             "3 1 2 0 4;"
                             "4 1 0 3 2;"
                             "0 1 2 3 4;"
                             "3 0 4 1 2;");
  correctDistances = correctDistances.t();

  // Perform the unmapping.
  arma::Mat<size_t> neighborsOut;
  arma::mat distancesOut;

  Unmap(neighbors, distances, refMap, queryMap, neighborsOut, distancesOut);

  for (size_t i = 0; i < correctNeighbors.n_elem; ++i)
  {
    BOOST_REQUIRE_EQUAL(neighborsOut[i], correctNeighbors[i]);
    BOOST_REQUIRE_CLOSE(distancesOut[i], correctDistances[i], 1e-5);
  }

  // Now try taking the square root.
  Unmap(neighbors, distances, refMap, queryMap, neighborsOut, distancesOut,
      true);

  for (size_t i = 0; i < correctNeighbors.n_elem; ++i)
  {
    BOOST_REQUIRE_EQUAL(neighborsOut[i], correctNeighbors[i]);
    BOOST_REQUIRE_CLOSE(distancesOut[i], sqrt(correctDistances[i]), 1e-5);
  }
}

/**
 * Check that Unmap() works in the single-tree case.
 */
BOOST_AUTO_TEST_CASE(SingleTreeUnmapTest)
{
  std::vector<size_t> refMap;
  refMap.push_back(3);
  refMap.push_back(4);
  refMap.push_back(1);
  refMap.push_back(2);
  refMap.push_back(0);

  // Now generate some results.  6 queries, 5 references.
  arma::Mat<size_t> neighbors("3 1 2 0 4;"
                              "1 0 2 3 4;"
                              "0 1 2 3 4;"
                              "4 1 0 3 2;"
                              "3 0 4 1 2;"
                              "3 0 4 1 2;");
  neighbors = neighbors.t();

  // Integer distances will work fine here.
  arma::mat distances("3 1 2 0 4;"
                      "1 0 2 3 4;"
                      "0 1 2 3 4;"
                      "4 1 0 3 2;"
                      "3 0 4 1 2;"
                      "3 0 4 1 2;");
  distances = distances.t();

  // This is what the results should be when they are unmapped.
  arma::Mat<size_t> correctNeighbors("2 4 1 3 0;"
                                     "4 3 1 2 0;"
                                     "3 4 1 2 0;"
                                     "0 4 3 2 1;"
                                     "2 3 0 4 1;"
                                     "2 3 0 4 1;");
  correctNeighbors = correctNeighbors.t();

  arma::mat correctDistances = distances;

  // Perform the unmapping.
  arma::Mat<size_t> neighborsOut;
  arma::mat distancesOut;

  Unmap(neighbors, distances, refMap, neighborsOut, distancesOut);

  for (size_t i = 0; i < correctNeighbors.n_elem; ++i)
  {
    BOOST_REQUIRE_EQUAL(neighborsOut[i], correctNeighbors[i]);
    BOOST_REQUIRE_CLOSE(distancesOut[i], correctDistances[i], 1e-5);
  }

  // Now try taking the square root.
  Unmap(neighbors, distances, refMap, neighborsOut, distancesOut, true);

  for (size_t i = 0; i < correctNeighbors.n_elem; ++i)
  {
    BOOST_REQUIRE_EQUAL(neighborsOut[i], correctNeighbors[i]);
    BOOST_REQUIRE_CLOSE(distancesOut[i], sqrt(correctDistances[i]), 1e-5);
  }
}

/**
 * Simple nearest-neighbors test with small, synthetic dataset.  This is an
 * exhaustive test, which checks that each method for performing the calculation
 * (dual-tree, single-tree, naive) produces the correct results.  An
 * eleven-point dataset and the ten nearest neighbors are taken.  The dataset is
 * in one dimension for simplicity -- the correct functionality of distance
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

  typedef BinarySpaceTree<HRectBound<2>,
      NeighborSearchStat<NearestNeighborSort> > TreeType;

  // We will loop through three times, one for each method of performing the
  // calculation.
  arma::mat dataMutable = data;
  std::vector<size_t> oldFromNew;
  std::vector<size_t> newFromOld;
  TreeType* tree = new TreeType(dataMutable, oldFromNew, newFromOld, 1);
  for (int i = 0; i < 3; i++)
  {
    AllkNN* allknn;

    switch (i)
    {
      case 0: // Use the dual-tree method.
        allknn = new AllkNN(tree, dataMutable, false);
        break;
      case 1: // Use the single-tree method.
        allknn = new AllkNN(tree, dataMutable, true);
        break;
      case 2: // Use the naive method.
        allknn = new AllkNN(dataMutable, true);
        break;
    }

    // Now perform the actual calculation.
    arma::Mat<size_t> neighbors;
    arma::mat distances;
    allknn->Search(10, neighbors, distances);

    // Now the exhaustive check for correctness.  This will be long.  We must
    // also remember that the distances returned are squared distances.  As a
    // result, distance comparisons are written out as (distance * distance) for
    // readability.

    // Neighbors of point 0.
    BOOST_REQUIRE_EQUAL(neighbors(0, newFromOld[0]), newFromOld[2]);
    BOOST_REQUIRE_CLOSE(distances(0, newFromOld[0]), 0.10, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(1, newFromOld[0]), newFromOld[5]);
    BOOST_REQUIRE_CLOSE(distances(1, newFromOld[0]), 0.27, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(2, newFromOld[0]), newFromOld[1]);
    BOOST_REQUIRE_CLOSE(distances(2, newFromOld[0]), 0.30, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(3, newFromOld[0]), newFromOld[8]);
    BOOST_REQUIRE_CLOSE(distances(3, newFromOld[0]), 0.40, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(4, newFromOld[0]), newFromOld[9]);
    BOOST_REQUIRE_CLOSE(distances(4, newFromOld[0]), 0.85, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(5, newFromOld[0]), newFromOld[10]);
    BOOST_REQUIRE_CLOSE(distances(5, newFromOld[0]), 0.95, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(6, newFromOld[0]), newFromOld[3]);
    BOOST_REQUIRE_CLOSE(distances(6, newFromOld[0]), 1.20, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(7, newFromOld[0]), newFromOld[7]);
    BOOST_REQUIRE_CLOSE(distances(7, newFromOld[0]), 1.35, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(8, newFromOld[0]), newFromOld[6]);
    BOOST_REQUIRE_CLOSE(distances(8, newFromOld[0]), 2.05, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(9, newFromOld[0]), newFromOld[4]);
    BOOST_REQUIRE_CLOSE(distances(9, newFromOld[0]), 5.00, 1e-5);

    // Neighbors of point 1.
    BOOST_REQUIRE_EQUAL(neighbors(0, newFromOld[1]), newFromOld[8]);
    BOOST_REQUIRE_CLOSE(distances(0, newFromOld[1]), 0.10, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(1, newFromOld[1]), newFromOld[2]);
    BOOST_REQUIRE_CLOSE(distances(1, newFromOld[1]), 0.20, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(2, newFromOld[1]), newFromOld[0]);
    BOOST_REQUIRE_CLOSE(distances(2, newFromOld[1]), 0.30, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(3, newFromOld[1]), newFromOld[9]);
    BOOST_REQUIRE_CLOSE(distances(3, newFromOld[1]), 0.55, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(4, newFromOld[1]), newFromOld[5]);
    BOOST_REQUIRE_CLOSE(distances(4, newFromOld[1]), 0.57, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(5, newFromOld[1]), newFromOld[10]);
    BOOST_REQUIRE_CLOSE(distances(5, newFromOld[1]), 0.65, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(6, newFromOld[1]), newFromOld[3]);
    BOOST_REQUIRE_CLOSE(distances(6, newFromOld[1]), 0.90, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(7, newFromOld[1]), newFromOld[7]);
    BOOST_REQUIRE_CLOSE(distances(7, newFromOld[1]), 1.65, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(8, newFromOld[1]), newFromOld[6]);
    BOOST_REQUIRE_CLOSE(distances(8, newFromOld[1]), 2.35, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(9, newFromOld[1]), newFromOld[4]);
    BOOST_REQUIRE_CLOSE(distances(9, newFromOld[1]), 4.70, 1e-5);

    // Neighbors of point 2.
    BOOST_REQUIRE_EQUAL(neighbors(0, newFromOld[2]), newFromOld[0]);
    BOOST_REQUIRE_CLOSE(distances(0, newFromOld[2]), 0.10, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(1, newFromOld[2]), newFromOld[1]);
    BOOST_REQUIRE_CLOSE(distances(1, newFromOld[2]), 0.20, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(2, newFromOld[2]), newFromOld[8]);
    BOOST_REQUIRE_CLOSE(distances(2, newFromOld[2]), 0.30, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(3, newFromOld[2]), newFromOld[5]);
    BOOST_REQUIRE_CLOSE(distances(3, newFromOld[2]), 0.37, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(4, newFromOld[2]), newFromOld[9]);
    BOOST_REQUIRE_CLOSE(distances(4, newFromOld[2]), 0.75, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(5, newFromOld[2]), newFromOld[10]);
    BOOST_REQUIRE_CLOSE(distances(5, newFromOld[2]), 0.85, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(6, newFromOld[2]), newFromOld[3]);
    BOOST_REQUIRE_CLOSE(distances(6, newFromOld[2]), 1.10, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(7, newFromOld[2]), newFromOld[7]);
    BOOST_REQUIRE_CLOSE(distances(7, newFromOld[2]), 1.45, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(8, newFromOld[2]), newFromOld[6]);
    BOOST_REQUIRE_CLOSE(distances(8, newFromOld[2]), 2.15, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(9, newFromOld[2]), newFromOld[4]);
    BOOST_REQUIRE_CLOSE(distances(9, newFromOld[2]), 4.90, 1e-5);

    // Neighbors of point 3.
    BOOST_REQUIRE_EQUAL(neighbors(0, newFromOld[3]), newFromOld[10]);
    BOOST_REQUIRE_CLOSE(distances(0, newFromOld[3]), 0.25, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(1, newFromOld[3]), newFromOld[9]);
    BOOST_REQUIRE_CLOSE(distances(1, newFromOld[3]), 0.35, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(2, newFromOld[3]), newFromOld[8]);
    BOOST_REQUIRE_CLOSE(distances(2, newFromOld[3]), 0.80, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(3, newFromOld[3]), newFromOld[1]);
    BOOST_REQUIRE_CLOSE(distances(3, newFromOld[3]), 0.90, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(4, newFromOld[3]), newFromOld[2]);
    BOOST_REQUIRE_CLOSE(distances(4, newFromOld[3]), 1.10, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(5, newFromOld[3]), newFromOld[0]);
    BOOST_REQUIRE_CLOSE(distances(5, newFromOld[3]), 1.20, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(6, newFromOld[3]), newFromOld[5]);
    BOOST_REQUIRE_CLOSE(distances(6, newFromOld[3]), 1.47, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(7, newFromOld[3]), newFromOld[7]);
    BOOST_REQUIRE_CLOSE(distances(7, newFromOld[3]), 2.55, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(8, newFromOld[3]), newFromOld[6]);
    BOOST_REQUIRE_CLOSE(distances(8, newFromOld[3]), 3.25, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(9, newFromOld[3]), newFromOld[4]);
    BOOST_REQUIRE_CLOSE(distances(9, newFromOld[3]), 3.80, 1e-5);

    // Neighbors of point 4.
    BOOST_REQUIRE_EQUAL(neighbors(0, newFromOld[4]), newFromOld[3]);
    BOOST_REQUIRE_CLOSE(distances(0, newFromOld[4]), 3.80, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(1, newFromOld[4]), newFromOld[10]);
    BOOST_REQUIRE_CLOSE(distances(1, newFromOld[4]), 4.05, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(2, newFromOld[4]), newFromOld[9]);
    BOOST_REQUIRE_CLOSE(distances(2, newFromOld[4]), 4.15, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(3, newFromOld[4]), newFromOld[8]);
    BOOST_REQUIRE_CLOSE(distances(3, newFromOld[4]), 4.60, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(4, newFromOld[4]), newFromOld[1]);
    BOOST_REQUIRE_CLOSE(distances(4, newFromOld[4]), 4.70, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(5, newFromOld[4]), newFromOld[2]);
    BOOST_REQUIRE_CLOSE(distances(5, newFromOld[4]), 4.90, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(6, newFromOld[4]), newFromOld[0]);
    BOOST_REQUIRE_CLOSE(distances(6, newFromOld[4]), 5.00, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(7, newFromOld[4]), newFromOld[5]);
    BOOST_REQUIRE_CLOSE(distances(7, newFromOld[4]), 5.27, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(8, newFromOld[4]), newFromOld[7]);
    BOOST_REQUIRE_CLOSE(distances(8, newFromOld[4]), 6.35, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(9, newFromOld[4]), newFromOld[6]);
    BOOST_REQUIRE_CLOSE(distances(9, newFromOld[4]), 7.05, 1e-5);

    // Neighbors of point 5.
    BOOST_REQUIRE_EQUAL(neighbors(0, newFromOld[5]), newFromOld[0]);
    BOOST_REQUIRE_CLOSE(distances(0, newFromOld[5]), 0.27, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(1, newFromOld[5]), newFromOld[2]);
    BOOST_REQUIRE_CLOSE(distances(1, newFromOld[5]), 0.37, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(2, newFromOld[5]), newFromOld[1]);
    BOOST_REQUIRE_CLOSE(distances(2, newFromOld[5]), 0.57, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(3, newFromOld[5]), newFromOld[8]);
    BOOST_REQUIRE_CLOSE(distances(3, newFromOld[5]), 0.67, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(4, newFromOld[5]), newFromOld[7]);
    BOOST_REQUIRE_CLOSE(distances(4, newFromOld[5]), 1.08, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(5, newFromOld[5]), newFromOld[9]);
    BOOST_REQUIRE_CLOSE(distances(5, newFromOld[5]), 1.12, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(6, newFromOld[5]), newFromOld[10]);
    BOOST_REQUIRE_CLOSE(distances(6, newFromOld[5]), 1.22, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(7, newFromOld[5]), newFromOld[3]);
    BOOST_REQUIRE_CLOSE(distances(7, newFromOld[5]), 1.47, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(8, newFromOld[5]), newFromOld[6]);
    BOOST_REQUIRE_CLOSE(distances(8, newFromOld[5]), 1.78, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(9, newFromOld[5]), newFromOld[4]);
    BOOST_REQUIRE_CLOSE(distances(9, newFromOld[5]), 5.27, 1e-5);

    // Neighbors of point 6.
    BOOST_REQUIRE_EQUAL(neighbors(0, newFromOld[6]), newFromOld[7]);
    BOOST_REQUIRE_CLOSE(distances(0, newFromOld[6]), 0.70, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(1, newFromOld[6]), newFromOld[5]);
    BOOST_REQUIRE_CLOSE(distances(1, newFromOld[6]), 1.78, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(2, newFromOld[6]), newFromOld[0]);
    BOOST_REQUIRE_CLOSE(distances(2, newFromOld[6]), 2.05, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(3, newFromOld[6]), newFromOld[2]);
    BOOST_REQUIRE_CLOSE(distances(3, newFromOld[6]), 2.15, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(4, newFromOld[6]), newFromOld[1]);
    BOOST_REQUIRE_CLOSE(distances(4, newFromOld[6]), 2.35, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(5, newFromOld[6]), newFromOld[8]);
    BOOST_REQUIRE_CLOSE(distances(5, newFromOld[6]), 2.45, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(6, newFromOld[6]), newFromOld[9]);
    BOOST_REQUIRE_CLOSE(distances(6, newFromOld[6]), 2.90, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(7, newFromOld[6]), newFromOld[10]);
    BOOST_REQUIRE_CLOSE(distances(7, newFromOld[6]), 3.00, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(8, newFromOld[6]), newFromOld[3]);
    BOOST_REQUIRE_CLOSE(distances(8, newFromOld[6]), 3.25, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(9, newFromOld[6]), newFromOld[4]);
    BOOST_REQUIRE_CLOSE(distances(9, newFromOld[6]), 7.05, 1e-5);

    // Neighbors of point 7.
    BOOST_REQUIRE_EQUAL(neighbors(0, newFromOld[7]), newFromOld[6]);
    BOOST_REQUIRE_CLOSE(distances(0, newFromOld[7]), 0.70, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(1, newFromOld[7]), newFromOld[5]);
    BOOST_REQUIRE_CLOSE(distances(1, newFromOld[7]), 1.08, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(2, newFromOld[7]), newFromOld[0]);
    BOOST_REQUIRE_CLOSE(distances(2, newFromOld[7]), 1.35, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(3, newFromOld[7]), newFromOld[2]);
    BOOST_REQUIRE_CLOSE(distances(3, newFromOld[7]), 1.45, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(4, newFromOld[7]), newFromOld[1]);
    BOOST_REQUIRE_CLOSE(distances(4, newFromOld[7]), 1.65, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(5, newFromOld[7]), newFromOld[8]);
    BOOST_REQUIRE_CLOSE(distances(5, newFromOld[7]), 1.75, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(6, newFromOld[7]), newFromOld[9]);
    BOOST_REQUIRE_CLOSE(distances(6, newFromOld[7]), 2.20, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(7, newFromOld[7]), newFromOld[10]);
    BOOST_REQUIRE_CLOSE(distances(7, newFromOld[7]), 2.30, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(8, newFromOld[7]), newFromOld[3]);
    BOOST_REQUIRE_CLOSE(distances(8, newFromOld[7]), 2.55, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(9, newFromOld[7]), newFromOld[4]);
    BOOST_REQUIRE_CLOSE(distances(9, newFromOld[7]), 6.35, 1e-5);

    // Neighbors of point 8.
    BOOST_REQUIRE_EQUAL(neighbors(0, newFromOld[8]), newFromOld[1]);
    BOOST_REQUIRE_CLOSE(distances(0, newFromOld[8]), 0.10, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(1, newFromOld[8]), newFromOld[2]);
    BOOST_REQUIRE_CLOSE(distances(1, newFromOld[8]), 0.30, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(2, newFromOld[8]), newFromOld[0]);
    BOOST_REQUIRE_CLOSE(distances(2, newFromOld[8]), 0.40, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(3, newFromOld[8]), newFromOld[9]);
    BOOST_REQUIRE_CLOSE(distances(3, newFromOld[8]), 0.45, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(4, newFromOld[8]), newFromOld[10]);
    BOOST_REQUIRE_CLOSE(distances(4, newFromOld[8]), 0.55, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(5, newFromOld[8]), newFromOld[5]);
    BOOST_REQUIRE_CLOSE(distances(5, newFromOld[8]), 0.67, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(6, newFromOld[8]), newFromOld[3]);
    BOOST_REQUIRE_CLOSE(distances(6, newFromOld[8]), 0.80, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(7, newFromOld[8]), newFromOld[7]);
    BOOST_REQUIRE_CLOSE(distances(7, newFromOld[8]), 1.75, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(8, newFromOld[8]), newFromOld[6]);
    BOOST_REQUIRE_CLOSE(distances(8, newFromOld[8]), 2.45, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(9, newFromOld[8]), newFromOld[4]);
    BOOST_REQUIRE_CLOSE(distances(9, newFromOld[8]), 4.60, 1e-5);

    // Neighbors of point 9.
    BOOST_REQUIRE_EQUAL(neighbors(0, newFromOld[9]), newFromOld[10]);
    BOOST_REQUIRE_CLOSE(distances(0, newFromOld[9]), 0.10, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(1, newFromOld[9]), newFromOld[3]);
    BOOST_REQUIRE_CLOSE(distances(1, newFromOld[9]), 0.35, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(2, newFromOld[9]), newFromOld[8]);
    BOOST_REQUIRE_CLOSE(distances(2, newFromOld[9]), 0.45, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(3, newFromOld[9]), newFromOld[1]);
    BOOST_REQUIRE_CLOSE(distances(3, newFromOld[9]), 0.55, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(4, newFromOld[9]), newFromOld[2]);
    BOOST_REQUIRE_CLOSE(distances(4, newFromOld[9]), 0.75, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(5, newFromOld[9]), newFromOld[0]);
    BOOST_REQUIRE_CLOSE(distances(5, newFromOld[9]), 0.85, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(6, newFromOld[9]), newFromOld[5]);
    BOOST_REQUIRE_CLOSE(distances(6, newFromOld[9]), 1.12, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(7, newFromOld[9]), newFromOld[7]);
    BOOST_REQUIRE_CLOSE(distances(7, newFromOld[9]), 2.20, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(8, newFromOld[9]), newFromOld[6]);
    BOOST_REQUIRE_CLOSE(distances(8, newFromOld[9]), 2.90, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(9, newFromOld[9]), newFromOld[4]);
    BOOST_REQUIRE_CLOSE(distances(9, newFromOld[9]), 4.15, 1e-5);

    // Neighbors of point 10.
    BOOST_REQUIRE_EQUAL(neighbors(0, newFromOld[10]), newFromOld[9]);
    BOOST_REQUIRE_CLOSE(distances(0, newFromOld[10]), 0.10, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(1, newFromOld[10]), newFromOld[3]);
    BOOST_REQUIRE_CLOSE(distances(1, newFromOld[10]), 0.25, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(2, newFromOld[10]), newFromOld[8]);
    BOOST_REQUIRE_CLOSE(distances(2, newFromOld[10]), 0.55, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(3, newFromOld[10]), newFromOld[1]);
    BOOST_REQUIRE_CLOSE(distances(3, newFromOld[10]), 0.65, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(4, newFromOld[10]), newFromOld[2]);
    BOOST_REQUIRE_CLOSE(distances(4, newFromOld[10]), 0.85, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(5, newFromOld[10]), newFromOld[0]);
    BOOST_REQUIRE_CLOSE(distances(5, newFromOld[10]), 0.95, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(6, newFromOld[10]), newFromOld[5]);
    BOOST_REQUIRE_CLOSE(distances(6, newFromOld[10]), 1.22, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(7, newFromOld[10]), newFromOld[7]);
    BOOST_REQUIRE_CLOSE(distances(7, newFromOld[10]), 2.30, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(8, newFromOld[10]), newFromOld[6]);
    BOOST_REQUIRE_CLOSE(distances(8, newFromOld[10]), 3.00, 1e-5);
    BOOST_REQUIRE_EQUAL(neighbors(9, newFromOld[10]), newFromOld[4]);
    BOOST_REQUIRE_CLOSE(distances(9, newFromOld[10]), 4.05, 1e-5);

    // Clean the memory.
    delete allknn;
  }

  // Delete the tree.
  delete tree;
}

/**
 * Test the dual-tree nearest-neighbors method with the naive method.  This
 * uses both a query and reference dataset.
 *
 * Errors are produced if the results are not identical.
 */
BOOST_AUTO_TEST_CASE(DualTreeVsNaive1)
{
  arma::mat dataForTree;

  // Hard-coded filename: bad!
  if (!data::Load("test_data_3_1000.csv", dataForTree))
    BOOST_FAIL("Cannot load test dataset test_data_3_1000.csv!");

  // Set up matrices to work with.
  arma::mat dualQuery(dataForTree);
  arma::mat dualReferences(dataForTree);
  arma::mat naiveQuery(dataForTree);
  arma::mat naiveReferences(dataForTree);

  AllkNN allknn(dualQuery, dualReferences);

  AllkNN naive(naiveQuery, naiveReferences, true);

  arma::Mat<size_t> resultingNeighborsTree;
  arma::mat distancesTree;
  allknn.Search(15, resultingNeighborsTree, distancesTree);

  arma::Mat<size_t> resultingNeighborsNaive;
  arma::mat distancesNaive;
  naive.Search(15, resultingNeighborsNaive, distancesNaive);

  for (size_t i = 0; i < resultingNeighborsTree.n_elem; i++)
  {
    BOOST_REQUIRE(resultingNeighborsTree(i) == resultingNeighborsNaive(i));
    BOOST_REQUIRE_CLOSE(distancesTree(i), distancesNaive(i), 1e-5);
  }
}

/**
 * Test the dual-tree nearest-neighbors method with the naive method.  This uses
 * only a reference dataset.
 *
 * Errors are produced if the results are not identical.
 */
BOOST_AUTO_TEST_CASE(DualTreeVsNaive2)
{
  arma::mat dataForTree;

  // Hard-coded filename: bad!
  // Code duplication: also bad!
  if (!data::Load("test_data_3_1000.csv", dataForTree))
    BOOST_FAIL("Cannot load test dataset test_data_3_1000.csv!");

  // Set up matrices to work with (may not be necessary with no ALIAS_MATRIX?).
  arma::mat dualQuery(dataForTree);
  arma::mat naiveQuery(dataForTree);

  AllkNN allknn(dualQuery);

  // Set naive mode.
  AllkNN naive(naiveQuery, true);

  arma::Mat<size_t> resultingNeighborsTree;
  arma::mat distancesTree;
  allknn.Search(15, resultingNeighborsTree, distancesTree);

  arma::Mat<size_t> resultingNeighborsNaive;
  arma::mat distancesNaive;
  naive.Search(15, resultingNeighborsNaive, distancesNaive);

  for (size_t i = 0; i < resultingNeighborsTree.n_elem; i++)
  {
    BOOST_REQUIRE(resultingNeighborsTree[i] == resultingNeighborsNaive[i]);
    BOOST_REQUIRE_CLOSE(distancesTree[i], distancesNaive[i], 1e-5);
  }
}

/**
 * Test the single-tree nearest-neighbors method with the naive method.  This
 * uses only a reference dataset.
 *
 * Errors are produced if the results are not identical.
 */
BOOST_AUTO_TEST_CASE(SingleTreeVsNaive)
{
  arma::mat dataForTree;

  // Hard-coded filename: bad!
  // Code duplication: also bad!
  if (!data::Load("test_data_3_1000.csv", dataForTree))
    BOOST_FAIL("Cannot load test dataset test_data_3_1000.csv!");

  // Set up matrices to work with (may not be necessary with no ALIAS_MATRIX?).
  arma::mat singleQuery(dataForTree);
  arma::mat naiveQuery(dataForTree);

  AllkNN allknn(singleQuery, false, true);

  // Set up computation for naive mode.
  AllkNN naive(naiveQuery, true);

  arma::Mat<size_t> resultingNeighborsTree;
  arma::mat distancesTree;
  allknn.Search(15, resultingNeighborsTree, distancesTree);

  arma::Mat<size_t> resultingNeighborsNaive;
  arma::mat distancesNaive;
  naive.Search(15, resultingNeighborsNaive, distancesNaive);

  for (size_t i = 0; i < resultingNeighborsTree.n_elem; i++)
  {
    BOOST_REQUIRE(resultingNeighborsTree[i] == resultingNeighborsNaive[i]);
    BOOST_REQUIRE_CLOSE(distancesTree[i], distancesNaive[i], 1e-5);
  }
}

/**
 * Test the cover tree single-tree nearest-neighbors method against the naive
 * method.  This uses only a random reference dataset.
 *
 * Errors are produced if the results are not identical.
 */
BOOST_AUTO_TEST_CASE(SingleCoverTreeTest)
{
  arma::mat data;
  data.randu(75, 1000); // 75 dimensional, 1000 points.

  arma::mat naiveQuery(data); // For naive AllkNN.

  CoverTree<LMetric<2>, FirstPointIsRoot,
      NeighborSearchStat<NearestNeighborSort> > tree = CoverTree<LMetric<2>,
      FirstPointIsRoot, NeighborSearchStat<NearestNeighborSort> >(data);

  NeighborSearch<NearestNeighborSort, LMetric<2>, CoverTree<LMetric<2>,
      FirstPointIsRoot, NeighborSearchStat<NearestNeighborSort> > >
      coverTreeSearch(&tree, data, true);

  AllkNN naive(naiveQuery, true);

  arma::Mat<size_t> coverTreeNeighbors;
  arma::mat coverTreeDistances;
  coverTreeSearch.Search(15, coverTreeNeighbors, coverTreeDistances);

  arma::Mat<size_t> naiveNeighbors;
  arma::mat naiveDistances;
  naive.Search(15, naiveNeighbors, naiveDistances);

  for (size_t i = 0; i < coverTreeNeighbors.n_elem; ++i)
  {
    BOOST_REQUIRE_EQUAL(coverTreeNeighbors[i], naiveNeighbors[i]);
    BOOST_REQUIRE_CLOSE(coverTreeDistances[i], naiveDistances[i], 1e-5);
  }
}

/**
 * Test the cover tree dual-tree nearest neighbors method against the naive
 * method.
 */
BOOST_AUTO_TEST_CASE(DualCoverTreeTest)
{
  arma::mat dataset;
  data::Load("test_data_3_1000.csv", dataset);

  arma::mat kdtreeData(dataset);

  AllkNN tree(kdtreeData);

  arma::Mat<size_t> kdNeighbors;
  arma::mat kdDistances;
  tree.Search(5, kdNeighbors, kdDistances);

  CoverTree<LMetric<2, true>, FirstPointIsRoot,
      NeighborSearchStat<NearestNeighborSort> > referenceTree = CoverTree<
      LMetric<2, true>, FirstPointIsRoot,
      NeighborSearchStat<NearestNeighborSort> >(dataset);

  NeighborSearch<NearestNeighborSort, LMetric<2, true>,
      CoverTree<LMetric<2, true>, FirstPointIsRoot,
      NeighborSearchStat<NearestNeighborSort> > >
      coverTreeSearch(&referenceTree, dataset);

  arma::Mat<size_t> coverNeighbors;
  arma::mat coverDistances;
  coverTreeSearch.Search(5, coverNeighbors, coverDistances);

  for (size_t i = 0; i < coverNeighbors.n_elem; ++i)
  {
    BOOST_REQUIRE_EQUAL(coverNeighbors(i), kdNeighbors(i));
    BOOST_REQUIRE_CLOSE(coverDistances(i), kdDistances(i), 1e-5);
  }
}

/**
 * Test the ball tree single-tree nearest-neighbors method against the naive
 * method.  This uses only a random reference dataset.
 *
 * Errors are produced if the results are not identical.
 */
BOOST_AUTO_TEST_CASE(SingleBallTreeTest)
{
  arma::mat data;
  data.randu(75, 1000); // 75 dimensional, 1000 points.

  typedef BinarySpaceTree<BallBound<arma::vec, LMetric<2, true> >,
      NeighborSearchStat<NearestNeighborSort> > TreeType;
  TreeType tree = TreeType(data);

  // BinarySpaceTree modifies data. Use modified data to maintain the
  // correspondance between points in the dataset for both methods. The order of
  // query points in both methods should be same.
  arma::mat naiveQuery(data); // For naive AllkNN.

  NeighborSearch<NearestNeighborSort, LMetric<2>, TreeType>
      ballTreeSearch(&tree, data, true);

  AllkNN naive(naiveQuery, true);

  arma::Mat<size_t> ballTreeNeighbors;
  arma::mat ballTreeDistances;
  ballTreeSearch.Search(1, ballTreeNeighbors, ballTreeDistances);

  arma::Mat<size_t> naiveNeighbors;
  arma::mat naiveDistances;
  naive.Search(1, naiveNeighbors, naiveDistances);

  for (size_t i = 0; i < ballTreeNeighbors.n_elem; ++i)
  {
    BOOST_REQUIRE_EQUAL(ballTreeNeighbors[i], naiveNeighbors[i]);
    BOOST_REQUIRE_CLOSE(ballTreeDistances[i], naiveDistances[i], 1e-5);
  }
}

/**
 * Test the ball tree dual-tree nearest neighbors method against the naive
 * method.
 */
BOOST_AUTO_TEST_CASE(DualBallTreeTest)
{
  arma::mat dataset;
  data::Load("test_data_3_1000.csv", dataset);

  arma::mat kdtreeData(dataset);

  AllkNN tree(kdtreeData);

  arma::Mat<size_t> kdNeighbors;
  arma::mat kdDistances;
  tree.Search(5, kdNeighbors, kdDistances);

  NeighborSearch<NearestNeighborSort, LMetric<2, true>,
      BinarySpaceTree<BallBound<arma::vec, LMetric<2, true> >,
      NeighborSearchStat<NearestNeighborSort> > >
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

// Make sure sparse nearest neighbors works with kd trees.
BOOST_AUTO_TEST_CASE(SparseAllkNNKDTreeTest)
{
  typedef BinarySpaceTree<HRectBound<2>,
      NeighborSearchStat<NearestNeighborSort>, arma::sp_mat> SparseKDTree;

  // The dimensionality of these datasets must be high so that the probability
  // of a completely empty point is very low.  In this case, with dimensionality
  // 70, the probability of all 70 dimensions being zero is 0.8^70 = 1.65e-7 in
  // the reference set and 0.9^70 = 6.27e-4 in the query set.
  arma::sp_mat queryDataset;
  queryDataset.sprandu(70, 500, 0.2);
  arma::sp_mat referenceDataset;
  referenceDataset.sprandu(70, 800, 0.1);
  arma::mat denseQuery(queryDataset);
  arma::mat denseReference(referenceDataset);

  typedef NeighborSearch<NearestNeighborSort, EuclideanDistance, SparseKDTree>
      SparseAllkNN;

  SparseAllkNN a(queryDataset, referenceDataset);
  AllkNN naive(denseQuery, denseReference, true);

  arma::mat sparseDistances;
  arma::Mat<size_t> sparseNeighbors;
  a.Search(10, sparseNeighbors, sparseDistances);

  arma::mat naiveDistances;
  arma::Mat<size_t> naiveNeighbors;
  naive.Search(10, naiveNeighbors, naiveDistances);

  for (size_t i = 0; i < naiveNeighbors.n_cols; ++i)
  {
    for (size_t j = 0; j < naiveNeighbors.n_rows; ++j)
    {
      BOOST_REQUIRE_EQUAL(naiveNeighbors(j, i), sparseNeighbors(j, i));
      BOOST_REQUIRE_CLOSE(naiveDistances(j, i), sparseDistances(j, i), 1e-5);
    }
  }
}

/*
BOOST_AUTO_TEST_CASE(SparseAllkNNCoverTreeTest)
{
  typedef CoverTree<LMetric<2, true>, FirstPointIsRoot,
      NeighborSearchStat<NearestNeighborSort>, arma::sp_mat> SparseCoverTree;

  // The dimensionality of these datasets must be high so that the probability
  // of a completely empty point is very low.  In this case, with dimensionality
  // 70, the probability of all 70 dimensions being zero is 0.8^70 = 1.65e-7 in
  // the reference set and 0.9^70 = 6.27e-4 in the query set.
  arma::sp_mat queryDataset;
  queryDataset.sprandu(50, 5000, 0.2);
  arma::sp_mat referenceDataset;
  referenceDataset.sprandu(50, 8000, 0.1);
  arma::mat denseQuery(queryDataset);
  arma::mat denseReference(referenceDataset);

  typedef NeighborSearch<NearestNeighborSort, EuclideanDistance,
      SparseCoverTree> SparseAllkNN;

  arma::mat sparseDistances;
  arma::Mat<size_t> sparseNeighbors;
  a.Search(10, sparseNeighbors, sparseDistances);

  arma::mat naiveDistances;
  arma::Mat<size_t> naiveNeighbors;
  naive.Search(10, naiveNeighbors, naiveDistances);

  for (size_t i = 0; i < naiveNeighbors.n_cols; ++i)
  {
    for (size_t j = 0; j < naiveNeighbors.n_rows; ++j)
    {
      BOOST_REQUIRE_EQUAL(naiveNeighbors(j, i), sparseNeighbors(j, i));
      BOOST_REQUIRE_CLOSE(naiveDistances(j, i), sparseDistances(j, i), 1e-5);
    }
  }
}
*/

BOOST_AUTO_TEST_SUITE_END();
