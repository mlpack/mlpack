/**
 * @file knn_test.cpp
 *
 * Test file for KNN class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <mlpack/methods/neighbor_search/unmap.hpp>
#include <mlpack/methods/neighbor_search/ns_model.hpp>
#include <mlpack/core/tree/cover_tree.hpp>
#include <mlpack/core/tree/example_tree.hpp>
#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::neighbor;
using namespace mlpack::tree;
using namespace mlpack::metric;
using namespace mlpack::bound;

BOOST_AUTO_TEST_SUITE(KNNTest);

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
 * Test that an empty KNN object will throw exceptions when Search() is
 * called.
 */
BOOST_AUTO_TEST_CASE(EmptySearchTest)
{
  KNN empty;

  arma::mat dataset = arma::randu<arma::mat>(5, 100);
  KNN::Tree queryTree(dataset);
  arma::Mat<size_t> neighbors;
  arma::mat distances;

  BOOST_REQUIRE_THROW(empty.Search(dataset, 5, neighbors, distances),
      std::invalid_argument);
  BOOST_REQUIRE_THROW(empty.Search(5, neighbors, distances),
      std::invalid_argument);
  BOOST_REQUIRE_THROW(empty.Search(queryTree, 5, neighbors, distances),
      std::invalid_argument);
}

/**
 * Test that when training is performed, the results are the same.
 */
BOOST_AUTO_TEST_CASE(TrainTest)
{
  KNN empty;

  arma::mat dataset = arma::randu<arma::mat>(5, 100);
  KNN baseline(dataset);

  arma::Mat<size_t> neighbors, baselineNeighbors;
  arma::mat distances, baselineDistances;

  empty.Train(dataset);

  empty.Search(5, neighbors, distances);
  baseline.Search(5, baselineNeighbors, baselineDistances);

  BOOST_REQUIRE_EQUAL(neighbors.n_rows, baselineNeighbors.n_rows);
  BOOST_REQUIRE_EQUAL(neighbors.n_cols, baselineNeighbors.n_cols);
  BOOST_REQUIRE_EQUAL(distances.n_rows, baselineDistances.n_rows);
  BOOST_REQUIRE_EQUAL(distances.n_cols, baselineDistances.n_cols);

  for (size_t i = 0; i < distances.n_elem; ++i)
  {
    if (std::abs(baselineDistances[i]) < 1e-5)
      BOOST_REQUIRE_SMALL(distances[i], 1e-5);
    else
      BOOST_REQUIRE_CLOSE(distances[i], baselineDistances[i], 1e-5);

    BOOST_REQUIRE_EQUAL(neighbors[i], baselineNeighbors[i]);
  }
}

/**
 * Test that when training is performed with a tree, the results are the same.
 */
BOOST_AUTO_TEST_CASE(TrainTreeTest)
{
  KNN empty;

  arma::mat dataset = arma::randu<arma::mat>(5, 100);
  KNN baseline(dataset);

  arma::Mat<size_t> neighbors, baselineNeighbors;
  arma::mat distances, baselineDistances;

  std::vector<size_t> oldFromNewReferences;
  KNN::Tree tree(dataset, oldFromNewReferences);
  empty.Train(std::move(tree));

  empty.Search(5, neighbors, distances);
  baseline.Search(5, baselineNeighbors, baselineDistances);

  BOOST_REQUIRE_EQUAL(neighbors.n_rows, baselineNeighbors.n_rows);
  BOOST_REQUIRE_EQUAL(neighbors.n_cols, baselineNeighbors.n_cols);
  BOOST_REQUIRE_EQUAL(distances.n_rows, baselineDistances.n_rows);
  BOOST_REQUIRE_EQUAL(distances.n_cols, baselineDistances.n_cols);

  // We have to unmap the results.
  arma::mat tmpDistances(distances.n_rows, distances.n_cols);
  arma::Mat<size_t> tmpNeighbors(neighbors.n_rows, neighbors.n_cols);
  for (size_t i = 0; i < distances.n_cols; ++i)
  {
    tmpDistances.col(oldFromNewReferences[i]) = distances.col(i);
    for (size_t j = 0; j < distances.n_rows; ++j)
    {
      tmpNeighbors(j, oldFromNewReferences[i]) =
          oldFromNewReferences[neighbors(j, i)];
    }
  }

  for (size_t i = 0; i < distances.n_elem; ++i)
  {
    if (std::abs(baselineDistances[i]) < 1e-5)
      BOOST_REQUIRE_SMALL(tmpDistances[i], 1e-5);
    else
      BOOST_REQUIRE_CLOSE(tmpDistances[i], baselineDistances[i], 1e-5);

    BOOST_REQUIRE_EQUAL(tmpNeighbors[i], baselineNeighbors[i]);
  }
}

/**
 * Test that training with a tree throws an exception when in naive mode.
 */
BOOST_AUTO_TEST_CASE(NaiveTrainTreeTest)
{
  KNN empty(NAIVE_MODE);

  arma::mat dataset = arma::randu<arma::mat>(5, 100);
  KNN::Tree tree(dataset);

  BOOST_REQUIRE_THROW(empty.Train(std::move(tree)), std::invalid_argument);
}

/**
 * Test that the rvalue reference move constructor works.
 */
BOOST_AUTO_TEST_CASE(MoveConstructorTest)
{
  arma::mat dataset = arma::randu<arma::mat>(3, 200);
  arma::mat copy(dataset);

  KNN moveknn(std::move(copy));
  KNN knn(dataset);

  BOOST_REQUIRE_EQUAL(copy.n_elem, 0);
  BOOST_REQUIRE_EQUAL(moveknn.ReferenceSet().n_rows, 3);
  BOOST_REQUIRE_EQUAL(moveknn.ReferenceSet().n_cols, 200);

  arma::mat moveDistances, distances;
  arma::Mat<size_t> moveNeighbors, neighbors;

  moveknn.Search(1, moveNeighbors, moveDistances);
  knn.Search(1, neighbors, distances);

  BOOST_REQUIRE_EQUAL(moveNeighbors.n_rows, neighbors.n_rows);
  BOOST_REQUIRE_EQUAL(moveNeighbors.n_cols, neighbors.n_cols);
  BOOST_REQUIRE_EQUAL(moveDistances.n_rows, distances.n_rows);
  BOOST_REQUIRE_EQUAL(moveDistances.n_cols, distances.n_cols);
  for (size_t i = 0; i < moveDistances.n_elem; ++i)
  {
    BOOST_REQUIRE_EQUAL(moveNeighbors[i], neighbors[i]);
    if (std::abs(distances[i]) < 1e-5)
      BOOST_REQUIRE_SMALL(moveDistances[i], 1e-5);
    else
      BOOST_REQUIRE_CLOSE(moveDistances[i], distances[i], 1e-5);
  }
}

/**
 * Test that the dataset can be retrained with the move Train() function.
 */
BOOST_AUTO_TEST_CASE(MoveTrainTest)
{
  arma::mat dataset = arma::randu<arma::mat>(3, 200);

  // Do it in tree mode, and in naive mode.
  KNN knn;
  knn.Train(std::move(dataset));

  arma::mat distances;
  arma::Mat<size_t> neighbors;
  knn.Search(1, neighbors, distances);

  BOOST_REQUIRE_EQUAL(dataset.n_elem, 0);
  BOOST_REQUIRE_EQUAL(neighbors.n_cols, 200);
  BOOST_REQUIRE_EQUAL(distances.n_cols, 200);

  dataset = arma::randu<arma::mat>(3, 300);
  knn.SearchMode() = NAIVE_MODE;
  knn.Train(std::move(dataset));
  knn.Search(1, neighbors, distances);

  BOOST_REQUIRE_EQUAL(dataset.n_elem, 0);
  BOOST_REQUIRE_EQUAL(neighbors.n_cols, 300);
  BOOST_REQUIRE_EQUAL(distances.n_cols, 300);
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

  typedef KDTree<EuclideanDistance, NeighborSearchStat<NearestNeighborSort>,
      arma::mat> TreeType;

  // We will loop through three times, one for each method of performing the
  // calculation.
  std::vector<size_t> oldFromNew;
  std::vector<size_t> newFromOld;
  TreeType tree(data, oldFromNew, newFromOld, 1);

  KNN knn(std::move(tree));

  for (int i = 0; i < 3; i++)
  {

    switch (i)
    {
      case 0: // Use the dual-tree method.
        knn.SearchMode() = DUAL_TREE_MODE;
        break;
      case 1: // Use the single-tree method.
        knn.SearchMode() = SINGLE_TREE_MODE;
        break;
      case 2: // Use the naive method.
        knn.SearchMode() = NAIVE_MODE;
        break;
    }

    // Now perform the actual calculation.
    arma::Mat<size_t> neighbors;
    arma::mat distances;
    knn.Search(10, neighbors, distances);

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

  }
}

/**
 * Test the dual-tree nearest-neighbors method with the naive method.  This
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

  KNN knn(dataset);

  KNN naive(dataset, NAIVE_MODE);

  arma::Mat<size_t> neighborsTree;
  arma::mat distancesTree;
  knn.Search(dataset, 15, neighborsTree, distancesTree);

  arma::Mat<size_t> neighborsNaive;
  arma::mat distancesNaive;
  naive.Search(dataset, 15, neighborsNaive, distancesNaive);

  for (size_t i = 0; i < neighborsTree.n_elem; i++)
  {
    BOOST_REQUIRE_EQUAL(neighborsTree(i), neighborsNaive(i));
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
  arma::mat dataset;

  // Hard-coded filename: bad?
  // Code duplication: also bad!
  if (!data::Load("test_data_3_1000.csv", dataset))
    BOOST_FAIL("Cannot load test dataset test_data_3_1000.csv!");

  KNN knn(dataset);

  // Set naive mode.
  KNN naive(dataset, NAIVE_MODE);

  arma::Mat<size_t> neighborsTree;
  arma::mat distancesTree;
  knn.Search(15, neighborsTree, distancesTree);

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
 * Test the single-tree nearest-neighbors method with the naive method.  This
 * uses only a reference dataset.
 *
 * Errors are produced if the results are not identical.
 */
BOOST_AUTO_TEST_CASE(SingleTreeVsNaive)
{
  arma::mat dataset;

  // Hard-coded filename: bad?
  // Code duplication: also bad!
  if (!data::Load("test_data_3_1000.csv", dataset))
    BOOST_FAIL("Cannot load test dataset test_data_3_1000.csv!");

  KNN knn(dataset, SINGLE_TREE_MODE);

  // Set up computation for naive mode.
  KNN naive(dataset, NAIVE_MODE);

  arma::Mat<size_t> neighborsTree;
  arma::mat distancesTree;
  knn.Search(15, neighborsTree, distancesTree);

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
 * Test the cover tree single-tree nearest-neighbors method against the naive
 * method.  This uses only a random reference dataset.
 *
 * Errors are produced if the results are not identical.
 */
BOOST_AUTO_TEST_CASE(SingleCoverTreeTest)
{
  arma::mat data;
  data.randu(75, 1000); // 75 dimensional, 1000 points.

  StandardCoverTree<EuclideanDistance, NeighborSearchStat<NearestNeighborSort>,
      arma::mat> tree(data);

  NeighborSearch<NearestNeighborSort, LMetric<2>, arma::mat, StandardCoverTree>
      coverTreeSearch(std::move(tree), SINGLE_TREE_MODE);

  KNN naive(data, NAIVE_MODE);

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

  KNN tree(dataset);

  arma::Mat<size_t> kdNeighbors;
  arma::mat kdDistances;
  tree.Search(dataset, 5, kdNeighbors, kdDistances);

  StandardCoverTree<EuclideanDistance, NeighborSearchStat<NearestNeighborSort>,
      arma::mat> referenceTree(dataset);

  NeighborSearch<NearestNeighborSort, EuclideanDistance, arma::mat,
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
 * Test the ball tree single-tree nearest-neighbors method against the naive
 * method.  This uses only a random reference dataset.
 *
 * Errors are produced if the results are not identical.
 */
BOOST_AUTO_TEST_CASE(SingleBallTreeTest)
{
  arma::mat data;
  data.randu(50, 300); // 50 dimensional, 300 points.

  typedef BallTree<EuclideanDistance, NeighborSearchStat<NearestNeighborSort>,
      arma::mat> TreeType;
  TreeType tree(data);

  KNN naive(tree.Dataset(), NAIVE_MODE);

  // BinarySpaceTree modifies data. Use modified data to maintain the
  // correspondance between points in the dataset for both methods. The order of
  // query points in both methods should be same.

  NeighborSearch<NearestNeighborSort, EuclideanDistance, arma::mat, BallTree>
      ballTreeSearch(std::move(tree), SINGLE_TREE_MODE);

  arma::Mat<size_t> ballTreeNeighbors;
  arma::mat ballTreeDistances;
  ballTreeSearch.Search(2, ballTreeNeighbors, ballTreeDistances);

  arma::Mat<size_t> naiveNeighbors;
  arma::mat naiveDistances;
  naive.Search(2, naiveNeighbors, naiveDistances);

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

  KNN tree(dataset);

  arma::Mat<size_t> kdNeighbors;
  arma::mat kdDistances;
  tree.Search(5, kdNeighbors, kdDistances);

  NeighborSearch<NearestNeighborSort, EuclideanDistance, arma::mat, BallTree>
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

/**
 * Test the spill tree hybrid sp-tree search (defeatist search on overlapping
 * nodes, and backtracking in non-overlapping nodes) against the naive method.
 * This uses only a random reference dataset.
 */
BOOST_AUTO_TEST_CASE(HybridSpillSearchTest)
{
  arma::mat dataset;
  dataset.randu(50, 300); // 50 dimensional, 300 points.

  const size_t k = 3;

  KNN naive(dataset);
  arma::Mat<size_t> neighborsNaive;
  arma::mat distancesNaive;
  naive.Search(dataset, k, neighborsNaive, distancesNaive);

  double maxDist = 0;
  for (size_t i = 0; i < neighborsNaive.n_cols; ++i)
    if (distancesNaive(k - 1, i) > maxDist)
      maxDist = distancesNaive(k - 1, i);

  // If we are sure that tau is a valid strict upper bound of the kth nearest
  // neighbor of the query points, then we can be sure that we will get an exact
  // solution.
  SpillKNN::Tree referenceTree(dataset, maxDist * 1.01 /* tau parameter */);
  SpillKNN spTreeSearch(std::move(referenceTree));

  for (size_t mode = 0; mode < 2; mode++)
  {
    if (mode)
      spTreeSearch.SearchMode() = SINGLE_TREE_MODE;

    arma::Mat<size_t> neighborsSPTree;
    arma::mat distancesSPTree;
    spTreeSearch.Search(dataset, k, neighborsSPTree, distancesSPTree);

    for (size_t i = 0; i < neighborsSPTree.n_elem; ++i)
    {
      BOOST_REQUIRE_EQUAL(neighborsSPTree(i), neighborsNaive(i));
      BOOST_REQUIRE_CLOSE(distancesSPTree(i), distancesNaive(i), 1e-5);
    }
  }
}

/**
 * Test hybrid sp-tree search doesn't repeat points.
 * This uses only a random reference dataset.
 */
BOOST_AUTO_TEST_CASE(DuplicatedSpillSearchTest)
{
  arma::mat dataset;
  dataset.randu(50, 300); // 50 dimensional, 300 points.

  const size_t k = 15;

  for (size_t test = 0; test < 2; test++)
  {
    double tau = test * 0.1;

    SpillKNN::Tree referenceTree(dataset, tau);
    SpillKNN spTreeSearch(std::move(referenceTree));

    arma::Mat<size_t> neighborsSPTree;
    arma::mat distancesSPTree;

    for (size_t mode = 0; mode < 2; mode++)
    {
      if (mode)
        spTreeSearch.SearchMode() = SINGLE_TREE_MODE;

      spTreeSearch.Search(dataset, k, neighborsSPTree, distancesSPTree);

      for (size_t i = 0; i < neighborsSPTree.n_cols; ++i)
      {
        // Test that at least one point was found.
        BOOST_REQUIRE(distancesSPTree(0, i) != DBL_MAX);

        for (size_t j = 0; j < neighborsSPTree.n_rows; ++j)
        {
          if (distancesSPTree(j, i) == DBL_MAX)
            break;
          // All candidates with same distances must be different points.
          for (size_t k = j + 1; k < neighborsSPTree.n_rows &&
              distancesSPTree(k, i) == distancesSPTree(j, i); ++k)
            BOOST_REQUIRE(neighborsSPTree(k, i) != neighborsSPTree(j, i));
        }
      }
    }
  }
}

/**
 * Make sure sparse nearest neighbors works with kd trees.
 */
BOOST_AUTO_TEST_CASE(SparseKNNKDTreeTest)
{
  // The dimensionality of these datasets must be high so that the probability
  // of a completely empty point is very low.  In this case, with dimensionality
  // 70, the probability of all 70 dimensions being zero is 0.8^70 = 1.65e-7 in
  // the reference set and 0.9^70 = 6.27e-4 in the query set.
  arma::sp_mat queryDataset;
  queryDataset.sprandu(70, 200, 0.2);
  arma::sp_mat referenceDataset;
  referenceDataset.sprandu(70, 500, 0.1);
  arma::mat denseQuery(queryDataset);
  arma::mat denseReference(referenceDataset);

  typedef NeighborSearch<NearestNeighborSort, EuclideanDistance, arma::sp_mat,
      KDTree> SparseKNN;

  SparseKNN a(referenceDataset);
  KNN naive(denseReference, NAIVE_MODE);

  arma::mat sparseDistances;
  arma::Mat<size_t> sparseNeighbors;
  a.Search(queryDataset, 10, sparseNeighbors, sparseDistances);

  arma::mat naiveDistances;
  arma::Mat<size_t> naiveNeighbors;
  naive.Search(denseQuery, 10, naiveNeighbors, naiveDistances);

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
BOOST_AUTO_TEST_CASE(SparseKNNCoverTreeTest)
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
      SparseCoverTree> SparseKNN;

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

BOOST_AUTO_TEST_CASE(KNNModelTest)
{
  // Ensure that we can build an NSModel<NearestNeighborSearch> and get correct
  // results.
  typedef NSModel<NearestNeighborSort> KNNModel;

  arma::mat queryData = arma::randu<arma::mat>(10, 50);
  arma::mat referenceData = arma::randu<arma::mat>(10, 200);

  // Build all the possible models.
  KNNModel models[28];
  models[0] = KNNModel(KNNModel::TreeTypes::KD_TREE, true);
  models[1] = KNNModel(KNNModel::TreeTypes::KD_TREE, false);
  models[2] = KNNModel(KNNModel::TreeTypes::COVER_TREE, true);
  models[3] = KNNModel(KNNModel::TreeTypes::COVER_TREE, false);
  models[4] = KNNModel(KNNModel::TreeTypes::R_TREE, true);
  models[5] = KNNModel(KNNModel::TreeTypes::R_TREE, false);
  models[6] = KNNModel(KNNModel::TreeTypes::R_STAR_TREE, true);
  models[7] = KNNModel(KNNModel::TreeTypes::R_STAR_TREE, false);
  models[8] = KNNModel(KNNModel::TreeTypes::X_TREE, true);
  models[9] = KNNModel(KNNModel::TreeTypes::X_TREE, false);
  models[10] = KNNModel(KNNModel::TreeTypes::BALL_TREE, true);
  models[11] = KNNModel(KNNModel::TreeTypes::BALL_TREE, false);
  models[12] = KNNModel(KNNModel::TreeTypes::HILBERT_R_TREE, true);
  models[13] = KNNModel(KNNModel::TreeTypes::HILBERT_R_TREE, false);
  models[14] = KNNModel(KNNModel::TreeTypes::R_PLUS_TREE, true);
  models[15] = KNNModel(KNNModel::TreeTypes::R_PLUS_TREE, false);
  models[16] = KNNModel(KNNModel::TreeTypes::R_PLUS_PLUS_TREE, true);
  models[17] = KNNModel(KNNModel::TreeTypes::R_PLUS_PLUS_TREE, false);
  models[18] = KNNModel(KNNModel::TreeTypes::VP_TREE, true);
  models[19] = KNNModel(KNNModel::TreeTypes::VP_TREE, false);
  models[20] = KNNModel(KNNModel::TreeTypes::RP_TREE, true);
  models[21] = KNNModel(KNNModel::TreeTypes::RP_TREE, false);
  models[22] = KNNModel(KNNModel::TreeTypes::MAX_RP_TREE, true);
  models[23] = KNNModel(KNNModel::TreeTypes::MAX_RP_TREE, false);
  models[24] = KNNModel(KNNModel::TreeTypes::UB_TREE, true);
  models[25] = KNNModel(KNNModel::TreeTypes::UB_TREE, false);
  models[26] = KNNModel(KNNModel::TreeTypes::OCTREE, true);
  models[27] = KNNModel(KNNModel::TreeTypes::OCTREE, false);

  for (size_t j = 0; j < 2; ++j)
  {
    // Get a baseline.
    KNN knn(referenceData);
    arma::Mat<size_t> baselineNeighbors;
    arma::mat baselineDistances;
    knn.Search(queryData, 3, baselineNeighbors, baselineDistances);

    for (size_t i = 0; i < 28; ++i)
    {
      // We only have std::move() constructors so make a copy of our data.
      arma::mat referenceCopy(referenceData);
      arma::mat queryCopy(queryData);
      if (j == 0)
        models[i].BuildModel(std::move(referenceCopy), 20, DUAL_TREE_MODE);
      if (j == 1)
        models[i].BuildModel(std::move(referenceCopy), 20,
            SINGLE_TREE_MODE);
      if (j == 2)
        models[i].BuildModel(std::move(referenceCopy), 20, NAIVE_MODE);

      arma::Mat<size_t> neighbors;
      arma::mat distances;

      models[i].Search(std::move(queryCopy), 3, neighbors, distances);

      BOOST_REQUIRE_EQUAL(neighbors.n_rows, baselineNeighbors.n_rows);
      BOOST_REQUIRE_EQUAL(neighbors.n_cols, baselineNeighbors.n_cols);
      BOOST_REQUIRE_EQUAL(neighbors.n_elem, baselineNeighbors.n_elem);
      BOOST_REQUIRE_EQUAL(distances.n_rows, baselineDistances.n_rows);
      BOOST_REQUIRE_EQUAL(distances.n_cols, baselineDistances.n_cols);
      BOOST_REQUIRE_EQUAL(distances.n_elem, baselineDistances.n_elem);
      for (size_t k = 0; k < distances.n_elem; ++k)
      {
        BOOST_REQUIRE_EQUAL(neighbors[k], baselineNeighbors[k]);
        if (std::abs(baselineDistances[k]) < 1e-5)
          BOOST_REQUIRE_SMALL(distances[k], 1e-5);
        else
          BOOST_REQUIRE_CLOSE(distances[k], baselineDistances[k], 1e-5);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(KNNModelMonochromaticTest)
{
  // Ensure that we can build an NSModel<NearestNeighborSearch> and get correct
  // results, in the case where the reference set is the same as the query set.
  typedef NSModel<NearestNeighborSort> KNNModel;

  arma::mat referenceData = arma::randu<arma::mat>(10, 200);

  // Build all the possible models.
  KNNModel models[28];
  models[0] = KNNModel(KNNModel::TreeTypes::KD_TREE, true);
  models[1] = KNNModel(KNNModel::TreeTypes::KD_TREE, false);
  models[2] = KNNModel(KNNModel::TreeTypes::COVER_TREE, true);
  models[3] = KNNModel(KNNModel::TreeTypes::COVER_TREE, false);
  models[4] = KNNModel(KNNModel::TreeTypes::R_TREE, true);
  models[5] = KNNModel(KNNModel::TreeTypes::R_TREE, false);
  models[6] = KNNModel(KNNModel::TreeTypes::R_STAR_TREE, true);
  models[7] = KNNModel(KNNModel::TreeTypes::R_STAR_TREE, false);
  models[8] = KNNModel(KNNModel::TreeTypes::X_TREE, true);
  models[9] = KNNModel(KNNModel::TreeTypes::X_TREE, false);
  models[10] = KNNModel(KNNModel::TreeTypes::BALL_TREE, true);
  models[11] = KNNModel(KNNModel::TreeTypes::BALL_TREE, false);
  models[12] = KNNModel(KNNModel::TreeTypes::HILBERT_R_TREE, true);
  models[13] = KNNModel(KNNModel::TreeTypes::HILBERT_R_TREE, false);
  models[14] = KNNModel(KNNModel::TreeTypes::R_PLUS_TREE, true);
  models[15] = KNNModel(KNNModel::TreeTypes::R_PLUS_TREE, false);
  models[16] = KNNModel(KNNModel::TreeTypes::R_PLUS_PLUS_TREE, true);
  models[17] = KNNModel(KNNModel::TreeTypes::R_PLUS_PLUS_TREE, false);
  models[18] = KNNModel(KNNModel::TreeTypes::VP_TREE, true);
  models[19] = KNNModel(KNNModel::TreeTypes::VP_TREE, false);
  models[20] = KNNModel(KNNModel::TreeTypes::RP_TREE, true);
  models[21] = KNNModel(KNNModel::TreeTypes::RP_TREE, false);
  models[22] = KNNModel(KNNModel::TreeTypes::MAX_RP_TREE, true);
  models[23] = KNNModel(KNNModel::TreeTypes::MAX_RP_TREE, false);
  models[24] = KNNModel(KNNModel::TreeTypes::UB_TREE, true);
  models[25] = KNNModel(KNNModel::TreeTypes::UB_TREE, false);
  models[26] = KNNModel(KNNModel::TreeTypes::OCTREE, true);
  models[27] = KNNModel(KNNModel::TreeTypes::OCTREE, false);

  for (size_t j = 0; j < 2; ++j)
  {
    // Get a baseline.
    KNN knn(referenceData);
    arma::Mat<size_t> baselineNeighbors;
    arma::mat baselineDistances;
    knn.Search(3, baselineNeighbors, baselineDistances);

    for (size_t i = 0; i < 28; ++i)
    {
      // We only have a std::move() constructor... so copy the data.
      arma::mat referenceCopy(referenceData);
      if (j == 0)
        models[i].BuildModel(std::move(referenceCopy), 20, DUAL_TREE_MODE);
      if (j == 1)
        models[i].BuildModel(std::move(referenceCopy), 20,
            SINGLE_TREE_MODE);
      if (j == 2)
        models[i].BuildModel(std::move(referenceCopy), 20, NAIVE_MODE);

      arma::Mat<size_t> neighbors;
      arma::mat distances;

      models[i].Search(3, neighbors, distances);

      BOOST_REQUIRE_EQUAL(neighbors.n_rows, baselineNeighbors.n_rows);
      BOOST_REQUIRE_EQUAL(neighbors.n_cols, baselineNeighbors.n_cols);
      BOOST_REQUIRE_EQUAL(neighbors.n_elem, baselineNeighbors.n_elem);
      BOOST_REQUIRE_EQUAL(distances.n_rows, baselineDistances.n_rows);
      BOOST_REQUIRE_EQUAL(distances.n_cols, baselineDistances.n_cols);
      BOOST_REQUIRE_EQUAL(distances.n_elem, baselineDistances.n_elem);
      for (size_t k = 0; k < distances.n_elem; ++k)
      {
        BOOST_REQUIRE_EQUAL(neighbors[k], baselineNeighbors[k]);
        if (std::abs(baselineDistances[k]) < 1e-5)
          BOOST_REQUIRE_SMALL(distances[k], 1e-5);
        else
          BOOST_REQUIRE_CLOSE(distances[k], baselineDistances[k], 1e-5);
      }
    }
  }
}

/**
 * If we search twice with the same reference tree, the bounds need to be reset
 * before the second search.  This test ensures that that happens, by making
 * sure the number of scores and base cases are equivalent for each search.
 */
BOOST_AUTO_TEST_CASE(DoubleReferenceSearchTest)
{
  arma::mat dataset = arma::randu<arma::mat>(5, 500);
  KNN knn(std::move(dataset));

  arma::mat distances, secondDistances;
  arma::Mat<size_t> neighbors, secondNeighbors;
  knn.Search(3, neighbors, distances);
  size_t baseCases = knn.BaseCases();
  size_t scores = knn.Scores();

  knn.Search(3, secondNeighbors, secondDistances);

  BOOST_REQUIRE_EQUAL(knn.BaseCases(), baseCases);
  BOOST_REQUIRE_EQUAL(knn.Scores(), scores);
}

/**
 * Make sure that the neighborPtr matrix isn't accidentally deleted.
 * See issue #478.
 */
BOOST_AUTO_TEST_CASE(NeighborPtrDeleteTest)
{
  arma::mat dataset = arma::randu<arma::mat>(5, 100);

  // Build the tree ourselves.
  std::vector<size_t> oldFromNewReferences;
  KNN::Tree tree(dataset);
  KNN knn(std::move(tree));

  // Now make a query set.
  arma::mat queryset = arma::randu<arma::mat>(5, 50);
  arma::mat distances;
  arma::Mat<size_t> neighbors;
  knn.Search(queryset, 3, neighbors, distances);

  // These will (hopefully) fail is either the neighbors or the distances matrix
  // has been accidentally deleted.
  BOOST_REQUIRE_EQUAL(neighbors.n_cols, 50);
  BOOST_REQUIRE_EQUAL(neighbors.n_rows, 3);
  BOOST_REQUIRE_EQUAL(distances.n_cols, 50);
  BOOST_REQUIRE_EQUAL(distances.n_rows, 3);
}

BOOST_AUTO_TEST_SUITE_END();
