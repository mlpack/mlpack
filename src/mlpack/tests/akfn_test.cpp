/**
 * @file akfn_test.cpp
 *
 * Tests for KFN (k-furthest-neighbors) with different values of epsilon.
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

BOOST_AUTO_TEST_SUITE(AKFNTest);

/**
 * Test the dual-tree furthest-neighbors method with different values for
 * epsilon. This uses both a query and reference dataset.
 *
 * Errors are produced if the results are not according to relative error.
 */
BOOST_AUTO_TEST_CASE(DualTreeVsNaive1)
{
  arma::mat dataset;

  if (!data::Load("test_data_3_1000.csv", dataset))
    BOOST_FAIL("Cannot load test dataset test_data_3_1000.csv!");

  KFN naive(dataset, true);
  arma::Mat<size_t> neighborsNaive;
  arma::mat distancesNaive;
  naive.Search(dataset, 15, neighborsNaive, distancesNaive);

  for (size_t c = 0; c < 4; c++)
  {
    KFN* kfn;
    double epsilon;

    switch (c)
    {
      case 0: // Use the dual-tree method with e=0.02.
        epsilon = 0.02;
        break;
      case 1: // Use the dual-tree method with e=0.05.
        epsilon = 0.05;
        break;
      case 2: // Use the dual-tree method with e=0.10.
        epsilon = 0.10;
        break;
      case 3: // Use the dual-tree method with e=0.20.
        epsilon = 0.20;
        break;
    }

    kfn = new KFN(dataset, false, false, epsilon);

    // Now perform the actual calculation.
    arma::Mat<size_t> neighborsTree;
    arma::mat distancesTree;
    kfn->Search(dataset, 15, neighborsTree, distancesTree);

    for (size_t i = 0; i < neighborsTree.n_elem; i++)
      REQUIRE_RELATIVE_ERR(distancesTree(i), distancesNaive(i), epsilon);

    // Clean the memory.
    delete kfn;
  }
}

/**
 * Test the dual-tree furthest-neighbors method with the naive method.  This
 * uses only a reference dataset.
 *
 * Errors are produced if the results are not according to relative error.
 */
BOOST_AUTO_TEST_CASE(DualTreeVsNaive2)
{
  arma::mat dataset;

  if (!data::Load("test_data_3_1000.csv", dataset))
    BOOST_FAIL("Cannot load test dataset test_data_3_1000.csv!");

  KFN naive(dataset, true);
  arma::Mat<size_t> neighborsNaive;
  arma::mat distancesNaive;
  naive.Search(15, neighborsNaive, distancesNaive);

  KFN kfn(dataset, false, false, 0.05);
  arma::Mat<size_t> neighborsTree;
  arma::mat distancesTree;
  kfn.Search(15, neighborsTree, distancesTree);

  for (size_t i = 0; i < neighborsTree.n_elem; i++)
    REQUIRE_RELATIVE_ERR(distancesTree[i], distancesNaive[i], 0.05);
}

/**
 * Test the single-tree furthest-neighbors method with the naive method.  This
 * uses only a reference dataset.
 *
 * Errors are produced if the results are not according to relative error.
 */
BOOST_AUTO_TEST_CASE(SingleTreeVsNaive)
{
  arma::mat dataset;

  if (!data::Load("test_data_3_1000.csv", dataset))
    BOOST_FAIL("Cannot load test dataset test_data_3_1000.csv!");

  KFN naive(dataset, true);
  arma::Mat<size_t> neighborsNaive;
  arma::mat distancesNaive;
  naive.Search(15, neighborsNaive, distancesNaive);

  KFN kfn(dataset, false, true, 0.05);
  arma::Mat<size_t> neighborsTree;
  arma::mat distancesTree;
  kfn.Search(15, neighborsTree, distancesTree);

  for (size_t i = 0; i < neighborsTree.n_elem; i++)
    REQUIRE_RELATIVE_ERR(distancesTree[i], distancesNaive[i], 0.05);
}

/**
 * Test the cover tree single-tree furthest-neighbors method against the naive
 * method.  This uses only a random reference dataset.
 *
 * Errors are produced if the results are not according to relative error.
 */
BOOST_AUTO_TEST_CASE(SingleCoverTreeTest)
{
  arma::mat data;
  data.randu(75, 1000); // 75 dimensional, 1000 points.

  KFN naive(data, true);
  arma::Mat<size_t> naiveNeighbors;
  arma::mat naiveDistances;
  naive.Search(data, 15, naiveNeighbors, naiveDistances);

  StandardCoverTree<EuclideanDistance, NeighborSearchStat<FurthestNeighborSort>,
      arma::mat> tree(data);

  NeighborSearch<FurthestNeighborSort, LMetric<2>, arma::mat, StandardCoverTree>
      coverTreeSearch(&tree, true, 0.05);

  arma::Mat<size_t> coverTreeNeighbors;
  arma::mat coverTreeDistances;
  coverTreeSearch.Search(data, 15, coverTreeNeighbors, coverTreeDistances);

  for (size_t i = 0; i < coverTreeNeighbors.n_elem; ++i)
    REQUIRE_RELATIVE_ERR(coverTreeDistances[i], naiveDistances[i], 0.05);
}

/**
 * Test the cover tree dual-tree furthest neighbors method against the naive
 * method.
 *
 * Errors are produced if the results are not according to relative error.
 */
BOOST_AUTO_TEST_CASE(DualCoverTreeTest)
{
  arma::mat dataset;
  data::Load("test_data_3_1000.csv", dataset);

  KFN naive(dataset, true);
  arma::Mat<size_t> naiveNeighbors;
  arma::mat naiveDistances;
  naive.Search(dataset, 15, naiveNeighbors, naiveDistances);

  StandardCoverTree<EuclideanDistance, NeighborSearchStat<FurthestNeighborSort>,
      arma::mat> referenceTree(dataset);

  NeighborSearch<FurthestNeighborSort, LMetric<2>, arma::mat, StandardCoverTree>
      coverTreeSearch(&referenceTree, false, 0.05);

  arma::Mat<size_t> coverTreeNeighbors;
  arma::mat coverTreeDistances;
  coverTreeSearch.Search(dataset, 15, coverTreeNeighbors, coverTreeDistances);

  for (size_t i = 0; i < coverTreeNeighbors.n_elem; ++i)
    REQUIRE_RELATIVE_ERR(coverTreeDistances[i], naiveDistances[i], 0.05);
}

/**
 * Test the ball tree single-tree furthest-neighbors method against the naive
 * method.  This uses only a random reference dataset.
 *
 * Errors are produced if the results are not according to relative error.
 */
BOOST_AUTO_TEST_CASE(SingleBallTreeTest)
{
  arma::mat data;
  data.randu(75, 1000); // 75 dimensional, 1000 points.

  KFN naive(data, true);
  arma::Mat<size_t> naiveNeighbors;
  arma::mat naiveDistances;
  naive.Search(data, 15, naiveNeighbors, naiveDistances);

  NeighborSearch<FurthestNeighborSort, EuclideanDistance, arma::mat, BallTree>
      ballTreeSearch(data, false, true, 0.05);

  arma::Mat<size_t> ballNeighbors;
  arma::mat ballDistances;
  ballTreeSearch.Search(data, 15, ballNeighbors, ballDistances);

  for (size_t i = 0; i < ballNeighbors.n_elem; ++i)
    REQUIRE_RELATIVE_ERR(ballDistances(i), naiveDistances(i), 0.05);
}

/**
 * Test the ball tree dual-tree furthest neighbors method against the naive
 * method.
 *
 * Errors are produced if the results are not according to relative error.
 */
BOOST_AUTO_TEST_CASE(DualBallTreeTest)
{
  arma::mat dataset;
  data::Load("test_data_3_1000.csv", dataset);

  KFN naive(dataset, true);
  arma::Mat<size_t> naiveNeighbors;
  arma::mat naiveDistances;
  naive.Search(15, naiveNeighbors, naiveDistances);

  NeighborSearch<FurthestNeighborSort, EuclideanDistance, arma::mat, BallTree>
      ballTreeSearch(dataset, false, false, 0.05);
  arma::Mat<size_t> ballNeighbors;
  arma::mat ballDistances;
  ballTreeSearch.Search(15, ballNeighbors, ballDistances);

  for (size_t i = 0; i < ballNeighbors.n_elem; ++i)
    REQUIRE_RELATIVE_ERR(ballDistances(i), naiveDistances(i), 0.05);
}

BOOST_AUTO_TEST_SUITE_END();
