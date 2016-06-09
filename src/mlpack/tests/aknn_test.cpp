/**
 * @file aknn_test.cpp
 *
 * Test file for KNN class with different values of epsilon.
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

BOOST_AUTO_TEST_SUITE(AKNNTest);

/**
 * Test the dual-tree nearest-neighbors method with different values for
 * epsilon. This uses both a query and reference dataset.
 *
 * Errors are produced if the results are not according to relative error.
 */
BOOST_AUTO_TEST_CASE(DualTreeVsNaive1)
{
  arma::mat dataset;

  if (!data::Load("test_data_3_1000.csv", dataset))
    BOOST_FAIL("Cannot load test dataset test_data_3_1000.csv!");

  KNN naive(dataset, true);
  arma::Mat<size_t> neighborsNaive;
  arma::mat distancesNaive;
  naive.Search(dataset, 15, neighborsNaive, distancesNaive);

  for (size_t c = 0; c < 4; c++)
  {
    KNN* knn;
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

    knn = new KNN(dataset, false, false, epsilon);

    // Now perform the actual calculation.
    arma::Mat<size_t> neighborsTree;
    arma::mat distancesTree;
    knn->Search(dataset, 15, neighborsTree, distancesTree);

    for (size_t i = 0; i < neighborsTree.n_elem; i++)
      REQUIRE_RELATIVE_ERR(distancesTree(i), distancesNaive(i), epsilon);

    // Clean the memory.
    delete knn;
  }
}

/**
 * Test the dual-tree nearest-neighbors method with the naive method.  This uses
 * only a reference dataset.
 *
 * Errors are produced if the results are not according to relative error.
 */
BOOST_AUTO_TEST_CASE(DualTreeVsNaive2)
{
  arma::mat dataset;

  if (!data::Load("test_data_3_1000.csv", dataset))
    BOOST_FAIL("Cannot load test dataset test_data_3_1000.csv!");

  KNN naive(dataset, true);
  arma::Mat<size_t> neighborsNaive;
  arma::mat distancesNaive;
  naive.Search(15, neighborsNaive, distancesNaive);

  KNN knn(dataset, false, false, 0.05);
  arma::Mat<size_t> neighborsTree;
  arma::mat distancesTree;
  knn.Search(15, neighborsTree, distancesTree);

  for (size_t i = 0; i < neighborsTree.n_elem; i++)
    REQUIRE_RELATIVE_ERR(distancesTree(i), distancesNaive(i), 0.05);
}

/**
 * Test the single-tree nearest-neighbors method with the naive method.  This
 * uses only a reference dataset.
 *
 * Errors are produced if the results are not according to relative error.
 */
BOOST_AUTO_TEST_CASE(SingleTreeVsNaive)
{
  arma::mat dataset;

  if (!data::Load("test_data_3_1000.csv", dataset))
    BOOST_FAIL("Cannot load test dataset test_data_3_1000.csv!");

  KNN naive(dataset, true);
  arma::Mat<size_t> neighborsNaive;
  arma::mat distancesNaive;
  naive.Search(15, neighborsNaive, distancesNaive);

  KNN knn(dataset, false, true, 0.05);
  arma::Mat<size_t> neighborsTree;
  arma::mat distancesTree;
  knn.Search(15, neighborsTree, distancesTree);

  for (size_t i = 0; i < neighborsTree.n_elem; i++)
    REQUIRE_RELATIVE_ERR(distancesTree[i], distancesNaive[i], 0.05);
}

/**
 * Test the cover tree single-tree nearest-neighbors method against the naive
 * method.  This uses only a random reference dataset.
 *
 * Errors are produced if the results are not according to relative error.
 */
BOOST_AUTO_TEST_CASE(SingleCoverTreeTest)
{
  arma::mat data;
  data.randu(75, 1000); // 75 dimensional, 1000 points.

  KNN naive(data, true);
  arma::Mat<size_t> naiveNeighbors;
  arma::mat naiveDistances;
  naive.Search(data, 15, naiveNeighbors, naiveDistances);

  StandardCoverTree<EuclideanDistance, NeighborSearchStat<NearestNeighborSort>,
      arma::mat> tree(data);

  NeighborSearch<NearestNeighborSort, LMetric<2>, arma::mat, StandardCoverTree>
      coverTreeSearch(&tree, true, 0.05);

  arma::Mat<size_t> coverTreeNeighbors;
  arma::mat coverTreeDistances;
  coverTreeSearch.Search(data, 15, coverTreeNeighbors, coverTreeDistances);

  for (size_t i = 0; i < coverTreeNeighbors.n_elem; ++i)
    REQUIRE_RELATIVE_ERR(coverTreeDistances[i], naiveDistances[i], 0.05);
}

/**
 * Test the cover tree dual-tree nearest neighbors method against the naive
 * method.
 *
 * Errors are produced if the results are not according to relative error.
 */
BOOST_AUTO_TEST_CASE(DualCoverTreeTest)
{
  arma::mat dataset;
  data::Load("test_data_3_1000.csv", dataset);

  KNN naive(dataset, true);
  arma::Mat<size_t> naiveNeighbors;
  arma::mat naiveDistances;
  naive.Search(dataset, 15, naiveNeighbors, naiveDistances);

  StandardCoverTree<EuclideanDistance, NeighborSearchStat<NearestNeighborSort>,
      arma::mat> referenceTree(dataset);

  NeighborSearch<NearestNeighborSort, EuclideanDistance, arma::mat,
      StandardCoverTree> coverTreeSearch(&referenceTree, false, 0.05);

  arma::Mat<size_t> coverNeighbors;
  arma::mat coverDistances;
  coverTreeSearch.Search(&referenceTree, 15, coverNeighbors, coverDistances);

  for (size_t i = 0; i < coverNeighbors.n_elem; ++i)
    REQUIRE_RELATIVE_ERR(coverDistances[i], naiveDistances[i], 0.05);
}

/**
 * Test the ball tree single-tree nearest-neighbors method against the naive
 * method.  This uses only a random reference dataset.
 *
 * Errors are produced if the results are not according to relative error.
 */
BOOST_AUTO_TEST_CASE(SingleBallTreeTest)
{
  arma::mat data;
  data.randu(50, 300); // 50 dimensional, 300 points.

  KNN naive(data, true);
  arma::Mat<size_t> naiveNeighbors;
  arma::mat naiveDistances;
  naive.Search(data, 15, naiveNeighbors, naiveDistances);

  NeighborSearch<NearestNeighborSort, EuclideanDistance, arma::mat, BallTree>
      ballTreeSearch(data, false, true, 0.05);

  arma::Mat<size_t> ballNeighbors;
  arma::mat ballDistances;
  ballTreeSearch.Search(data, 15, ballNeighbors, ballDistances);

  for (size_t i = 0; i < ballNeighbors.n_elem; ++i)
    REQUIRE_RELATIVE_ERR(ballDistances(i), naiveDistances(i), 0.05);
}

/**
 * Test the ball tree dual-tree nearest neighbors method against the naive
 * method.
 *
 * Errors are produced if the results are not according to relative error.
 */
BOOST_AUTO_TEST_CASE(DualBallTreeTest)
{
  arma::mat dataset;
  data::Load("test_data_3_1000.csv", dataset);

  KNN naive(dataset, true);
  arma::Mat<size_t> naiveNeighbors;
  arma::mat naiveDistances;
  naive.Search(15, naiveNeighbors, naiveDistances);

  NeighborSearch<NearestNeighborSort, EuclideanDistance, arma::mat, BallTree>
      ballTreeSearch(dataset, false, false, 0.05);
  arma::Mat<size_t> ballNeighbors;
  arma::mat ballDistances;
  ballTreeSearch.Search(15, ballNeighbors, ballDistances);

  for (size_t i = 0; i < ballNeighbors.n_elem; ++i)
    REQUIRE_RELATIVE_ERR(ballDistances(i), naiveDistances(i), 0.05);
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

  SparseKNN a(referenceDataset, false, false, 0.05);
  KNN naive(denseReference, true);

  arma::mat sparseDistances;
  arma::Mat<size_t> sparseNeighbors;
  a.Search(queryDataset, 10, sparseNeighbors, sparseDistances);

  arma::mat naiveDistances;
  arma::Mat<size_t> naiveNeighbors;
  naive.Search(denseQuery, 10, naiveNeighbors, naiveDistances);

  for (size_t i = 0; i < naiveNeighbors.n_cols; ++i)
    for (size_t j = 0; j < naiveNeighbors.n_rows; ++j)
      REQUIRE_RELATIVE_ERR(sparseDistances(j, i), naiveDistances(j, i), 0.05);
}

/**
 * Ensure that we can build an NSModel<NearestNeighborSearch> and get correct
 * results.
 */
BOOST_AUTO_TEST_CASE(KNNModelTest)
{
  typedef NSModel<NearestNeighborSort> KNNModel;

  arma::mat queryData = arma::randu<arma::mat>(10, 50);
  arma::mat referenceData = arma::randu<arma::mat>(10, 200);

  // Build all the possible models.
  KNNModel models[12];
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

  for (size_t j = 0; j < 3; ++j)
  {
    // Get a baseline.
    KNN knn(referenceData);
    arma::Mat<size_t> baselineNeighbors;
    arma::mat baselineDistances;
    knn.Search(queryData, 3, baselineNeighbors, baselineDistances);

    for (size_t i = 0; i < 12; ++i)
    {
      // We only have std::move() constructors so make a copy of our data.
      arma::mat referenceCopy(referenceData);
      arma::mat queryCopy(queryData);
      if (j == 0)
        models[i].BuildModel(std::move(referenceCopy), 20, false, false, 0.05);
      if (j == 1)
        models[i].BuildModel(std::move(referenceCopy), 20, false, true, 0.05);
      if (j == 2)
        models[i].BuildModel(std::move(referenceCopy), 20, true, false);

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
        REQUIRE_RELATIVE_ERR(distances[k], baselineDistances[k], 0.05);
    }
  }
}

/**
 * Ensure that we can build an NSModel<NearestNeighborSearch> and get correct
 * results, in the case where the reference set is the same as the query set.
 */
BOOST_AUTO_TEST_CASE(KNNModelMonochromaticTest)
{
  typedef NSModel<NearestNeighborSort> KNNModel;

  arma::mat referenceData = arma::randu<arma::mat>(10, 200);

  // Build all the possible models.
  KNNModel models[12];
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

  for (size_t j = 0; j < 3; ++j)
  {
    // Get a baseline.
    KNN knn(referenceData);
    arma::Mat<size_t> baselineNeighbors;
    arma::mat baselineDistances;
    knn.Search(3, baselineNeighbors, baselineDistances);

    for (size_t i = 0; i < 12; ++i)
    {
      // We only have a std::move() constructor... so copy the data.
      arma::mat referenceCopy(referenceData);
      if (j == 0)
        models[i].BuildModel(std::move(referenceCopy), 20, false, false, 0.05);
      if (j == 1)
        models[i].BuildModel(std::move(referenceCopy), 20, false, true, 0.05);
      if (j == 2)
        models[i].BuildModel(std::move(referenceCopy), 20, true, false);

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
        REQUIRE_RELATIVE_ERR(distances[k], baselineDistances[k], 0.05);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END();
