/**
 * @file aknn_test.cpp
 *
 * Test file for KNN class with different values of epsilon.
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

BOOST_AUTO_TEST_SUITE(AKNNTest);

/**
 * Test the dual-tree nearest-neighbors method with different values for
 * epsilon. This uses both a query and reference dataset.
 *
 * Errors are produced if the results are not according to relative error.
 */
BOOST_AUTO_TEST_CASE(ApproxVsExact1)
{
  arma::mat dataset;

  if (!data::Load("test_data_3_1000.csv", dataset))
    BOOST_FAIL("Cannot load test dataset test_data_3_1000.csv!");

  KNN exact(dataset);
  arma::Mat<size_t> neighborsExact;
  arma::mat distancesExact;
  exact.Search(dataset, 15, neighborsExact, distancesExact);

  for (size_t c = 0; c < 4; c++)
  {
    KNN* aknn;
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

    // Now perform the actual calculation.
    aknn = new KNN(dataset, DUAL_TREE_MODE, epsilon);
    arma::Mat<size_t> neighborsApprox;
    arma::mat distancesApprox;
    aknn->Search(dataset, 15, neighborsApprox, distancesApprox);

    for (size_t i = 0; i < neighborsApprox.n_elem; i++)
      REQUIRE_RELATIVE_ERR(distancesApprox(i), distancesExact(i), epsilon);

    // Clean the memory.
    delete aknn;
  }
}

/**
 * Test the dual-tree nearest-neighbors method with the exact method.  This uses
 * only a reference dataset.
 *
 * Errors are produced if the results are not according to relative error.
 */
BOOST_AUTO_TEST_CASE(ApproxVsExact2)
{
  arma::mat dataset;

  if (!data::Load("test_data_3_1000.csv", dataset))
    BOOST_FAIL("Cannot load test dataset test_data_3_1000.csv!");

  KNN exact(dataset);
  arma::Mat<size_t> neighborsExact;
  arma::mat distancesExact;
  exact.Search(15, neighborsExact, distancesExact);

  KNN aknn(dataset, DUAL_TREE_MODE, 0.05);
  arma::Mat<size_t> neighborsApprox;
  arma::mat distancesApprox;
  aknn.Search(15, neighborsApprox, distancesApprox);

  for (size_t i = 0; i < neighborsApprox.n_elem; i++)
    REQUIRE_RELATIVE_ERR(distancesApprox(i), distancesExact(i), 0.05);
}

/**
 * Test the single-tree nearest-neighbors method with the exact method.  This
 * uses only a reference dataset.
 *
 * Errors are produced if the results are not according to relative error.
 */
BOOST_AUTO_TEST_CASE(SingleTreeApproxVsExact)
{
  arma::mat dataset;

  if (!data::Load("test_data_3_1000.csv", dataset))
    BOOST_FAIL("Cannot load test dataset test_data_3_1000.csv!");

  KNN exact(dataset);
  arma::Mat<size_t> neighborsExact;
  arma::mat distancesExact;
  exact.Search(15, neighborsExact, distancesExact);

  KNN aknn(dataset, SINGLE_TREE_MODE, 0.05);
  arma::Mat<size_t> neighborsApprox;
  arma::mat distancesApprox;
  aknn.Search(15, neighborsApprox, distancesApprox);

  for (size_t i = 0; i < neighborsApprox.n_elem; i++)
    REQUIRE_RELATIVE_ERR(distancesApprox[i], distancesExact[i], 0.05);
}

/**
 * Test the cover tree single-tree nearest-neighbors method against the exact
 * method.  This uses only a random reference dataset.
 *
 * Errors are produced if the results are not according to relative error.
 */
BOOST_AUTO_TEST_CASE(SingleCoverTreeTest)
{
  arma::mat dataset;
  dataset.randu(75, 1000); // 75 dimensional, 1000 points.

  KNN exact(dataset);
  arma::Mat<size_t> neighborsExact;
  arma::mat distancesExact;
  exact.Search(dataset, 15, neighborsExact, distancesExact);

  StandardCoverTree<EuclideanDistance, NeighborSearchStat<NearestNeighborSort>,
      arma::mat> tree(dataset);

  NeighborSearch<NearestNeighborSort, LMetric<2>, arma::mat, StandardCoverTree>
      coverTreeSearch(std::move(tree), SINGLE_TREE_MODE, 0.05);

  arma::Mat<size_t> neighborsCoverTree;
  arma::mat distancesCoverTree;
  coverTreeSearch.Search(dataset, 15, neighborsCoverTree, distancesCoverTree);

  for (size_t i = 0; i < neighborsCoverTree.n_elem; ++i)
    REQUIRE_RELATIVE_ERR(distancesCoverTree[i], distancesExact[i], 0.05);
}

/**
 * Test the cover tree dual-tree nearest neighbors method against the exact
 * method.
 *
 * Errors are produced if the results are not according to relative error.
 */
BOOST_AUTO_TEST_CASE(DualCoverTreeTest)
{
  arma::mat dataset;
  data::Load("test_data_3_1000.csv", dataset);

  KNN exact(dataset);
  arma::Mat<size_t> neighborsExact;
  arma::mat distancesExact;
  exact.Search(dataset, 15, neighborsExact, distancesExact);

  NeighborSearch<NearestNeighborSort, EuclideanDistance, arma::mat,
      StandardCoverTree> coverTreeSearch(dataset, DUAL_TREE_MODE, 0.05);

  arma::Mat<size_t> neighborsCoverTree;
  arma::mat distancesCoverTree;
  coverTreeSearch.Search(dataset, 15, neighborsCoverTree, distancesCoverTree);

  for (size_t i = 0; i < neighborsCoverTree.n_elem; ++i)
    REQUIRE_RELATIVE_ERR(distancesCoverTree[i], distancesExact[i], 0.05);
}

/**
 * Test the ball tree single-tree nearest-neighbors method against the exact
 * method.  This uses only a random reference dataset.
 *
 * Errors are produced if the results are not according to relative error.
 */
BOOST_AUTO_TEST_CASE(SingleBallTreeTest)
{
  arma::mat dataset;
  dataset.randu(50, 300); // 50 dimensional, 300 points.

  KNN exact(dataset);
  arma::Mat<size_t> neighborsExact;
  arma::mat distancesExact;
  exact.Search(dataset, 15, neighborsExact, distancesExact);

  NeighborSearch<NearestNeighborSort, EuclideanDistance, arma::mat, BallTree>
      ballTreeSearch(dataset, SINGLE_TREE_MODE, 0.05);

  arma::Mat<size_t> neighborsBallTree;
  arma::mat distancesBallTree;
  ballTreeSearch.Search(dataset, 15, neighborsBallTree, distancesBallTree);

  for (size_t i = 0; i < neighborsBallTree.n_elem; ++i)
    REQUIRE_RELATIVE_ERR(distancesBallTree(i), distancesExact(i), 0.05);
}

/**
 * Test the ball tree dual-tree nearest neighbors method against the exact
 * method.
 *
 * Errors are produced if the results are not according to relative error.
 */
BOOST_AUTO_TEST_CASE(DualBallTreeTest)
{
  arma::mat dataset;
  data::Load("test_data_3_1000.csv", dataset);

  KNN exact(dataset);
  arma::Mat<size_t> neighborsExact;
  arma::mat distancesExact;
  exact.Search(15, neighborsExact, distancesExact);

  NeighborSearch<NearestNeighborSort, EuclideanDistance, arma::mat, BallTree>
      ballTreeSearch(dataset, DUAL_TREE_MODE, 0.05);
  arma::Mat<size_t> neighborsBallTree;
  arma::mat distancesBallTree;
  ballTreeSearch.Search(15, neighborsBallTree, distancesBallTree);

  for (size_t i = 0; i < neighborsBallTree.n_elem; ++i)
    REQUIRE_RELATIVE_ERR(distancesBallTree(i), distancesExact(i), 0.05);
}

/**
 * Test the spill tree hybrid sp-tree search (defeatist search on overlapping
 * nodes, and backtracking in non-overlapping nodes) against the naive method.
 * This uses only a random reference dataset.
 *
 * Errors are produced if the results are not according to relative error.
 */
BOOST_AUTO_TEST_CASE(SingleSpillTreeTest)
{
  arma::mat dataset;
  dataset.randu(50, 300); // 50 dimensional, 300 points.

  const size_t k = 3;

  KNN exact(dataset);
  arma::Mat<size_t> neighborsExact;
  arma::mat distancesExact;
  exact.Search(dataset, k, neighborsExact, distancesExact);

  double maxDist = 0;
  for (size_t i = 0; i < neighborsExact.n_cols; ++i)
    if (distancesExact(k - 1, i) > maxDist)
      maxDist = distancesExact(k - 1, i);

  // If we are sure that tau is a valid upper bound of the kth nearest neighbor
  // of the query points, then we can be sure that we will satisfy the
  // requirements on the relative error.
  SPTree<EuclideanDistance, NeighborSearchStat<NearestNeighborSort>, arma::mat>
      referenceTree(dataset, maxDist * 1.01 /* tau parameter */);

  NeighborSearch<NearestNeighborSort, EuclideanDistance, arma::mat, SPTree>
      spTreeSearch(std::move(referenceTree), SINGLE_TREE_MODE, 0.05);

  arma::Mat<size_t> neighborsSPTree;
  arma::mat distancesSPTree;
  spTreeSearch.Search(dataset, k, neighborsSPTree, distancesSPTree);

  for (size_t i = 0; i < neighborsSPTree.n_elem; ++i)
    REQUIRE_RELATIVE_ERR(distancesSPTree(i), distancesExact(i), 0.05);
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

  SparseKNN aknn(referenceDataset, DUAL_TREE_MODE, 0.05);
  arma::mat distancesSparse;
  arma::Mat<size_t> neighborsSparse;
  aknn.Search(queryDataset, 10, neighborsSparse, distancesSparse);

  KNN exact(denseReference);
  arma::mat distancesExact;
  arma::Mat<size_t> neighborsExact;
  exact.Search(denseQuery, 10, neighborsExact, distancesExact);

  for (size_t i = 0; i < neighborsExact.n_cols; ++i)
    for (size_t j = 0; j < neighborsExact.n_rows; ++j)
      REQUIRE_RELATIVE_ERR(distancesSparse(j, i), distancesExact(j, i), 0.05);
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
  KNNModel models[26];
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

  for (size_t j = 0; j < 3; ++j)
  {
    // Get a baseline.
    KNN aknn(referenceData);
    arma::Mat<size_t> neighborsExact;
    arma::mat distancesExact;
    aknn.Search(queryData, 3, neighborsExact, distancesExact);

    for (size_t i = 0; i < 26; ++i)
    {
      // We only have std::move() constructors so make a copy of our data.
      arma::mat referenceCopy(referenceData);
      arma::mat queryCopy(queryData);
      if (j == 0)
        models[i].BuildModel(std::move(referenceCopy), 20, DUAL_TREE_MODE,
            0.05);
      if (j == 1)
        models[i].BuildModel(std::move(referenceCopy), 20,
            SINGLE_TREE_MODE, 0.05);
      if (j == 2)
        models[i].BuildModel(std::move(referenceCopy), 20, NAIVE_MODE);

      arma::Mat<size_t> neighborsApprox;
      arma::mat distancesApprox;

      models[i].Search(std::move(queryCopy), 3, neighborsApprox,
          distancesApprox);

      BOOST_REQUIRE_EQUAL(neighborsApprox.n_rows, neighborsExact.n_rows);
      BOOST_REQUIRE_EQUAL(neighborsApprox.n_cols, neighborsExact.n_cols);
      BOOST_REQUIRE_EQUAL(neighborsApprox.n_elem, neighborsExact.n_elem);
      BOOST_REQUIRE_EQUAL(distancesApprox.n_rows, distancesExact.n_rows);
      BOOST_REQUIRE_EQUAL(distancesApprox.n_cols, distancesExact.n_cols);
      BOOST_REQUIRE_EQUAL(distancesApprox.n_elem, distancesExact.n_elem);
      for (size_t k = 0; k < distancesApprox.n_elem; ++k)
        REQUIRE_RELATIVE_ERR(distancesApprox[k], distancesExact[k], 0.05);
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
  KNNModel models[26];
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

  for (size_t j = 0; j < 2; ++j)
  {
    // Get a baseline.
    KNN exact(referenceData);
    arma::Mat<size_t> neighborsExact;
    arma::mat distancesExact;
    exact.Search(3, neighborsExact, distancesExact);

    for (size_t i = 0; i < 26; ++i)
    {
      // We only have a std::move() constructor... so copy the data.
      arma::mat referenceCopy(referenceData);
      if (j == 0)
        models[i].BuildModel(std::move(referenceCopy), 20, DUAL_TREE_MODE,
            0.05);
      if (j == 1)
        models[i].BuildModel(std::move(referenceCopy), 20,
            SINGLE_TREE_MODE, 0.05);

      arma::Mat<size_t> neighborsApprox;
      arma::mat distancesApprox;

      models[i].Search(3, neighborsApprox, distancesApprox);

      BOOST_REQUIRE_EQUAL(neighborsApprox.n_rows, neighborsExact.n_rows);
      BOOST_REQUIRE_EQUAL(neighborsApprox.n_cols, neighborsExact.n_cols);
      BOOST_REQUIRE_EQUAL(neighborsApprox.n_elem, neighborsExact.n_elem);
      BOOST_REQUIRE_EQUAL(distancesApprox.n_rows, distancesExact.n_rows);
      BOOST_REQUIRE_EQUAL(distancesApprox.n_cols, distancesExact.n_cols);
      BOOST_REQUIRE_EQUAL(distancesApprox.n_elem, distancesExact.n_elem);
      for (size_t k = 0; k < distancesApprox.n_elem; ++k)
        REQUIRE_RELATIVE_ERR(distancesApprox[k], distancesExact[k], 0.05);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END();
