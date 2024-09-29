/**
 * @file tests/krann_search_test.cpp
 *
 * Unit tests for the 'RASearch' class and consequently the
 * 'RASearchRules' class
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <time.h>
#include <mlpack/core.hpp>
#include <mlpack/methods/rann.hpp>
#include <mlpack/methods/rann/ra_model.hpp>

#include "catch.hpp"

using namespace std;
using namespace mlpack;

// Test the correctness and guarantees of KRANN when in naive mode.
TEST_CASE("NaiveGuaranteeTest", "[KRANNTest]")
{
  arma::Mat<size_t> neighbors;
  arma::mat distances;

  arma::mat refData;
  arma::mat queryData;

  if (!data::Load("rann_test_r_3_900.csv", refData))
    FAIL("Cannot load dataset rann_test_r_3_900.csv");
  if (!data::Load("rann_test_q_3_100.csv", queryData))
    FAIL("Cannot load dataset rann_test_q_3_100.csv");

  RASearch<> rsRann(refData, true, false, 1.0);

  arma::mat qrRanks;
  if (!data::Load("rann_test_qr_ranks.csv", qrRanks, false, false))
    FAIL("Cannot load dataset rann_test_qr_ranks.csv");

  size_t numRounds = 1000;
  arma::Col<size_t> numSuccessRounds(queryData.n_cols);
  numSuccessRounds.fill(0);

  // 1% of 900 is 9, so the rank is expected to be less than 10
  size_t expectedRankErrorUB = 10;

  for (size_t rounds = 0; rounds < numRounds; rounds++)
  {
    rsRann.Search(queryData, 1, neighbors, distances);

    for (size_t i = 0; i < queryData.n_cols; ++i)
      if (qrRanks(i, neighbors(0, i)) < expectedRankErrorUB)
        numSuccessRounds[i]++;

    neighbors.reset();
    distances.reset();
  }

  // Find the 95%-tile threshold so that 95% of the queries should pass this
  // threshold.
  size_t threshold = floor(numRounds *
      (0.95 - (1.96 * sqrt(0.95 * 0.05 / numRounds))));
  size_t numQueriesFail = 0;
  for (size_t i = 0; i < queryData.n_cols; ++i)
    if (numSuccessRounds[i] < threshold)
      numQueriesFail++;

  Log::Warn << "RANN-RS: RANN guarantee fails on " << numQueriesFail
      << " queries." << endl;

  // assert that at most 5% of the queries fall out of this threshold
  // 5% of 100 queries is 5.
  size_t maxNumQueriesFail = 6;

  REQUIRE(numQueriesFail < maxNumQueriesFail);
}

// Test single-tree rank-approximate search (harder to test because of
// the randomness involved).
TEST_CASE("SingleTreeSearch", "[KRANNTest]")
{
  arma::mat refData;
  arma::mat queryData;

  if (!data::Load("rann_test_r_3_900.csv", refData))
    FAIL("Cannot load dataset rann_test_r_3_900.csv");
  if (!data::Load("rann_test_q_3_100.csv", queryData))
    FAIL("Cannot load dataset rann_test_q_3_100.csv");

  // Search for 1 rank-approximate nearest-neighbors in the top 30% of the point
  // (rank error of 3).
  arma::Mat<size_t> neighbors;
  arma::mat distances;

  RASearch<> tssRann(refData, false, true, 1.0, 0.95, false, false);

  // The relative ranks for the given query reference pair
  arma::Mat<size_t> qrRanks;
  if (!data::Load("rann_test_qr_ranks.csv", qrRanks, false, false))
    FAIL("Cannot load dataset rann_test_qr_ranks.csv");

  size_t numRounds = 1000;
  arma::Col<size_t> numSuccessRounds(queryData.n_cols);
  numSuccessRounds.fill(0);

  // 1% of 900 is 9, so the rank is expected to be less than 10.
  size_t expectedRankErrorUB = 10;

  for (size_t rounds = 0; rounds < numRounds; rounds++)
  {
    tssRann.Search(queryData, 1, neighbors, distances);

    for (size_t i = 0; i < queryData.n_cols; ++i)
      if (qrRanks(i, neighbors(0, i)) < expectedRankErrorUB)
        numSuccessRounds[i]++;

    neighbors.reset();
    distances.reset();
  }

  // Find the 95%-tile threshold so that 95% of the queries should pass this
  // threshold.
  size_t threshold = floor(numRounds *
      (0.95 - (1.96 * sqrt(0.95 * 0.05 / numRounds))));
  size_t numQueriesFail = 0;
  for (size_t i = 0; i < queryData.n_cols; ++i)
    if (numSuccessRounds[i] < threshold)
      numQueriesFail++;

  Log::Warn << "RANN-TSS: RANN guarantee fails on " << numQueriesFail
      << " queries." << endl;

  // Assert that at most 5% of the queries fall out of this threshold.
  // 5% of 100 queries is 5.
  size_t maxNumQueriesFail = 6;

  REQUIRE(numQueriesFail < maxNumQueriesFail);
}

// Test dual-tree rank-approximate search (harder to test because of the
// randomness involved).
TEST_CASE("DualTreeSearch", "[KRANNTest]")
{
  arma::mat refData;
  arma::mat queryData;

  if (!data::Load("rann_test_r_3_900.csv", refData))
    FAIL("Cannot load dataset rann_test_r_3_900.csv");
  if (!data::Load("rann_test_q_3_100.csv", queryData))
    FAIL("Cannot load dataset rann_test_q_3_100.csv");

  // Search for 1 rank-approximate nearest-neighbors in the top 30% of the point
  // (rank error of 3).
  arma::Mat<size_t> neighbors;
  arma::mat distances;

  RASearch<> tsdRann(refData, false, false, 1.0, 0.95, false, false, 5);

  arma::Mat<size_t> qrRanks;
  if (!data::Load("rann_test_qr_ranks.csv", qrRanks, false, false))
    FAIL("Cannot load dataset rann_test_qr_ranks.csv");

  size_t numRounds = 1000;
  arma::Col<size_t> numSuccessRounds(queryData.n_cols);
  numSuccessRounds.fill(0);

  // 1% of 900 is 9, so the rank is expected to be less than 10.
  size_t expectedRankErrorUB = 10;

  // Build query tree by hand.
  typedef KDTree<EuclideanDistance, RAQueryStat<NearestNeighborSort>,
      arma::mat> TreeType;
  std::vector<size_t> oldFromNewQueries;
  TreeType queryTree(queryData, oldFromNewQueries);

  for (size_t rounds = 0; rounds < numRounds; rounds++)
  {
    tsdRann.Search(&queryTree, 1, neighbors, distances);

    for (size_t i = 0; i < queryData.n_cols; ++i)
    {
      const size_t oldIndex = oldFromNewQueries[i];
      if (qrRanks(oldIndex, neighbors(0, i)) < expectedRankErrorUB)
        numSuccessRounds[i]++;
    }

    neighbors.reset();
    distances.reset();

    tsdRann.ResetQueryTree(&queryTree);
  }

  // Find the 95%-tile threshold so that 95% of the queries should pass this
  // threshold.
  size_t threshold = floor(numRounds *
      (0.95 - (1.96 * sqrt(0.95 * 0.05 / numRounds))));
  size_t numQueriesFail = 0;
  for (size_t i = 0; i < queryData.n_cols; ++i)
    if (numSuccessRounds[i] < threshold)
      numQueriesFail++;

  Log::Warn << "RANN-TSD: RANN guarantee fails on " << numQueriesFail
      << " queries." << endl;

  // assert that at most 5% of the queries fall out of this threshold
  // 5% of 100 queries is 5.
  size_t maxNumQueriesFail = 6;

  REQUIRE(numQueriesFail < maxNumQueriesFail);
}

// Test rank-approximate search with just a single dataset.  These tests just
// ensure that the method runs okay.
TEST_CASE("SingleDatasetNaiveSearch", "[KRANNTest]")
{
  arma::mat dataset(5, 2500);
  dataset.randn();

  arma::Mat<size_t> neighbors;
  arma::mat distances;

  RASearch<> naive(dataset, true);

  naive.Search(1, neighbors, distances);

  REQUIRE(neighbors.n_rows == 1);
  REQUIRE(neighbors.n_cols == 2500);
  REQUIRE(distances.n_rows == 1);
  REQUIRE(distances.n_cols == 2500);
}

// Test rank-approximate search with just a single dataset in single-tree mode.
// These tests just ensure that the method runs okay.
TEST_CASE("SingleDatasetSingleSearch", "[KRANNTest]")
{
  arma::mat dataset(5, 2500);
  dataset.randn();

  arma::Mat<size_t> neighbors;
  arma::mat distances;

  RASearch<> single(dataset, false, true);

  single.Search(1, neighbors, distances);

  REQUIRE(neighbors.n_rows == 1);
  REQUIRE(neighbors.n_cols == 2500);
  REQUIRE(distances.n_rows == 1);
  REQUIRE(distances.n_cols == 2500);
}

// Test rank-approximate search with just a single dataset in dual-tree mode.
// These tests just ensure that the method runs okay.
TEST_CASE("SingleDatasetSearch", "[KRANNTest]")
{
  arma::mat dataset(5, 2500);
  dataset.randn();

  arma::Mat<size_t> neighbors;
  arma::mat distances;

  RASearch<> allkrann(dataset);
  allkrann.Search(1, neighbors, distances);

  REQUIRE(neighbors.n_rows == 1);
  REQUIRE(neighbors.n_cols == 2500);
  REQUIRE(distances.n_rows == 1);
  REQUIRE(distances.n_cols == 2500);
}

// Test single-tree rank-approximate search with cover trees.
TEST_CASE("SingleCoverTreeTest", "[KRANNTest]")
{
  arma::mat refData;
  arma::mat queryData;

  if (!data::Load("rann_test_r_3_900.csv", refData))
    FAIL("Cannot load dataset rann_test_r_3_900.csv");
  if (!data::Load("rann_test_q_3_100.csv", queryData))
    FAIL("Cannot load dataset rann_test_q_3_100.csv");

  // Search for 1 rank-approximate nearest-neighbors in the top 30% of the point
  // (rank error of 3).
  arma::Mat<size_t> neighbors;
  arma::mat distances;

  typedef RASearch<NearestNeighborSort, EuclideanDistance, arma::mat,
      StandardCoverTree> RACoverTreeSearch;

  RACoverTreeSearch tssRann(refData, false, true, 1.0, 0.95, false, false, 5);

  // The relative ranks for the given query reference pair.
  arma::Mat<size_t> qrRanks;
  if (!data::Load("rann_test_qr_ranks.csv", qrRanks, false, false))
    FAIL("Cannot load dataset rann_test_qr_ranks.csv");

  size_t numRounds = 100;
  arma::Col<size_t> numSuccessRounds(queryData.n_cols);
  numSuccessRounds.fill(0);

  // 1% of 900 is 9, so the rank is expected to be less than 10.
  size_t expectedRankErrorUB = 10;

  for (size_t rounds = 0; rounds < numRounds; rounds++)
  {
    tssRann.Search(queryData, 1, neighbors, distances);

    for (size_t i = 0; i < queryData.n_cols; ++i)
      if (qrRanks(i, neighbors(0, i)) < expectedRankErrorUB)
        numSuccessRounds[i]++;

    neighbors.reset();
    distances.reset();
  }

  // Find the 95%-tile threshold so that 95% of the queries should pass this
  // threshold.
  size_t threshold = floor(numRounds *
      (0.95 - (1.96 * sqrt(0.95 * 0.05 / numRounds))));
  size_t numQueriesFail = 0;
  for (size_t i = 0; i < queryData.n_cols; ++i)
    if (numSuccessRounds[i] < threshold)
      numQueriesFail++;

  Log::Warn << "RANN-TSS (cover tree): RANN guarantee fails on "
      << numQueriesFail << " queries." << endl;

  // Assert that at most 5% of the queries fall out of this threshold.
  // 5% of 100 queries is 5.
  size_t maxNumQueriesFail = 6;

  REQUIRE(numQueriesFail < maxNumQueriesFail);
}

// Test dual-tree rank-approximate search with cover trees.
TEST_CASE("DualCoverTreeTest", "[KRANNTest]")
{
  arma::mat refData;
  arma::mat queryData;

  if (!data::Load("rann_test_r_3_900.csv", refData))
    FAIL("Cannot load dataset rann_test_r_3_900.csv");
  if (!data::Load("rann_test_q_3_100.csv", queryData))
    FAIL("Cannot load dataset rann_test_q_3_100.csv");

  // Search for 1 rank-approximate nearest-neighbors in the top 30% of the point
  // (rank error of 3).
  arma::Mat<size_t> neighbors;
  arma::mat distances;

  typedef StandardCoverTree<EuclideanDistance, RAQueryStat<NearestNeighborSort>,
      arma::mat> TreeType;
  typedef RASearch<NearestNeighborSort, EuclideanDistance, arma::mat,
      StandardCoverTree> RACoverTreeSearch;

  TreeType refTree(refData);
  TreeType queryTree(queryData);

  RACoverTreeSearch tsdRann(&refTree, false, 1.0, 0.95, false, false, 5);

  arma::Mat<size_t> qrRanks;
  // No transpose.
  if (!data::Load("rann_test_qr_ranks.csv", qrRanks, false, false))
    FAIL("Cannot load dataset rann_test_qr_ranks.csv");

  size_t numRounds = 100;
  arma::Col<size_t> numSuccessRounds(queryData.n_cols);
  numSuccessRounds.fill(0);

  // 1% of 900 is 9, so the rank is expected to be less than 10.
  size_t expectedRankErrorUB = 10;

  for (size_t rounds = 0; rounds < numRounds; rounds++)
  {
    tsdRann.Search(&queryTree, 1, neighbors, distances);

    for (size_t i = 0; i < queryData.n_cols; ++i)
      if (qrRanks(i, neighbors(0, i)) < expectedRankErrorUB)
        numSuccessRounds[i]++;

    neighbors.reset();
    distances.reset();

    tsdRann.ResetQueryTree(&queryTree);
  }

  // Find the 95%-tile threshold so that 95% of the queries should pass this
  // threshold.
  size_t threshold = floor(numRounds *
      (0.95 - (1.96 * sqrt(0.95 * 0.05 / numRounds))));
  size_t numQueriesFail = 0;
  for (size_t i = 0; i < queryData.n_cols; ++i)
    if (numSuccessRounds[i] < threshold)
      numQueriesFail++;

  Log::Warn << "RANN-TSD (cover tree): RANN guarantee fails on "
      << numQueriesFail << " queries." << endl;

  // assert that at most 5% of the queries fall out of this threshold
  // 5% of 100 queries is 5.
  size_t maxNumQueriesFail = 6;

  REQUIRE(numQueriesFail < maxNumQueriesFail);
}

// Test single-tree rank-approximate search with ball trees.
// This is known to not work right now.
/*
TEST_CASE("SingleBallTreeTest", "[KRANNTest]")
{
  arma::mat refData;
  arma::mat queryData;

  data::Load("rann_test_r_3_900.csv", refData, true);
  data::Load("rann_test_q_3_100.csv", queryData, true);

  // Search for 1 rank-approximate nearest-neighbors in the top 30% of the point
  // (rank error of 3).
  arma::Mat<size_t> neighbors;
  arma::mat distances;

  typedef BinarySpaceTree<BallBound<>, RAQueryStat<NearestNeighborSort> >
      TreeType;
  typedef RASearch<NearestNeighborSort, EuclideanDistance, TreeType>
      RABallTreeSearch;

  RABallTreeSearch tssRann(refData, queryData, false, true);

  // The relative ranks for the given query reference pair.
  arma::Mat<size_t> qrRanks;
  data::Load("rann_test_qr_ranks.csv", qrRanks, true, false); // No transpose.

  size_t numRounds = 30;
  arma::Col<size_t> numSuccessRounds(queryData.n_cols);
  numSuccessRounds.fill(0);

  // 1% of 900 is 9, so the rank is expected to be less than 10.
  size_t expectedRankErrorUB = 10;

  for (size_t rounds = 0; rounds < numRounds; rounds++)
  {
    tssRann.Search(1, neighbors, distances, 1.0, 0.95, false, false, 5);

    for (size_t i = 0; i < queryData.n_cols; ++i)
      if (qrRanks(i, neighbors(0, i)) < expectedRankErrorUB)
        numSuccessRounds[i]++;

    neighbors.reset();
    distances.reset();
  }

  // Find the 95%-tile threshold so that 95% of the queries should pass this
  // threshold.
  size_t threshold = floor(numRounds *
      (0.95 - (1.96 * sqrt(0.95 * 0.05 / numRounds))));
  size_t numQueriesFail = 0;
  for (size_t i = 0; i < queryData.n_cols; ++i)
    if (numSuccessRounds[i] < threshold)
      numQueriesFail++;

  Log::Warn << "RANN-TSS (ball tree): RANN guarantee fails on "
      << numQueriesFail << " queries." << endl;

  // Assert that at most 5% of the queries fall out of this threshold.
  // 5% of 100 queries is 5.
  size_t maxNumQueriesFail = 6;

  REQUIRE(numQueriesFail < maxNumQueriesFail);
}

// Test dual-tree rank-approximate search with Ball trees.
TEST_CASE("DualBallTreeTest", "[KRANNTest]")
{
  arma::mat refData;
  arma::mat queryData;

  data::Load("rann_test_r_3_900.csv", refData, true);
  data::Load("rann_test_q_3_100.csv", queryData, true);

  // Search for 1 rank-approximate nearest-neighbors in the top 30% of the point
  // (rank error of 3).
  arma::Mat<size_t> neighbors;
  arma::mat distances;

  typedef BinarySpaceTree<BallBound<>, RAQueryStat<NearestNeighborSort> >
    TreeType;
  typedef RASearch<NearestNeighborSort, EuclideanDistance, TreeType>
      RABallTreeSearch;

  TreeType refTree(refData);
  TreeType queryTree(queryData);

  RABallTreeSearch tsdRann(&refTree, &queryTree, refData, queryData, false);

  arma::Mat<size_t> qrRanks;
  data::Load("rann_test_qr_ranks.csv", qrRanks, true, false); // No transpose.

  size_t numRounds = 1000;
  arma::Col<size_t> numSuccessRounds(queryData.n_cols);
  numSuccessRounds.fill(0);

  // 1% of 900 is 9, so the rank is expected to be less than 10.
  size_t expectedRankErrorUB = 10;

  for (size_t rounds = 0; rounds < numRounds; rounds++)
  {
    tsdRann.Search(1, neighbors, distances, 1.0, 0.95, false, false, 5);

    for (size_t i = 0; i < queryData.n_cols; ++i)
      if (qrRanks(i, neighbors(0, i)) < expectedRankErrorUB)
        numSuccessRounds[i]++;

    neighbors.reset();
    distances.reset();

    tsdRann.ResetQueryTree();
  }

  // Find the 95%-tile threshold so that 95% of the queries should pass this
  // threshold.
  size_t threshold = floor(numRounds *
      (0.95 - (1.96 * sqrt(0.95 * 0.05 / numRounds))));
  size_t numQueriesFail = 0;
  for (size_t i = 0; i < queryData.n_cols; ++i)
    if (numSuccessRounds[i] < threshold)
      numQueriesFail++;

  Log::Warn << "RANN-TSD (Ball tree): RANN guarantee fails on "
      << numQueriesFail << " queries." << endl;

  // assert that at most 5% of the queries fall out of this threshold
  // 5% of 100 queries is 5.
  size_t maxNumQueriesFail = 6;

  REQUIRE(numQueriesFail < maxNumQueriesFail);
}
*/

/**
 * Make sure that the neighborPtr matrix isn't accidentally deleted.
 * See issue #478.
 */
TEST_CASE("KRANNNeighborPtrDeleteTest", "[KRANNTest]")
{
  arma::mat dataset = arma::randu<arma::mat>(5, 100);

  // Build the tree ourselves.
  std::vector<size_t> oldFromNewReferences;
  RASearch<>::Tree tree(dataset);
  RASearch<> allkrann(&tree);

  // Now make a query set.
  arma::mat queryset = arma::randu<arma::mat>(5, 50);
  arma::mat distances;
  arma::Mat<size_t> neighbors;
  allkrann.Search(queryset, 3, neighbors, distances);

  // These will (hopefully) fail is either the neighbors or the distances matrix
  // has been accidentally deleted.
  REQUIRE(neighbors.n_cols == 50);
  REQUIRE(neighbors.n_rows == 3);
  REQUIRE(distances.n_cols == 50);
  REQUIRE(distances.n_rows == 3);
}

/**
 * Test that the rvalue reference move constructor works.
 */
TEST_CASE("KRANNMoveConstructorTest", "[KRANNTest]")
{
  arma::mat dataset = arma::randu<arma::mat>(3, 200);
  arma::mat copy(dataset);

  KRANN moveknn(std::move(copy));
  KRANN knn(dataset);

  REQUIRE(copy.n_elem == 0);
  REQUIRE(moveknn.ReferenceSet().n_rows == 3);
  REQUIRE(moveknn.ReferenceSet().n_cols == 200);

  arma::mat moveDistances, distances;
  arma::Mat<size_t> moveNeighbors, neighbors;

  moveknn.Search(1, moveNeighbors, moveDistances);
  knn.Search(1, neighbors, distances);

  REQUIRE(moveNeighbors.n_rows == neighbors.n_rows);
  REQUIRE(moveNeighbors.n_rows == neighbors.n_rows);
  REQUIRE(moveNeighbors.n_cols == neighbors.n_cols);
  REQUIRE(moveDistances.n_rows == distances.n_rows);
  REQUIRE(moveDistances.n_cols == distances.n_cols);
}

/**
 * Test that the dataset can be retrained with the move Train() function.
 */
TEST_CASE("KRANNMoveTrainTest", "[KRANNTest]")
{
  arma::mat dataset = arma::randu<arma::mat>(3, 200);

  // Do it in tree mode, and in naive mode.
  KRANN knn;
  knn.Train(std::move(dataset));

  arma::mat distances;
  arma::Mat<size_t> neighbors;
  knn.Search(1, neighbors, distances);

  REQUIRE(dataset.n_elem == 0);
  REQUIRE(neighbors.n_cols == 200);
  REQUIRE(distances.n_cols == 200);

  dataset = arma::randu<arma::mat>(3, 300);
  knn.Naive() = true;
  knn.Train(std::move(dataset));
  knn.Search(1, neighbors, distances);

  REQUIRE(dataset.n_elem == 0);
  REQUIRE(neighbors.n_cols == 300);
  REQUIRE(distances.n_cols == 300);
}

/**
 * Make sure the RAModel class works.
 */
TEST_CASE("RAModelTest", "[KRANNTest]")
{
  // Ensure that we can build an RAModel<NearestNeighborSearch> and get correct
  // results.
  arma::mat queryData, referenceData;
  if (!data::Load("rann_test_r_3_900.csv", referenceData))
    FAIL("Cannot load dataset rann_test_r_3_900.csv");
  if (!data::Load("rann_test_q_3_100.csv", queryData))
    FAIL("Cannot load dataset rann_test_q_3_100.csv");

  // Build all the possible models.
  RAModel models[20];
  models[0] = RAModel(RAModel::TreeTypes::KD_TREE, false);
  models[1] = RAModel(RAModel::TreeTypes::KD_TREE, true);
  models[2] = RAModel(RAModel::TreeTypes::COVER_TREE, false);
  models[3] = RAModel(RAModel::TreeTypes::COVER_TREE, true);
  models[4] = RAModel(RAModel::TreeTypes::R_TREE, false);
  models[5] = RAModel(RAModel::TreeTypes::R_TREE, true);
  models[6] = RAModel(RAModel::TreeTypes::R_STAR_TREE, false);
  models[7] = RAModel(RAModel::TreeTypes::R_STAR_TREE, true);
  models[8] = RAModel(RAModel::TreeTypes::X_TREE, false);
  models[9] = RAModel(RAModel::TreeTypes::X_TREE, true);
  models[10] = RAModel(RAModel::TreeTypes::HILBERT_R_TREE, false);
  models[11] = RAModel(RAModel::TreeTypes::HILBERT_R_TREE, true);
  models[12] = RAModel(RAModel::TreeTypes::R_PLUS_TREE, false);
  models[13] = RAModel(RAModel::TreeTypes::R_PLUS_TREE, true);
  models[14] = RAModel(RAModel::TreeTypes::R_PLUS_PLUS_TREE, false);
  models[15] = RAModel(RAModel::TreeTypes::R_PLUS_PLUS_TREE, true);
  models[16] = RAModel(RAModel::TreeTypes::UB_TREE, false);
  models[17] = RAModel(RAModel::TreeTypes::UB_TREE, true);
  models[18] = RAModel(RAModel::TreeTypes::OCTREE, false);
  models[19] = RAModel(RAModel::TreeTypes::OCTREE, true);

  util::Timers timers;

  arma::Mat<size_t> qrRanks;
  if (!data::Load("rann_test_qr_ranks.csv", qrRanks, false, false))
    FAIL("Cannot load dataset rann_test_qr_ranks.csv");

  for (size_t j = 0; j < 3; ++j)
  {
    for (size_t i = 0; i < 20; ++i)
    {
      // We only have std::move() constructors so make a copy of our data.
      arma::mat referenceCopy(referenceData);
      if (j == 0)
      {
        models[i].BuildModel(timers, std::move(referenceCopy), 20, false,
            false);
      }
      if (j == 1)
      {
        models[i].BuildModel(timers, std::move(referenceCopy), 20, false, true);
      }
      if (j == 2)
      {
        models[i].BuildModel(timers, std::move(referenceCopy), 20, true, false);
      }

      // Set the search parameters.
      models[i].Tau() = 1.0;
      models[i].Alpha() = 0.95;
      models[i].SampleAtLeaves() = false;
      models[i].FirstLeafExact() = false;
      models[i].SingleSampleLimit() = 5;

      arma::Mat<size_t> neighbors;
      arma::mat distances;

      arma::Col<size_t> numSuccessRounds(queryData.n_cols);
      numSuccessRounds.fill(0);

      // 1% of 900 is 9, so the rank is expected to be less than 10.
      size_t expectedRankErrorUB = 10;

      size_t numRounds = 100;
      for (size_t round = 0; round < numRounds; round++)
      {
        arma::mat queryCopy(queryData);
        models[i].Search(timers, std::move(queryCopy), 1, neighbors, distances);
        for (size_t k = 0; k < queryData.n_cols; ++k)
          if (qrRanks(k, neighbors(0, k)) < expectedRankErrorUB)
            numSuccessRounds[k]++;

        neighbors.reset();
        distances.reset();
      }

      // Find the 95%-tile threshold so that 95% of the queries should pass this
      // threshold.
      size_t threshold = floor(numRounds *
          (0.95 - (1.96 * sqrt(0.95 * 0.05 / numRounds))));
      size_t numQueriesFail = 0;
      for (size_t k = 0; k < queryData.n_cols; ++k)
        if (numSuccessRounds[k] < threshold)
          numQueriesFail++;

      // assert that at most 5% of the queries fall out of this threshold
      // 5% of 100 queries is 5.
      size_t maxNumQueriesFail = 50; // See #734 for why this is so high.

      REQUIRE(numQueriesFail < maxNumQueriesFail);
    }
  }
}
