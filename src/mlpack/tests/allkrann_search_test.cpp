/**
 * @file allkrann_search_test.cpp
 *
 * Unit tests for the 'RASearch' class and consequently the
 * 'RASearchRules' class
 */
#include <time.h>
#include <mlpack/core.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/core/tree/cover_tree.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

#include <mlpack/methods/rann/ra_search.hpp>

using namespace std;
using namespace mlpack;
using namespace mlpack::neighbor;
using namespace mlpack::tree;
using namespace mlpack::metric;
using namespace mlpack::bound;

BOOST_AUTO_TEST_SUITE(AllkRANNTest);

// Test the correctness and guarantees of AllkRANN when in naive mode.
BOOST_AUTO_TEST_CASE(NaiveGuaranteeTest)
{
  arma::Mat<size_t> neighbors;
  arma::mat distances;

  arma::mat refData;
  arma::mat queryData;

  data::Load("rann_test_r_3_900.csv", refData, true);
  data::Load("rann_test_q_3_100.csv", queryData, true);

  RASearch<> rsRann(refData, true, false, 1.0);

  arma::mat qrRanks;
  data::Load("rann_test_qr_ranks.csv", qrRanks, true, false); // No transpose.

  size_t numRounds = 1000;
  arma::Col<size_t> numSuccessRounds(queryData.n_cols);
  numSuccessRounds.fill(0);

  // 1% of 900 is 9, so the rank is expected to be less than 10
  size_t expectedRankErrorUB = 10;

  for (size_t rounds = 0; rounds < numRounds; rounds++)
  {
    rsRann.Search(queryData, 1, neighbors, distances);

    for (size_t i = 0; i < queryData.n_cols; i++)
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
  for (size_t i = 0; i < queryData.n_cols; i++)
    if (numSuccessRounds[i] < threshold)
      numQueriesFail++;

  Log::Warn << "RANN-RS: RANN guarantee fails on " << numQueriesFail
      << " queries." << endl;

  // assert that at most 5% of the queries fall out of this threshold
  // 5% of 100 queries is 5.
  size_t maxNumQueriesFail = 6;

  BOOST_REQUIRE_LT(numQueriesFail, maxNumQueriesFail);
}

// Test single-tree rank-approximate search (harder to test because of
// the randomness involved).
BOOST_AUTO_TEST_CASE(SingleTreeSearch)
{
  arma::mat refData;
  arma::mat queryData;

  data::Load("rann_test_r_3_900.csv", refData, true);
  data::Load("rann_test_q_3_100.csv", queryData, true);

  // Search for 1 rank-approximate nearest-neighbors in the top 30% of the point
  // (rank error of 3).
  arma::Mat<size_t> neighbors;
  arma::mat distances;

  RASearch<> tssRann(refData, false, true, 1.0, 0.95, false, false);

  // The relative ranks for the given query reference pair
  arma::Mat<size_t> qrRanks;
  data::Load("rann_test_qr_ranks.csv", qrRanks, true, false); // No transpose.

  size_t numRounds = 1000;
  arma::Col<size_t> numSuccessRounds(queryData.n_cols);
  numSuccessRounds.fill(0);

  // 1% of 900 is 9, so the rank is expected to be less than 10.
  size_t expectedRankErrorUB = 10;

  for (size_t rounds = 0; rounds < numRounds; rounds++)
  {
    tssRann.Search(queryData, 1, neighbors, distances);

    for (size_t i = 0; i < queryData.n_cols; i++)
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
  for (size_t i = 0; i < queryData.n_cols; i++)
    if (numSuccessRounds[i] < threshold)
      numQueriesFail++;

  Log::Warn << "RANN-TSS: RANN guarantee fails on " << numQueriesFail
      << " queries." << endl;

  // Assert that at most 5% of the queries fall out of this threshold.
  // 5% of 100 queries is 5.
  size_t maxNumQueriesFail = 6;

  BOOST_REQUIRE_LT(numQueriesFail, maxNumQueriesFail);
}

// Test dual-tree rank-approximate search (harder to test because of the
// randomness involved).
BOOST_AUTO_TEST_CASE(DualTreeSearch)
{
  arma::mat refData;
  arma::mat queryData;

  data::Load("rann_test_r_3_900.csv", refData, true);
  data::Load("rann_test_q_3_100.csv", queryData, true);

  // Search for 1 rank-approximate nearest-neighbors in the top 30% of the point
  // (rank error of 3).
  arma::Mat<size_t> neighbors;
  arma::mat distances;

  RASearch<> tsdRann(refData, false, false, 1.0, 0.95, false, false, 5);

  arma::Mat<size_t> qrRanks;
  data::Load("rann_test_qr_ranks.csv", qrRanks, true, false); // No transpose.

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

    for (size_t i = 0; i < queryData.n_cols; i++)
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
  for (size_t i = 0; i < queryData.n_cols; i++)
    if (numSuccessRounds[i] < threshold)
      numQueriesFail++;

  Log::Warn << "RANN-TSD: RANN guarantee fails on " << numQueriesFail
      << " queries." << endl;

  // assert that at most 5% of the queries fall out of this threshold
  // 5% of 100 queries is 5.
  size_t maxNumQueriesFail = 6;

  BOOST_REQUIRE_LT(numQueriesFail, maxNumQueriesFail);
}

// Test rank-approximate search with just a single dataset.  These tests just
// ensure that the method runs okay.
BOOST_AUTO_TEST_CASE(SingleDatasetNaiveSearch)
{
  arma::mat dataset(5, 2500);
  dataset.randn();

  arma::Mat<size_t> neighbors;
  arma::mat distances;

  RASearch<> naive(dataset, true);

  naive.Search(1, neighbors, distances);

  BOOST_REQUIRE_EQUAL(neighbors.n_rows, 1);
  BOOST_REQUIRE_EQUAL(neighbors.n_cols, 2500);
  BOOST_REQUIRE_EQUAL(distances.n_rows, 1);
  BOOST_REQUIRE_EQUAL(distances.n_cols, 2500);
}

// Test rank-approximate search with just a single dataset in single-tree mode.
// These tests just ensure that the method runs okay.
BOOST_AUTO_TEST_CASE(SingleDatasetSingleSearch)
{
  arma::mat dataset(5, 2500);
  dataset.randn();

  arma::Mat<size_t> neighbors;
  arma::mat distances;

  RASearch<> single(dataset, false, true);

  single.Search(1, neighbors, distances);

  BOOST_REQUIRE_EQUAL(neighbors.n_rows, 1);
  BOOST_REQUIRE_EQUAL(neighbors.n_cols, 2500);
  BOOST_REQUIRE_EQUAL(distances.n_rows, 1);
  BOOST_REQUIRE_EQUAL(distances.n_cols, 2500);
}

// Test rank-approximate search with just a single dataset in dual-tree mode.
// These tests just ensure that the method runs okay.
BOOST_AUTO_TEST_CASE(SingleDatasetSearch)
{
  arma::mat dataset(5, 2500);
  dataset.randn();

  arma::Mat<size_t> neighbors;
  arma::mat distances;

  RASearch<> allkrann(dataset);
  allkrann.Search(1, neighbors, distances);

  BOOST_REQUIRE_EQUAL(neighbors.n_rows, 1);
  BOOST_REQUIRE_EQUAL(neighbors.n_cols, 2500);
  BOOST_REQUIRE_EQUAL(distances.n_rows, 1);
  BOOST_REQUIRE_EQUAL(distances.n_cols, 2500);
}

// Test single-tree rank-approximate search with cover trees.
BOOST_AUTO_TEST_CASE(SingleCoverTreeTest)
{
  arma::mat refData;
  arma::mat queryData;

  data::Load("rann_test_r_3_900.csv", refData, true);
  data::Load("rann_test_q_3_100.csv", queryData, true);

  // Search for 1 rank-approximate nearest-neighbors in the top 30% of the point
  // (rank error of 3).
  arma::Mat<size_t> neighbors;
  arma::mat distances;

  typedef RASearch<NearestNeighborSort, EuclideanDistance, arma::mat,
      StandardCoverTree> RACoverTreeSearch;

  RACoverTreeSearch tssRann(refData, false, true, 1.0, 0.95, false, false, 5);

  // The relative ranks for the given query reference pair.
  arma::Mat<size_t> qrRanks;
  data::Load("rann_test_qr_ranks.csv", qrRanks, true, false); // No transpose.

  size_t numRounds = 1000;
  arma::Col<size_t> numSuccessRounds(queryData.n_cols);
  numSuccessRounds.fill(0);

  // 1% of 900 is 9, so the rank is expected to be less than 10.
  size_t expectedRankErrorUB = 10;

  for (size_t rounds = 0; rounds < numRounds; rounds++)
  {
    tssRann.Search(queryData, 1, neighbors, distances);

    for (size_t i = 0; i < queryData.n_cols; i++)
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
  for (size_t i = 0; i < queryData.n_cols; i++)
    if (numSuccessRounds[i] < threshold)
      numQueriesFail++;

  Log::Warn << "RANN-TSS (cover tree): RANN guarantee fails on "
      << numQueriesFail << " queries." << endl;

  // Assert that at most 5% of the queries fall out of this threshold.
  // 5% of 100 queries is 5.
  size_t maxNumQueriesFail = 6;

  BOOST_REQUIRE_LT(numQueriesFail, maxNumQueriesFail);
}

// Test dual-tree rank-approximate search with cover trees.
BOOST_AUTO_TEST_CASE(DualCoverTreeTest)
{
  arma::mat refData;
  arma::mat queryData;

  data::Load("rann_test_r_3_900.csv", refData, true);
  data::Load("rann_test_q_3_100.csv", queryData, true);

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
  data::Load("rann_test_qr_ranks.csv", qrRanks, true, false); // No transpose.

  size_t numRounds = 1000;
  arma::Col<size_t> numSuccessRounds(queryData.n_cols);
  numSuccessRounds.fill(0);

  // 1% of 900 is 9, so the rank is expected to be less than 10.
  size_t expectedRankErrorUB = 10;

  for (size_t rounds = 0; rounds < numRounds; rounds++)
  {
    tsdRann.Search(&queryTree, 1, neighbors, distances);

    for (size_t i = 0; i < queryData.n_cols; i++)
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
  for (size_t i = 0; i < queryData.n_cols; i++)
    if (numSuccessRounds[i] < threshold)
      numQueriesFail++;

  Log::Warn << "RANN-TSD (cover tree): RANN guarantee fails on "
      << numQueriesFail << " queries." << endl;

  // assert that at most 5% of the queries fall out of this threshold
  // 5% of 100 queries is 5.
  size_t maxNumQueriesFail = 6;

  BOOST_REQUIRE_LT(numQueriesFail, maxNumQueriesFail);
}

// Test single-tree rank-approximate search with ball trees.
// This is known to not work right now.
/*
BOOST_AUTO_TEST_CASE(SingleBallTreeTest)
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
  typedef RASearch<NearestNeighborSort, metric::EuclideanDistance, TreeType>
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

    for (size_t i = 0; i < queryData.n_cols; i++)
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
  for (size_t i = 0; i < queryData.n_cols; i++)
    if (numSuccessRounds[i] < threshold)
      numQueriesFail++;

  Log::Warn << "RANN-TSS (ball tree): RANN guarantee fails on "
      << numQueriesFail << " queries." << endl;

  // Assert that at most 5% of the queries fall out of this threshold.
  // 5% of 100 queries is 5.
  size_t maxNumQueriesFail = 6;

  BOOST_REQUIRE_LT(numQueriesFail, maxNumQueriesFail);
}

// Test dual-tree rank-approximate search with Ball trees.
BOOST_AUTO_TEST_CASE(DualBallTreeTest)
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
  typedef RASearch<NearestNeighborSort, metric::EuclideanDistance, TreeType>
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

    for (size_t i = 0; i < queryData.n_cols; i++)
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
  for (size_t i = 0; i < queryData.n_cols; i++)
    if (numSuccessRounds[i] < threshold)
      numQueriesFail++;

  Log::Warn << "RANN-TSD (Ball tree): RANN guarantee fails on "
      << numQueriesFail << " queries." << endl;

  // assert that at most 5% of the queries fall out of this threshold
  // 5% of 100 queries is 5.
  size_t maxNumQueriesFail = 6;

  BOOST_REQUIRE_LT(numQueriesFail, maxNumQueriesFail);
}
*/

BOOST_AUTO_TEST_SUITE_END();
