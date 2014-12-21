/**
 * @file allkrann_search_test.cpp
 *
 * Unit tests for the 'RASearch' class and consequently the
 * 'RASearchRules' class
 *
 * This file is part of MLPACK 1.0.9.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
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

// Test AllkRANN in naive mode for exact results when the random seeds are set
// the same.  This may not be the best test; if the implementation of RANN-RS
// gets random numbers in a different way, then this test might fail.
BOOST_AUTO_TEST_CASE(NaiveSearchExact)
{
  // First test on a small set.
  arma::mat rdata(2, 10);
  rdata << 3 << 2 << 4 << 3 << 5 << 6 << 0 << 8 << 3 << 1 << arma::endr <<
           0 << 3 << 4 << 7 << 8 << 4 << 1 << 0 << 4 << 3 << arma::endr;

  arma::mat qdata(2, 3);
  qdata << 3 << 2 << 0 << arma::endr
        << 5 << 3 << 4 << arma::endr;

  metric::SquaredEuclideanDistance dMetric;
  double rankApproximation = 30;
  double successProb = 0.95;

  // Search for 1 rank-approximate nearest-neighbors in the top 30% of the point
  // (rank error of 3).
  arma::Mat<size_t> neighbors;
  arma::mat distances;

  // Test naive rank-approximate search.
  // Predict what the actual RANN-RS result would be.
  math::RandomSeed(0);

  size_t numSamples = (size_t) ceil(log(1.0 / (1.0 - successProb)) /
      log(1.0 / (1.0 - (rankApproximation / 100.0))));

  arma::Mat<size_t> samples(qdata.n_cols, numSamples);
  for (size_t j = 0; j < qdata.n_cols; j++)
    for (size_t i = 0; i < numSamples; i++)
      samples(j, i) = (size_t) math::RandInt(10);

  arma::Col<size_t> rann(qdata.n_cols);
  arma::vec rannDistances(qdata.n_cols);
  rannDistances.fill(DBL_MAX);

  for (size_t j = 0; j < qdata.n_cols; j++)
  {
    for (size_t i = 0; i < numSamples; i++)
    {
      double dist = dMetric.Evaluate(qdata.unsafe_col(j),
                                     rdata.unsafe_col(samples(j, i)));
      if (dist < rannDistances[j])
      {
        rann[j] = samples(j, i);
        rannDistances[j] = dist;
      }
    }
  }

  // Use RANN-RS implementation.
  math::RandomSeed(0);

  RASearch<> naive(rdata, qdata, true);
  naive.Search(1, neighbors, distances, rankApproximation);

  // Things to check:
  //
  // 1. (implicitly) The minimum number of required samples for guaranteed
  //    approximation.
  // 2. (implicitly) Check the samples obtained.
  // 3. Check the neighbor returned.
  for (size_t i = 0; i < qdata.n_cols; i++)
  {
    BOOST_REQUIRE_EQUAL(neighbors(0, i), rann[i]);
    BOOST_REQUIRE_CLOSE(distances(0, i), rannDistances[i], 1e-5);
  }
}

// Test the correctness and guarantees of AllkRANN when in naive mode.
BOOST_AUTO_TEST_CASE(NaiveGuaranteeTest)
{
  arma::Mat<size_t> neighbors;
  arma::mat distances;

  arma::mat refData;
  arma::mat queryData;

  data::Load("rann_test_r_3_900.csv", refData, true);
  data::Load("rann_test_q_3_100.csv", queryData, true);

  RASearch<> rsRann(refData, queryData, true);

  arma::mat qrRanks;
  data::Load("rann_test_qr_ranks.csv", qrRanks, true, false); // No transpose.

  size_t numRounds = 1000;
  arma::Col<size_t> numSuccessRounds(queryData.n_cols);
  numSuccessRounds.fill(0);

  // 1% of 900 is 9, so the rank is expected to be less than 10
  size_t expectedRankErrorUB = 10;

  for (size_t rounds = 0; rounds < numRounds; rounds++)
  {
    rsRann.Search(1, neighbors, distances, 1.0);

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

  RASearch<> tssRann(refData, queryData, false, true);

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
    tssRann.Search(1, neighbors, distances, 1.0, 0.95, false, false);

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

  RASearch<> tsdRann(refData, queryData, false, false);

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

  typedef tree::CoverTree<metric::EuclideanDistance, tree::FirstPointIsRoot,
      RAQueryStat<NearestNeighborSort> > TreeType;
  typedef RASearch<NearestNeighborSort, metric::EuclideanDistance, TreeType>
      RACoverTreeSearch;

  RACoverTreeSearch tssRann(refData, queryData, false, true);

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

  typedef tree::CoverTree<metric::EuclideanDistance, tree::FirstPointIsRoot,
      RAQueryStat<NearestNeighborSort> > TreeType;
  typedef RASearch<NearestNeighborSort, metric::EuclideanDistance, TreeType>
      RACoverTreeSearch;

  TreeType refTree(refData);
  TreeType queryTree(queryData);

  RACoverTreeSearch tsdRann(&refTree, &queryTree, refData, queryData, false);

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
