/**
 * @file lsh_test.cpp
 *
 * Unit tests for the 'LSHSearch' class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

#include <mlpack/methods/lsh/lsh_search.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>

using namespace std;
using namespace mlpack;
using namespace mlpack::neighbor;

/**
 * Generates a point set of four clusters:
 * -C1 around (0.5, 3.5),
 * -C2 around (3.5, 3.5),
 * -C3 around (0.5, 0.5),
 * -C4 around (3.5, 3.5).
 *
 * It then merges these clusters into one set, rdata.
 */
void GetPointset(const size_t N, arma::mat& rdata)
{
  const size_t d = 2;
  // Create four clusters of points.
  arma::mat c1(d, N / 4, arma::fill::randu);
  arma::mat c2(d, N / 4, arma::fill::randu);
  arma::mat c3(d, N / 4, arma::fill::randu);
  arma::mat c4(d, N / 4, arma::fill::randu);

  arma::colvec offset1;
  offset1 << 0 << arma::endr
          << 3 << arma::endr;

  arma::colvec offset2;
  offset2 << 3 << arma::endr
          << 3 << arma::endr;

  arma::colvec offset4;
  offset4 << 3 << arma::endr
          << 0 << arma::endr;

  // Spread points in plane.
  for (size_t p = 0; p < N / 4; ++p)
  {
    c1.col(p) += offset1;
    c2.col(p) += offset2;
    c4.col(p) += offset4;
  }

  rdata.set_size(d, N);
  rdata.cols(0, (N / 4) - 1) = c1;
  rdata.cols(N / 4, (N / 2) - 1) = c2;
  rdata.cols(N / 2, (3 * N / 4) - 1) = c3;
  rdata.cols(3 * N / 4, N - 1) = c4;
}

/**
 * Generates two queries, one around (0.5, 0.5) and one around (3.5, 3.5).
 */
void GetQueries(arma::mat& qdata)
{
  const size_t d = 2;
  // Generate two queries inside two of the clusters.

  // Put query 1 into cluster 3.
  arma::colvec q1, q2;
  q1.randu(d, 1);

  // Offset second query to go into cluster 2.
  q2.randu(d, 1);
  q2.row(0) += 3;
  q2.row(1) += 3;

  qdata.set_size(d, 2);
  qdata.col(0) = q1;
  qdata.col(1) = q2;
}

BOOST_AUTO_TEST_SUITE(LSHTest);

/**
 * Test: Run LSH with varying number of tables, keeping all other parameters
 * constant. Compute the recall, i.e. the number of reported neighbors that
 * are real neighbors of the query.
 * LSH's property is that (with high probability), increasing the number of
 * tables will increase recall. Epsilon ensures that if noise lightly affects
 * the projections, the test will not fail.
 * This produces false negatives, so we attempt the test numTries times and
 * only declare failure if all of them fail.
 */
BOOST_AUTO_TEST_CASE(NumTablesTest)
{
  // kNN and LSH parameters (use LSH default parameters).
  const int k = 4;
  const int numProj = 10;
  const double hashWidth = 0;
  const int secondHashSize = 99901;
  const int bucketSize = 500;

  // Test parameters.
  const double epsilon = 0.1; // Allowed deviation from expected monotonicity.
  const int numTries = 5; // Tries for each test before declaring failure.

  // Read iris training and testing data as reference and query sets.
  const string trainSet = "iris_train.csv";
  const string testSet = "iris_test.csv";
  arma::mat rdata;
  arma::mat qdata;
  data::Load(trainSet, rdata, true);
  data::Load(testSet, qdata, true);

  // Run classic knn on reference data.
  KNN knn(rdata);
  arma::Mat<size_t> groundTruth;
  arma::mat groundDistances;
  knn.Search(qdata, k, groundTruth, groundDistances);

  bool fail;
  for (int t = 0; t < numTries; ++t)
  {
    fail = false;

    const int lSize = 6; // Number of runs.
    const int lValue[] = {1, 8, 16, 32, 64, 128}; // Number of tables.
    double lValueRecall[lSize] = {0.0}; // Recall of each LSH run.

    for (size_t l = 0; l < lSize; ++l)
    {
      // Run LSH with only numTables varying (other values are defaults).
      LSHSearch<> lshTest(rdata, numProj, lValue[l], hashWidth, secondHashSize,
          bucketSize);
      arma::Mat<size_t> lshNeighbors;
      arma::mat lshDistances;
      lshTest.Search(qdata, k, lshNeighbors, lshDistances);

      // Compute recall for each query.
      lValueRecall[l] = LSHSearch<>::ComputeRecall(lshNeighbors, groundTruth);

      if (l > 0)
      {
        if (lValueRecall[l] < lValueRecall[l - 1] - epsilon)
        {
          fail = true; // If test fails at one point, stop and retry.
          break;
        }
      }
    }

    if (!fail)
      break; // If test passes one time, it is sufficient.
  }

  BOOST_REQUIRE(fail == false);
}

/**
 * Test: Run LSH with varying hash width, keeping all other parameters
 * constant. Compute the recall, i.e. the number of reported neighbors that
 * are real neighbors of the query.
 * LSH's property is that (with high probability), increasing the hash width
 * will increase recall. Epsilon ensures that if noise lightly affects the
 * projections, the test will not fail.
 */
BOOST_AUTO_TEST_CASE(HashWidthTest)
{
  // kNN and LSH parameters (use LSH default parameters).
  const int k = 4;
  const int numTables = 30;
  const int numProj = 10;
  const int secondHashSize = 99901;
  const int bucketSize = 500;

  // Test parameters.
  const double epsilon = 0.1; // Allowed deviation from expected monotonicity.

  // Read iris training and testing data as reference and query.
  const string trainSet = "iris_train.csv";
  const string testSet = "iris_test.csv";
  arma::mat rdata;
  arma::mat qdata;
  data::Load(trainSet, rdata, true);
  data::Load(testSet, qdata, true);

  // Run classic knn on reference data.
  KNN knn(rdata);
  arma::Mat<size_t> groundTruth;
  arma::mat groundDistances;
  knn.Search(qdata, k, groundTruth, groundDistances);
  const int hSize = 7; // Number of runs.
  const double hValue[] = {0.1, 0.5, 1, 5, 10, 50, 500}; // Hash width.
  double hValueRecall[hSize] = {0.0}; // Recall of each run.

  for (size_t h = 0; h < hSize; ++h)
  {
    // Run LSH with only hashWidth varying (other values are defaults).
    LSHSearch<> lshTest(
        rdata,
        numProj,
        numTables,
        hValue[h],
        secondHashSize,
        bucketSize);

    arma::Mat<size_t> lshNeighbors;
    arma::mat lshDistances;
    lshTest.Search(qdata, k, lshNeighbors, lshDistances);

    // Compute recall for each query.
    hValueRecall[h] = LSHSearch<>::ComputeRecall(lshNeighbors, groundTruth);

    if (h > 0)
      BOOST_REQUIRE_GE(hValueRecall[h], hValueRecall[h - 1] - epsilon);
  }
}

/**
 * Test: Run LSH with varying number of projections, keeping other parameters
 * constant. Compute the recall, i.e. the number of reported neighbors that
 * are real neighbors of the query.
 * LSH's property is that (with high probability), increasing the number of
 * projections per table will decrease recall. Epsilon ensures that if noise
 * lightly affects the projections, the test will not fail.
 */
BOOST_AUTO_TEST_CASE(NumProjTest)
{
  // kNN and LSH parameters (use LSH default parameters).
  const int k = 4;
  const int numTables = 30;
  const double hashWidth = 0;
  const int secondHashSize = 99901;
  const int bucketSize = 500;

  // Test parameters.
  const double epsilon = 0.1; // Allowed deviation from expected monotonicity.

  // Read iris training and testing data as reference and query sets.
  const string trainSet = "iris_train.csv";
  const string testSet = "iris_test.csv";
  arma::mat rdata;
  arma::mat qdata;
  data::Load(trainSet, rdata, true);
  data::Load(testSet, qdata, true);

  // Run classic knn on reference data.
  KNN knn(rdata);
  arma::Mat<size_t> groundTruth;
  arma::mat groundDistances;
  knn.Search(qdata, k, groundTruth, groundDistances);

  // LSH test parameters for numProj.
  const int pSize = 5; // Number of runs.
  const int pValue[] = {1, 10, 20, 50, 100}; // Number of projections.
  double pValueRecall[pSize] = {0.0}; // Recall of each run.

  for (size_t p = 0; p < pSize; ++p)
  {
    // Run LSH with only numProj varying (other values are defaults).
    LSHSearch<> lshTest(
        rdata,
        pValue[p],
        numTables,
        hashWidth,
        secondHashSize,
        bucketSize);

    arma::Mat<size_t> lshNeighbors;
    arma::mat lshDistances;
    lshTest.Search(qdata, k, lshNeighbors, lshDistances);

    // Compute recall for each query.
    pValueRecall[p] = LSHSearch<>::ComputeRecall(lshNeighbors, groundTruth);

    // Don't check the first run; only check that increasing P decreases recall.
    if (p > 0)
      BOOST_REQUIRE_LE(pValueRecall[p] - epsilon, pValueRecall[p - 1]);
  }
}

/**
 * Test: Run two LSH searches:
 * First, a very expensive LSH search, with a large number of hash tables
 * and a large hash width. This run should return an acceptable recall. We set
 * the bar very low (recall >= 50%) to make sure that a test fail means bad
 * implementation.
 * Second, a very cheap LSH search, with parameters that should cause recall
 * to be very low. Set the threshhold very high (recall <= 25%) to make sure
 * that a test fail means bad implementation.
 */
BOOST_AUTO_TEST_CASE(RecallTest)
{
  // kNN and LSH parameters (use LSH default parameters).
  const int k = 4;
  const int secondHashSize = 99901;
  const int bucketSize = 500;

  // Read iris training and testing data as reference and query sets.
  const string trainSet = "iris_train.csv";
  const string testSet = "iris_test.csv";
  arma::mat rdata;
  arma::mat qdata;
  data::Load(trainSet, rdata, true);
  data::Load(testSet, qdata, true);

  // Run classic knn on reference data.
  KNN knn(rdata);
  arma::Mat<size_t> groundTruth;
  arma::mat groundDistances;
  knn.Search(qdata, k, groundTruth, groundDistances);

  // Expensive LSH run.
  const int hExp = 10000; // First-level hash width.
  const int kExp = 1; // Projections per table.
  const int tExp = 128; // Number of tables.
  const double recallThreshExp = 0.5;

  LSHSearch<> lshTestExp(
      rdata,
      kExp,
      tExp,
      hExp,
      secondHashSize,
      bucketSize);
  arma::Mat<size_t> lshNeighborsExp;
  arma::mat lshDistancesExp;
  lshTestExp.Search(qdata, k, lshNeighborsExp, lshDistancesExp);

  const double recallExp = LSHSearch<>::ComputeRecall(lshNeighborsExp, groundTruth);

  // This run should have recall higher than the threshold.
  BOOST_REQUIRE_GE(recallExp, recallThreshExp);

  // Cheap LSH run.
  const int hChp = 1; // Small first-level hash width.
  const int kChp = 100; // Large number of projections per table.
  const int tChp = 1; // Only one table.
  const double recallThreshChp = 0.25; // Recall threshold.

  LSHSearch<> lshTestChp(
      rdata,
      kChp,
      tChp,
      hChp,
      secondHashSize,
      bucketSize);
  arma::Mat<size_t> lshNeighborsChp;
  arma::mat lshDistancesChp;
  lshTestChp.Search(qdata, k, lshNeighborsChp, lshDistancesChp);

  const double recallChp = LSHSearch<>::ComputeRecall(lshNeighborsChp,
      groundTruth);

  // This run should have recall lower than the threshold.
  BOOST_REQUIRE_LE(recallChp, recallThreshChp);
}

/**
 * Test: This is a deterministic test that projects 2-dpoints to a known line
 * (axis 2). The reference set contains 4 well-separated clusters that will
 * merge into 2 clusters when projected on that axis.
 *
 * We create two queries, each one belonging in one cluster (q1 in cluster 3
 * located around (0, 0) and q2 in cluster 2 located around (3, 3). After the
 * projection, q1 should have neighbors in C3 and C4 and q2 in C1 and C2.
 */
BOOST_AUTO_TEST_CASE(DeterministicMerge)
{
  const size_t N = 40; // Must be divisible by 4 to create 4 clusters properly.
  arma::mat rdata;
  arma::mat qdata;
  GetPointset(N, rdata);
  GetQueries(qdata);

  const int k = N / 2;
  const double hashWidth = 1;
  const int secondHashSize = 99901;
  const int bucketSize = 500;

  // 1 table, with one projection to axis 1.
  arma::cube projections(2, 1, 1);
  projections(0, 0, 0) = 0;
  projections(1, 0, 0) = 1;

  LSHSearch<> lshTest(rdata, projections, hashWidth, secondHashSize,
      bucketSize);

  arma::Mat<size_t> neighbors;
  arma::mat distances;
  lshTest.Search(qdata, k, neighbors, distances);

  // Test query 1.
  size_t q;
  for (size_t j = 0; j < k; ++j) // For each neighbor.
  {
    // If the neighbor is not found, ignore the point.
    if (neighbors(j, 0) == N || neighbors(j, 1) == N)
      continue;

    // Query 1 is in cluster 3, which under this projection was merged with
    // cluster 4. Clusters 3 and 4 have points 20:39, so only neighbors among
    //those should be found.
    q = 0;
    BOOST_REQUIRE_GE(neighbors(j, q), N / 2);

    // Query 2 is in cluster 2, which under this projection was merged with
    // cluster 1. Clusters 1 and 2 have points 0:19, so only neighbors among
    // those should be found.
    q = 1;
    BOOST_REQUIRE_LT(neighbors(j, q), N / 2);
  }
}

/**
 * Test: This is a deterministic test that projects 2-d points to the plane.
 * The reference set contains 4 well-separated clusters that should not merge.
 *
 * We create two queries, each one belonging in one cluster (q1 in cluster 3
 * located around (0, 0) and q2 in cluster 2 located around (3, 3). The test is
 * a success if, after the projection, q1 should have neighbors in c3 and q2
 * in c2.
 */
BOOST_AUTO_TEST_CASE(DeterministicNoMerge)
{
  const size_t N = 40;
  arma::mat rdata;
  arma::mat qdata;
  GetPointset(N, rdata);
  GetQueries(qdata);

  const int k = N / 2;
  const double hashWidth = 1;
  const int secondHashSize = 99901;
  const int bucketSize = 500;

  // 1 table, with one projection to axis 1.
  arma::cube projections(2, 2, 1);
  projections(0, 0, 0) = 0;
  projections(1, 0, 0) = 1;
  projections(0, 1, 0) = 1;
  projections(1, 1, 0) = 0;

  LSHSearch<> lshTest(rdata, projections, hashWidth, secondHashSize,
      bucketSize);

  arma::Mat<size_t> neighbors;
  arma::mat distances;
  lshTest.Search(qdata, k, neighbors, distances);

  // Test query 1.
  size_t q;
  for (size_t j = 0; j < k; ++j) // For each neighbor.
  {
    // If the neighbor is not found, ignore the point.
    if (neighbors(j, 0) == N || neighbors(j, 1) == N)
      continue;

    // Query 1 is in cluster 3, which is points 20:29.
    q = 0;
    BOOST_REQUIRE_LT(neighbors(j, q), 3 * N / 4);
    BOOST_REQUIRE_GE(neighbors(j, q), N / 2);

    // Query 2 is in cluster 2, which is points 10:19.
    q = 1;
    BOOST_REQUIRE_LT(neighbors(j, q), N / 2);
    BOOST_REQUIRE_GE(neighbors(j, q), N / 4);
  }
}

/**
 * Test: Create an LSHSearch object and use an increasing number of probes to
 * search for points. Require that recall for the same object doesn't decrease
 * with increasing number of probes. Also require that at least a few times
 * there's some increase in recall.
 */
BOOST_AUTO_TEST_CASE(MultiprobeTest)
{
  // Test parameters.
  const double epsilonIncrease = 0.05;
  const size_t repetitions = 5; // Train five objects.

  const size_t probeTrials = 5;
  const size_t numProbes[probeTrials] = {0, 1, 2, 3, 4};

  // Algorithm parameters.
  const int k = 4;
  const int numTables = 16;
  const int numProj = 3;
  const double hashWidth = 0;
  const int secondHashSize = 99901;
  const int bucketSize = 500;

  const string trainSet = "iris_train.csv";
  const string testSet = "iris_test.csv";
  arma::mat rdata;
  arma::mat qdata;
  data::Load(trainSet, rdata, true);
  data::Load(testSet, qdata, true);

  // Run classic knn on reference set.
  KNN knn(rdata);
  arma::Mat<size_t> groundTruth;
  arma::mat groundDistances;
  knn.Search(qdata, k, groundTruth, groundDistances);

  bool foundIncrease = 0;

  for (size_t rep = 0; rep < repetitions; ++rep)
  {
    // Train a model.
    LSHSearch<> multiprobeTest(rdata, numProj, numTables, hashWidth,
        secondHashSize, bucketSize);

    double prevRecall = 0;
    // Search with varying number of probes.
    for (size_t p = 0; p < probeTrials; ++p)
    {
      arma::Mat<size_t> lshNeighbors;
      arma::mat lshDistances;

      multiprobeTest.Search(qdata, k, lshNeighbors, lshDistances, 0,
          numProbes[p]);

      // Compute recall of this run.
      double recall = LSHSearch<>::ComputeRecall(lshNeighbors, groundTruth);
      if (p > 0)
      {
        // More probes should at the very least not lower recall...
        BOOST_REQUIRE_GE(recall, prevRecall);

        // ... and should ideally increase it a bit.
        if (recall > prevRecall + epsilonIncrease)
          foundIncrease = true;
        prevRecall = recall;
      }
    }
  }
  BOOST_REQUIRE(foundIncrease);
}

/**
 * Test: This is a deterministic test that verifies multiprobe LSH works
 * correctly. To do this, we generate two queries, q1 and q2. q1 is hashed
 * directly under cluster C2, q2 is hashed in C2's center.
 * We verify that:
 * 1) q1 should have no neighbors without multiprobe.
 * 2) q1 should have neighbors only from C2 with 1 additional probe.
 * 3) q2 should have all neighbors found with 3 additional probes.
 */
BOOST_AUTO_TEST_CASE(MultiprobeDeterministicTest)
{
  // Generate known deterministic clusters of points.
  const size_t N = 40;
  arma::mat rdata;
  GetPointset(N, rdata);

  const int k = N / 4;
  const double hashWidth = 1;
  const int secondHashSize = 99901;
  const int bucketSize = 500;

  // 1 table, projections on orthonormal plane.
  arma::cube projections(2, 2, 1);
  projections(0, 0, 0) = 1;
  projections(1, 0, 0) = 0;
  projections(0, 1, 0) = 0;
  projections(1, 1, 0) = 1;

  // Construct LSH object with given tables.
  LSHSearch<> lshTest(rdata, projections,
                      hashWidth, secondHashSize, bucketSize);

  const arma::mat offsets = lshTest.Offsets();

  // Construct q1 so it is hashed directly under C2.
  arma::mat q1;
  q1 << 3.9 << arma::endr << 2.99;
  q1 -= offsets;

  // Construct q2 so it is hashed near the center of C2.
  arma::mat q2;
  q2 << 3.6 << arma::endr << 3.6;
  q2 -= offsets;

  arma::Mat<size_t> neighbors;
  arma::mat distances;

  // Test that q1 simple search comes up empty.
  lshTest.Search(q1, k, neighbors, distances);
  BOOST_REQUIRE(arma::all(neighbors.col(0) == N));

  // Test that q1 search with 1 additional probe returns some C2 points.
  lshTest.Search(q1, k, neighbors, distances, 0, 1);
  BOOST_REQUIRE(arma::all(
        (neighbors.col(0) == N) ||
        ((neighbors.col(0) >= N / 4) && (neighbors.col(0) < N / 2))));

  // Test that q2 simple search returns some C2 points.
  lshTest.Search(q2, k, neighbors, distances);
  BOOST_REQUIRE(arma::all(
      (neighbors.col(0) == N) ||
      ((neighbors.col(0) >= N / 4) && (neighbors.col(0) < N / 2))));

  // Test that q2 with 3 additional probes returns all C2 points.
  lshTest.Search(q2, k, neighbors, distances, 0, 3);
  BOOST_REQUIRE(arma::all(
      (neighbors.col(0) >= N / 4) && (neighbors.col(0) < N / 2)));
}

BOOST_AUTO_TEST_CASE(LSHTrainTest)
{
  // This is a not very good test that simply checks that the re-trained LSH
  // model operates on the correct dimensionality and returns the correct number
  // of results.
  arma::mat referenceData = arma::randu<arma::mat>(3, 100);
  arma::mat newReferenceData = arma::randu<arma::mat>(10, 400);
  arma::mat queryData = arma::randu<arma::mat>(10, 200);

  LSHSearch<> lsh(referenceData, 3, 2, 2.0, 11, 3);

  lsh.Train(newReferenceData, 4, 3, 3.0, 12, 4);

  arma::Mat<size_t> neighbors;
  arma::mat distances;

  lsh.Search(queryData, 3, neighbors, distances);

  BOOST_REQUIRE_EQUAL(neighbors.n_cols, 200);
  BOOST_REQUIRE_EQUAL(neighbors.n_rows, 3);
  BOOST_REQUIRE_EQUAL(distances.n_cols, 200);
  BOOST_REQUIRE_EQUAL(distances.n_rows, 3);
}

/**
 * Test: this verifies ComputeRecall works correctly by providing two identical
 * vectors and requiring that Recall is equal to 1.
 */
BOOST_AUTO_TEST_CASE(RecallTestIdentical)
{
  const size_t k = 5; // 5 nearest neighbors
  const size_t numQueries = 1;

  // base = [1; 2; 3; 4; 5]
  arma::Mat<size_t> base;
  base.set_size(k, numQueries);
  base.col(0) = arma::linspace< arma::Col<size_t> >(1, k, k);

  // q1 = [1; 2; 3; 4; 5]. Expect recall = 1
  arma::Mat<size_t> q1;
  q1.set_size(k, numQueries);
  q1.col(0) = arma::linspace< arma::Col<size_t> >(1, k, k);

  BOOST_REQUIRE_EQUAL(LSHSearch<>::ComputeRecall(base, q1), 1);
}

/**
 * Test: this verifies ComputeRecall returns correct values for partially
 * correct found neighbors. This is important because this is a good example of
 * how the recall and accuracy metrics differ - accuracy in this case would be
 * 0, recall should not be
 */
BOOST_AUTO_TEST_CASE(RecallTestPartiallyCorrect)
{
  const size_t k = 5; // 5 nearest neighbors
  const size_t numQueries = 1;

  // base = [1; 2; 3; 4; 5]
  arma::Mat<size_t> base;
  base.set_size(k, numQueries);
  base.col(0) = arma::linspace< arma::Col<size_t> >(1, k, k);

  // q2 = [2; 3; 4; 6; 7]. Expect recall = 0.6. This is important because this
  // is a good example of how recall and accuracy differ. Accuracy here would
  // be 0 but recall should not be.
  arma::Mat<size_t> q2;
  q2.set_size(k, numQueries);
  q2 <<
    2 << arma::endr <<
    3 << arma::endr <<
    4 << arma::endr <<
    6 << arma::endr <<
    7 << arma::endr;

  BOOST_REQUIRE_CLOSE(LSHSearch<>::ComputeRecall(base, q2), 0.6, 0.0001);
}

/**
 * Test: If given a completely wrong vector, ComputeRecall should return 0
 */
BOOST_AUTO_TEST_CASE(RecallTestIncorrect)
{
  const size_t k = 5; // 5 nearest neighbors
  const size_t numQueries = 1;

  // base = [1; 2; 3; 4; 5]
  arma::Mat<size_t> base;
  base.set_size(k, numQueries);
  base.col(0) = arma::linspace< arma::Col<size_t> >(1, k, k);
  // q3 = [6; 7; 8; 9; 10]. Expected recall = 0
  arma::Mat<size_t> q3;
  q3.set_size(k, numQueries);
  q3.col(0) = arma::linspace< arma::Col<size_t> >(k + 1, 2 * k, k);

  BOOST_REQUIRE_EQUAL(LSHSearch<>::ComputeRecall(base, q3), 0);
}

/**
 * Test: If given a vector of wrong shape, ComputeRecall should throw an
 * exception.
 */
BOOST_AUTO_TEST_CASE(RecallTestException)
{
  const size_t k = 5; // 5 nearest neighbors
  const size_t numQueries = 1;

  // base = [1; 2; 3; 4; 5]
  arma::Mat<size_t> base;
  base.set_size(k, numQueries);
  base.col(0) = arma::linspace< arma::Col<size_t> >(1, k, k);
  // verify that nonsense arguments throw exception
  arma::Mat<size_t> q4;
  q4.set_size(2 * k, numQueries);

  BOOST_REQUIRE_THROW(LSHSearch<>::ComputeRecall(base, q4),
      std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(EmptyConstructorTest)
{
  // If we create an empty LSH model and then call Search(), it should throw an
  // exception.
  LSHSearch<> lsh;

  arma::mat dataset = arma::randu<arma::mat>(5, 50);
  arma::mat distances;
  arma::Mat<size_t> neighbors;
  BOOST_REQUIRE_THROW(lsh.Search(dataset, 2, neighbors, distances),
      std::invalid_argument);

  // Now, train.
  lsh.Train(dataset, 4, 3, 3.0, 12, 4);

  lsh.Search(dataset, 3, neighbors, distances);

  BOOST_REQUIRE_EQUAL(neighbors.n_cols, 50);
  BOOST_REQUIRE_EQUAL(neighbors.n_rows, 3);
  BOOST_REQUIRE_EQUAL(distances.n_cols, 50);
  BOOST_REQUIRE_EQUAL(distances.n_rows, 3);
}

// These two tests are only compiled if the user has specified OpenMP to be
// used.
#ifdef HAS_OPENMP
/**
 * Test: This test verifies that parallel query processing returns correct
 * results for the bichromatic search.
 */
BOOST_AUTO_TEST_CASE(ParallelBichromatic)
{
  // kNN and LSH parameters (use LSH default parameters).
  const int k = 4;
  const int numTables = 16;
  const int numProj = 3;

  // Read iris training and testing data as reference and query sets.
  const string trainSet = "iris_train.csv";
  const string testSet = "iris_test.csv";
  arma::mat rdata;
  arma::mat qdata;
  data::Load(trainSet, rdata, true);
  data::Load(testSet, qdata, true);

  // Where to store neighbors and distances
  arma::Mat<size_t> sequentialNeighbors;
  arma::Mat<size_t> parallelNeighbors;
  arma::mat distances;

  // Construct an LSH object. By default, it uses the maximum number of threads
  LSHSearch<> lshTest(rdata, numProj, numTables); //default parameters
  lshTest.Search(qdata, k, parallelNeighbors, distances);

  // Now perform same search but with 1 thread
  size_t prevNumThreads = omp_get_max_threads(); // Store number of threads used.
  omp_set_num_threads(1);
  lshTest.Search(qdata, k, sequentialNeighbors, distances);
  omp_set_num_threads(prevNumThreads);

  // Require both have same results
  double recall = LSHSearch<>::ComputeRecall(sequentialNeighbors, parallelNeighbors);
  BOOST_REQUIRE_EQUAL(recall, 1);
}

/**
 * Test: This test verifies that parallel query processing returns correct
 * results for the monochromatic search.
 */
BOOST_AUTO_TEST_CASE(ParallelMonochromatic)
{
  // kNN and LSH parameters.
  const int k = 4;
  const int numTables = 16;
  const int numProj = 3;

  // Read iris training data as reference and query set.
  const string trainSet = "iris_train.csv";
  arma::mat rdata;
  data::Load(trainSet, rdata, true);

  // Where to store neighbors and distances
  arma::Mat<size_t> sequentialNeighbors;
  arma::Mat<size_t> parallelNeighbors;
  arma::mat distances;

  // Construct an LSH object, using maximum number of available threads.
  LSHSearch<> lshTest(rdata, numProj, numTables);
  lshTest.Search(k, parallelNeighbors, distances);

  // Now perform same search but with 1 thread.
  size_t prevNumThreads = omp_get_max_threads(); // Store number of threads used.
  omp_set_num_threads(1);
  lshTest.Search(k, sequentialNeighbors, distances);
  omp_set_num_threads(prevNumThreads);

  // Require both have same results.
  double recall = LSHSearch<>::ComputeRecall(sequentialNeighbors, parallelNeighbors);
  BOOST_REQUIRE_EQUAL(recall, 1);
}
#endif

BOOST_AUTO_TEST_SUITE_END();
