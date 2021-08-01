
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_lsh_test.cpp:

Program Listing for File lsh_test.cpp
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_lsh_test.cpp>` (``/home/aakash/mlpack/src/mlpack/tests/lsh_test.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include <mlpack/core.hpp>
   #include <mlpack/core/metrics/lmetric.hpp>
   #include "catch.hpp"
   #include "test_catch_tools.hpp"
   
   #include <mlpack/methods/lsh/lsh_search.hpp>
   #include <mlpack/methods/neighbor_search/neighbor_search.hpp>
   
   using namespace std;
   using namespace mlpack;
   using namespace mlpack::neighbor;
   
   void GetPointset(const size_t N, arma::mat& rdata)
   {
     const size_t d = 2;
     // Create four clusters of points.
     arma::mat c1(d, N / 4, arma::fill::randu);
     arma::mat c2(d, N / 4, arma::fill::randu);
     arma::mat c3(d, N / 4, arma::fill::randu);
     arma::mat c4(d, N / 4, arma::fill::randu);
   
     arma::colvec offset1;
     offset1 = { { 0 },
                 { 3 } };
   
     arma::colvec offset2;
     offset2 = { { 3 },
                 { 3 } };
   
     arma::colvec offset4;
     offset4 = { { 3 },
                 { 0 } };
   
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
   
   TEST_CASE("NumTablesTest", "[LSHTest]")
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
     if (!data::Load(trainSet, rdata))
       FAIL("Cannot load dataset");
     if (!data::Load(testSet, qdata))
       FAIL("Cannot load dataset");
   
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
       const int lValue[] = { 1, 8, 16, 32, 64, 128 }; // Number of tables.
       double lValueRecall[lSize] = { 0.0 }; // Recall of each LSH run.
   
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
   
     REQUIRE(fail == false);
   }
   
   TEST_CASE("HashWidthTest", "[LSHTest]")
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
     if (!data::Load(trainSet, rdata))
       FAIL("Cannot load dataset");
     if (!data::Load(testSet, qdata))
       FAIL("Cannot load dataset");
   
     // Run classic knn on reference data.
     KNN knn(rdata);
     arma::Mat<size_t> groundTruth;
     arma::mat groundDistances;
     knn.Search(qdata, k, groundTruth, groundDistances);
     const int hSize = 7; // Number of runs.
     const double hValue[] = { 0.1, 0.5, 1, 5, 10, 50, 500 }; // Hash width.
     double hValueRecall[hSize] = { 0.0 }; // Recall of each run.
   
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
         REQUIRE(hValueRecall[h] >= hValueRecall[h - 1] - epsilon);
     }
   }
   
   TEST_CASE("NumProjTest", "[LSHTest]")
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
     if (!data::Load(trainSet, rdata))
       FAIL("Cannot load dataset");
     if (!data::Load(testSet, qdata))
       FAIL("Cannot load dataset");
   
     // Run classic knn on reference data.
     KNN knn(rdata);
     arma::Mat<size_t> groundTruth;
     arma::mat groundDistances;
     knn.Search(qdata, k, groundTruth, groundDistances);
   
     // LSH test parameters for numProj.
     const int pSize = 5; // Number of runs.
     const int pValue[] = { 1, 10, 20, 50, 100 }; // Number of projections.
     double pValueRecall[pSize] = { 0.0 }; // Recall of each run.
   
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
         REQUIRE(pValueRecall[p] - epsilon < pValueRecall[p - 1]);
     }
   }
   
   TEST_CASE("RecallTest", "[LSHTest]")
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
     if (!data::Load(trainSet, rdata))
       FAIL("Cannot load dataset");
     if (!data::Load(testSet, qdata))
       FAIL("Cannot load dataset");
   
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
   
     const double recallExp = LSHSearch<>::ComputeRecall(
         lshNeighborsExp, groundTruth);
   
     // This run should have recall higher than the threshold.
     REQUIRE(recallExp >= recallThreshExp);
   
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
     REQUIRE(recallChp <= recallThreshChp);
   }
   
   TEST_CASE("DeterministicMerge", "[LSHTest]")
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
       // those should be found.
       q = 0;
       REQUIRE(neighbors(j, q) >= N / 2);
   
       // Query 2 is in cluster 2, which under this projection was merged with
       // cluster 1. Clusters 1 and 2 have points 0:19, so only neighbors among
       // those should be found.
       q = 1;
       REQUIRE(neighbors(j, q) < N / 2);
     }
   }
   
   TEST_CASE("DeterministicNoMerge", "[LSHTest]")
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
       REQUIRE(neighbors(j, q) < 3 * N / 4);
       REQUIRE(neighbors(j, q) >= N / 2);
   
       // Query 2 is in cluster 2, which is points 10:19.
       q = 1;
       REQUIRE(neighbors(j, q) < N / 2);
       REQUIRE(neighbors(j, q) >= N / 4);
     }
   }
   
   TEST_CASE("MultiprobeTest", "[LSHTest]")
   {
     // Test parameters.
     const double epsilonIncrease = 0.01;
     const size_t repetitions = 5; // Train five objects.
   
     const size_t probeTrials = 5;
     const size_t numProbes[probeTrials] = { 0, 1, 2, 3, 4 };
   
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
     if (!data::Load(trainSet, rdata))
       FAIL("Cannot load dataset");
     if (!data::Load(testSet, qdata))
       FAIL("Cannot load dataset");
   
     // Add a slight amount of noise to the dataset, so that we don't end up with
     // points that have the same distance (hopefully).
     rdata += 0.0001 * arma::randn<arma::mat>(rdata.n_rows, rdata.n_cols);
     qdata += 0.0001 * arma::randn<arma::mat>(qdata.n_rows, qdata.n_cols);
   
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
           REQUIRE(recall >= prevRecall);
   
           // ... and should ideally increase it a bit.
           if (recall > prevRecall + epsilonIncrease)
             foundIncrease = true;
           prevRecall = recall;
         }
       }
     }
     REQUIRE(foundIncrease);
   }
   
   TEST_CASE("MultiprobeDeterministicTest", "[LSHTest]")
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
     q1 = arma::mat({ 3.9, 2.99 }).t();
     q1 -= offsets;
   
     // Construct q2 so it is hashed near the center of C2.
     arma::mat q2;
     q2 = arma::mat({ 3.6, 3.6 }).t();
     q2 -= offsets;
   
     arma::Mat<size_t> neighbors;
     arma::mat distances;
   
     // Test that q1 simple search comes up empty.
     lshTest.Search(q1, k, neighbors, distances);
     REQUIRE(arma::all(neighbors.col(0) == N));
   
     // Test that q1 search with 1 additional probe returns some C2 points.
     lshTest.Search(q1, k, neighbors, distances, 0, 1);
     REQUIRE(arma::all(
           (neighbors.col(0) == N) ||
           ((neighbors.col(0) >= N / 4) && (neighbors.col(0) < N / 2))));
   
     // Test that q2 simple search returns some C2 points.
     lshTest.Search(q2, k, neighbors, distances);
     REQUIRE(arma::all(
         (neighbors.col(0) == N) ||
         ((neighbors.col(0) >= N / 4) && (neighbors.col(0) < N / 2))));
   
     // Test that q2 with 3 additional probes returns all C2 points.
     lshTest.Search(q2, k, neighbors, distances, 0, 3);
     REQUIRE(arma::all(
         (neighbors.col(0) >= N / 4) && (neighbors.col(0) < N / 2)));
   }
   
   TEST_CASE("LSHTrainTest", "[LSHTest]")
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
   
     REQUIRE(neighbors.n_cols == 200);
     REQUIRE(neighbors.n_rows == 3);
     REQUIRE(distances.n_cols == 200);
     REQUIRE(distances.n_rows == 3);
   }
   
   TEST_CASE("RecallTestIdentical", "[LSHTest]")
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
   
     REQUIRE(LSHSearch<>::ComputeRecall(base, q1) == 1);
   }
   
   TEST_CASE("RecallTestPartiallyCorrect", "[LSHTest]")
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
     q2 = arma::Mat<size_t>({ 2, 3, 4, 6, 7 }).t();
   
     REQUIRE(LSHSearch<>::ComputeRecall(base, q2) == Approx(0.6).epsilon(1e-6));
   }
   
   TEST_CASE("RecallTestIncorrect", "[LSHTest]")
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
   
     REQUIRE(LSHSearch<>::ComputeRecall(base, q3) == 0);
   }
   
   TEST_CASE("RecallTestException", "[LSHTest]")
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
   
     REQUIRE_THROWS_AS(LSHSearch<>::ComputeRecall(base, q4),
         std::invalid_argument);
   }
   
   TEST_CASE("LSHTestEmptyConstructorTest", "[LSHTest]")
   {
     // If we create an empty LSH model and then call Search(), it should throw an
     // exception.
     LSHSearch<> lsh;
   
     arma::mat dataset = arma::randu<arma::mat>(5, 50);
     arma::mat distances;
     arma::Mat<size_t> neighbors;
     REQUIRE_THROWS_AS(lsh.Search(dataset, 2, neighbors, distances),
         std::invalid_argument);
   
     // Now, train.
     lsh.Train(dataset, 4, 3, 3.0, 12, 4);
   
     lsh.Search(dataset, 3, neighbors, distances);
   
     REQUIRE(neighbors.n_cols == 50);
     REQUIRE(neighbors.n_rows == 3);
     REQUIRE(distances.n_cols == 50);
     REQUIRE(distances.n_rows == 3);
   }
   
   // These two tests are only compiled if the user has specified OpenMP to be
   // used.
   #ifdef HAS_OPENMP
   
   TEST_CASE("ParallelBichromatic", "[LSHTest]")
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
     if (!data::Load(trainSet, rdata))
       FAIL("Cannot load dataset");
     if (!data::Load(testSet, qdata))
       FAIL("Cannot load dataset");
   
     // Where to store neighbors and distances
     arma::Mat<size_t> sequentialNeighbors;
     arma::Mat<size_t> parallelNeighbors;
     arma::mat distances;
   
     // Construct an LSH object. By default, it uses the maximum number of threads
     LSHSearch<> lshTest(rdata, numProj, numTables); // Default parameters.
     lshTest.Search(qdata, k, parallelNeighbors, distances);
   
     // Now perform same search but with 1 thread
     // Store number of threads used.
     size_t prevNumThreads = omp_get_max_threads();
     omp_set_num_threads(1);
     lshTest.Search(qdata, k, sequentialNeighbors, distances);
     omp_set_num_threads(prevNumThreads);
   
     // Require both have same results
     double recall = LSHSearch<>::ComputeRecall(
         sequentialNeighbors, parallelNeighbors);
     REQUIRE(recall == 1);
   }
   
   TEST_CASE("ParallelMonochromatic", "[LSHTest]")
   {
     // kNN and LSH parameters.
     const int k = 4;
     const int numTables = 16;
     const int numProj = 3;
   
     // Read iris training data as reference and query set.
     const string trainSet = "iris_train.csv";
     arma::mat rdata;
     if (!data::Load(trainSet, rdata))
       FAIL("Cannot load dataset");
   
     // Where to store neighbors and distances
     arma::Mat<size_t> sequentialNeighbors;
     arma::Mat<size_t> parallelNeighbors;
     arma::mat distances;
   
     // Construct an LSH object, using maximum number of available threads.
     LSHSearch<> lshTest(rdata, numProj, numTables);
     lshTest.Search(k, parallelNeighbors, distances);
   
     // Now perform same search but with 1 thread.
     // Store number of threads used.
     size_t prevNumThreads = omp_get_max_threads();
     omp_set_num_threads(1);
     lshTest.Search(k, sequentialNeighbors, distances);
     omp_set_num_threads(prevNumThreads);
   
     // Require both have same results.
     double recall = LSHSearch<>::ComputeRecall(
         sequentialNeighbors, parallelNeighbors);
     REQUIRE(recall == 1);
   }
   #endif
   
   // Test the copy constructor and the copy operator.
   TEST_CASE("LSHTestCopyConstructorAndOperatorTest", "[LSHTest]")
   {
     arma::mat dataset = arma::randu<arma::mat>(10, 1000);
   
     // Use default parameters.
     LSHSearch<> lsh(dataset, 10, 10);
   
     // Copy the model.
     LSHSearch<> lsh2(lsh);
     LSHSearch<> lsh3 = lsh;
   
     arma::Mat<size_t> neighbors, neighbors2, neighbors3;
     arma::mat distances, distances2, distances3;
   
     lsh.Search(5, neighbors, distances);
     lsh2.Search(5, neighbors2, distances2);
     lsh3.Search(5, neighbors3, distances3);
   
     CheckMatrices(neighbors, neighbors2);
     CheckMatrices(neighbors, neighbors3);
     CheckMatrices(distances, distances2);
     CheckMatrices(distances, distances3);
   }
   
   // Test the move constructor.
   TEST_CASE("LSHTestMoveConstructorTest", "[LSHTest]")
   {
     arma::mat dataset = arma::randu<arma::mat>(10, 1000);
   
     // Use default parameters.
     LSHSearch<>* lsh = new LSHSearch<>(dataset, 10, 10);
   
     // Get results.
     arma::Mat<size_t> neighbors, neighbors2;
     arma::mat distances, distances2;
   
     lsh->Search(5, neighbors, distances);
   
     LSHSearch<> lsh2(std::move(*lsh));
     delete lsh;
   
     lsh2.Search(5, neighbors2, distances2);
   
     CheckMatrices(neighbors, neighbors2);
     CheckMatrices(distances, distances2);
   }
   
   // Test the move operator.
   TEST_CASE("LSHTestMoveOperatorTest", "[LSHTest]")
   {
     arma::mat dataset = arma::randu<arma::mat>(10, 1000);
   
     // Use default parameters.
     LSHSearch<>* lsh = new LSHSearch<>(dataset, 10, 10);
   
     // Get results.
     arma::Mat<size_t> neighbors, neighbors2;
     arma::mat distances, distances2;
   
     lsh->Search(5, neighbors, distances);
   
     LSHSearch<> lsh2 = std::move(*lsh);
     delete lsh;
   
     lsh2.Search(5, neighbors2, distances2);
   
     CheckMatrices(neighbors, neighbors2);
     CheckMatrices(distances, distances2);
   }
   
   TEST_CASE("SparseLSHTest", "[LSHTest]")
   {
     // kNN and LSH parameters (use LSH default parameters).
     const int k = 5;
     const int numTables = 5;
     const int numProj = 2;
     const double hashWidth = 50.0;
     const int secondHashSize = 99901;
     const int bucketSize = 500;
   
     // Read iris training and testing data as reference and query sets.
     const string trainSet = "iris_train.csv";
     const string testSet = "iris_test.csv";
     arma::mat rdata;
     arma::mat qdata;
     if (!data::Load(trainSet, rdata))
       FAIL("Cannot load dataset");
     if (!data::Load(testSet, qdata))
       FAIL("Cannot load dataset");
   
     // Run on dense data.
     LSHSearch<> denseLSH(
         rdata,
         numProj,
         numTables,
         hashWidth,
         secondHashSize,
         bucketSize);
   
     arma::Mat<size_t> denseNeighbors;
     arma::mat denseDistances;
     denseLSH.Search(qdata, k, denseNeighbors, denseDistances);
   
     // Now create and run on sparse data.
     arma::sp_mat sparseRData(rdata);
     arma::sp_mat sparseQData(qdata);
   
     LSHSearch<NearestNeighborSort, arma::sp_mat> sparseLSH(
         sparseRData,
         denseLSH.Projections(),
         hashWidth,
         secondHashSize,
         bucketSize);
   
     arma::Mat<size_t> sparseNeighbors;
     arma::mat sparseDistances;
     sparseLSH.Search(sparseQData, k, sparseNeighbors, sparseDistances);
   
     REQUIRE(denseNeighbors.n_rows == sparseNeighbors.n_rows);
     REQUIRE(denseNeighbors.n_cols == sparseNeighbors.n_cols);
     REQUIRE(denseDistances.n_rows == sparseDistances.n_rows);
     REQUIRE(denseDistances.n_cols == sparseDistances.n_cols);
   
     // Make sure that sparse LSH distances aren't garbage.
     for (size_t i = 0; i < sparseNeighbors.n_elem; ++i)
     {
       REQUIRE(sparseNeighbors[i] < rdata.n_cols);
       REQUIRE(sparseDistances[i] >= 0.0);
       REQUIRE(!std::isinf(sparseDistances[i]));
       REQUIRE(!std::isnan(sparseDistances[i]));
     }
   }
