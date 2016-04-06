/**
 * @file lsh_test.cpp
 *
 * Unit tests for the 'LSHSearch' class.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

#include <mlpack/methods/lsh/lsh_search.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>

using namespace std;
using namespace mlpack;
using namespace mlpack::neighbor;

double compute_recall(
    const arma::Mat<size_t>& LSHneighbors, 
    const arma::Mat<size_t>& groundTruth)
{
  const size_t queries = LSHneighbors.n_cols;
  const size_t neigh = LSHneighbors.n_rows;

  int same = 0;
  for (size_t q = 0; q < queries; ++q)
  {
    for (size_t n = 0; n < neigh; ++n)
    {
      same+=(LSHneighbors(n,q)==groundTruth(n,q));
    }
  }
  return static_cast<double>(same)/
    (static_cast<double>(queries*neigh));
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
BOOST_AUTO_TEST_CASE(numTablesTest)
{

  //math::RandomSeed(time(0));
  //kNN and LSH parameters (use LSH default parameters)
  const int k = 4;
  const int numProj = 10;
  const double hashWidth = 0;
  const int secondHashSize = 99901;
  const int bucketSize = 500;
  
  //test parameters
  const double epsilon = 0.1; //allowed deviation from expected monotonicity
  const int numTries = 5; //tries for each test before declaring failure

  //read iris training and testing data as reference and query
  const string trainSet="iris_train.csv";
  const string testSet="iris_test.csv";
  arma::mat rdata;
  arma::mat qdata;
  data::Load(trainSet, rdata, true);
  data::Load(testSet, qdata, true);

  //Run classic knn on reference data
  AllkNN knn(rdata);
  arma::Mat<size_t> groundTruth;
  arma::mat groundDistances;
  knn.Search(qdata, k, groundTruth, groundDistances);

  
  bool fail;
  for (int t = 0; t < numTries; ++t)
  {

    fail = false;
    const int lSize = 6; //number of runs
    const int lValue[] = {1, 8, 16, 32, 64, 128}; //number of tables
    double lValueRecall[lSize] = {0.0}; //recall of each LSH run

    for (size_t l=0; l < lSize; ++l)
    {
      //run LSH with only numTables varying (other values default)
      LSHSearch<> lshTest(rdata, numProj, lValue[l], 
          hashWidth, secondHashSize, bucketSize);
      arma::Mat<size_t> LSHneighbors;
      arma::mat LSHdistances;
      lshTest.Search(qdata, k, LSHneighbors, LSHdistances);

      //compute recall for each query
      lValueRecall[l] = compute_recall(LSHneighbors, groundTruth);

      if (l > 0)
      {
        if(lValueRecall[l] < lValueRecall[l-1]-epsilon)
        {
          fail = true; //if test fails at one point, stop and retry
          break;
        }
      }
    }

    if ( !fail )
      break; //if test passes one time, it is sufficient
  }
  BOOST_REQUIRE(fail == false);
}

/*Test: Run LSH with varying hash width, keeping all other parameters 
 * constant. Compute the recall, i.e. the number of reported neighbors that
 * are real neighbors of the query.
 * LSH's property is that (with high probability), increasing the hash width
 * will increase recall. Epsilon ensures that if noise lightly affects the 
 * projections, the test will not fail.
 */
BOOST_AUTO_TEST_CASE(hashWidthTest)
{

  //math::RandomSeed(time(0));
  //kNN and LSH parameters (use LSH default parameters)
  const int k = 4;
  const int numTables = 30;
  const int numProj = 10;
  const int secondHashSize = 99901;
  const int bucketSize = 500;
  
  //test parameters
  const double epsilon = 0.1; //allowed deviation from expected monotonicity

  //read iris training and testing data as reference and query
  const string trainSet="iris_train.csv";
  const string testSet="iris_test.csv";
  arma::mat rdata;
  arma::mat qdata;
  data::Load(trainSet, rdata, true);
  data::Load(testSet, qdata, true);

  //Run classic knn on reference data
  AllkNN knn(rdata);
  arma::Mat<size_t> groundTruth;
  arma::mat groundDistances;
  knn.Search(qdata, k, groundTruth, groundDistances);
  const int hSize = 7; //number of runs
  const double hValue[] = {0.1, 0.5, 1, 5, 10, 50, 500}; //hash width
  double hValueRecall[hSize] = {0.0}; //recall of each run

  for (size_t h=0; h < hSize; ++h)
  {
    //run LSH with only hashWidth varying (other values default)
    LSHSearch<> lshTest(
        rdata, 
        numProj, 
        numTables, 
        hValue[h], 
        secondHashSize, 
        bucketSize);
    
    arma::Mat<size_t> LSHneighbors;
    arma::mat LSHdistances;
    lshTest.Search(qdata, k, LSHneighbors, LSHdistances);

    //compute recall for each query
    hValueRecall[h] = compute_recall(LSHneighbors, groundTruth);

    if (h > 0)
        BOOST_REQUIRE_GE(hValueRecall[h], hValueRecall[h-1]-epsilon);
    
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
BOOST_AUTO_TEST_CASE(numProjTest)
{

  //math::RandomSeed(time(0));
  //kNN and LSH parameters (use LSH default parameters)
  const int k = 4;
  const int numTables = 30;
  const double hashWidth = 0;
  const int secondHashSize = 99901;
  const int bucketSize = 500;
  
  //test parameters
  const double epsilon = 0.1; //allowed deviation from expected monotonicity

  //read iris training and testing data as reference and query
  const string trainSet="iris_train.csv";
  const string testSet="iris_test.csv";
  arma::mat rdata;
  arma::mat qdata;
  data::Load(trainSet, rdata, true);
  data::Load(testSet, qdata, true);

  //Run classic knn on reference data
  AllkNN knn(rdata);
  arma::Mat<size_t> groundTruth;
  arma::mat groundDistances;
  knn.Search(qdata, k, groundTruth, groundDistances);

  //LSH test parameters for numProj
  const int pSize = 5; //number of runs
  const int pValue[] = {1, 10, 20, 50, 100}; //number of projections
  double pValueRecall[pSize] = {0.0}; //recall of each run

  for (size_t p=0; p < pSize; ++p)
  {
    //run LSH with only numProj varying (other values default)
    LSHSearch<> lshTest(
        rdata, 
        pValue[p], 
        numTables, 
        hashWidth, 
        secondHashSize, 
        bucketSize);

    arma::Mat<size_t> LSHneighbors;
    arma::mat LSHdistances;
    lshTest.Search(qdata, k, LSHneighbors, LSHdistances);

    //compute recall for each query
    pValueRecall[p] = compute_recall(LSHneighbors, groundTruth);

    if (p > 0) //don't check first run, only that increasing P decreases recall
        BOOST_REQUIRE_LE(pValueRecall[p] - epsilon, pValueRecall[p-1]);
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
BOOST_AUTO_TEST_CASE(recallTest)
{
  //math::RandomSeed(time(0));
  //kNN and LSH parameters (use LSH default parameters)
  const int k = 4;
  const int secondHashSize = 99901;
  const int bucketSize = 500;
  

  //read iris training and testing data as reference and query
  const string trainSet="iris_train.csv";
  const string testSet="iris_test.csv";
  arma::mat rdata;
  arma::mat qdata;
  data::Load(trainSet, rdata, true);
  data::Load(testSet, qdata, true);

  //Run classic knn on reference data
  AllkNN knn(rdata);
  arma::Mat<size_t> groundTruth;
  arma::mat groundDistances;
  knn.Search(qdata, k, groundTruth, groundDistances);
 
  //Expensive LSH run
  const int hExp = 10000; //first-level hash width
  const int kExp = 1; //projections per table
  const int tExp = 128; //number of tables
  const double recallThreshExp = 0.5;

  LSHSearch<> lshTestExp(
      rdata, 
      kExp, 
      tExp, 
      hExp, 
      secondHashSize, 
      bucketSize);
  arma::Mat<size_t> LSHneighborsExp;
  arma::mat LSHdistancesExp;
  lshTestExp.Search(qdata, k, LSHneighborsExp, LSHdistancesExp);
  
  const double recallExp = compute_recall(LSHneighborsExp, groundTruth);

  //This run should have recall higher than the threshold
  BOOST_REQUIRE_GE(recallExp, recallThreshExp);

  //Cheap LSH Run
  const int hChp = 1; //small first-level hash width
  const int kChp = 1000; //large number of projections per table
  const int tChp = 1; //only one table
  const double recallThreshChp = 0.25; //recall threshold

  LSHSearch<> lshTestChp(
      rdata, 
      kChp, 
      tChp, 
      hChp, 
      secondHashSize, 
      bucketSize);
  arma::Mat<size_t> LSHneighborsChp;
  arma::mat LSHdistancesChp;
  lshTestChp.Search(qdata, k, LSHneighborsChp, LSHdistancesChp);

  const double recallChp = compute_recall(LSHneighborsChp, groundTruth);

  //This run should have recall lower than the threshold
  BOOST_REQUIRE_LE(recallChp, recallThreshChp);
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

BOOST_AUTO_TEST_SUITE_END();
