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

double compute_recall(arma::Mat<size_t> LSHneighbors, 
        arma::Mat<size_t> groundTruth)
{
  const int n_queries = LSHneighbors.n_cols;
  const int n_neigh = LSHneighbors.n_rows;

  int found_same = 0;
  for (int q = 0; q < n_queries; ++q)
  {
    for (int n = 0; n < n_neigh; ++n)
    {
      found_same+=(LSHneighbors(n,q)==groundTruth(n,q));
    }
  }
  return static_cast<double>(found_same)/
      (static_cast<double>(n_queries*n_neigh));
}

BOOST_AUTO_TEST_SUITE(LSHTest);

BOOST_AUTO_TEST_CASE(LSHSearchTest)
{

  math::RandomSeed(time(0));
  //kNN and LSH parameters (use LSH default parameters)
  const int k = 4;
  const int numTables = 30;
  const int numProj = 10;
  const double hashWidth = 0;
  const int secondHashSize = 99901;
  const int bucketSize = 500;
  
  //test parameters
  const double epsilon = 0.1;

  //read iris training and testing data as reference and query
  const string data_train="iris_train.csv";
  const string data_test="iris_test.csv";
  arma::mat rdata;
  arma::mat qdata;
  data::Load(data_train, rdata, true);
  data::Load(data_test, qdata, true);

  //Run classic knn on reference data
  AllkNN knn(rdata);
  arma::Mat<size_t> groundTruth;
  arma::mat groundDistances;
  knn.Search(qdata, k, groundTruth, groundDistances);

  //Test: Run LSH with varying number of tables, keeping all other parameters 
  //constant. Compute the recall, i.e. the number of reported neighbors that
  //are real neighbors of the query.
  //LSH's property is that (with high probability), increasing the number of
  //tables will increase recall. Epsilon ensures that if noise lightly affects
  //the projections, the test will not fail.
  
  const int numTries = 5; //tries for each test before declaring failure
  bool fail = false;

  for (int t = 0; t < numTries; ++t){

    const int Lsize = 6; //number of runs
    const int L_value[] = {1, 8, 16, 32, 64, 128}; //number of tables
    double L_value_recall[Lsize] = {0.0}; //recall of each LSH run

    for (int l=0; l < Lsize; ++l)
    {
      //run LSH with only numTables varying (other values default)
      LSHSearch<> lsh_test1(rdata, numProj, L_value[l], 
          hashWidth, secondHashSize, bucketSize);
      arma::Mat<size_t> LSHneighbors;
      arma::mat LSHdistances;
      lsh_test1.Search(qdata, k, LSHneighbors, LSHdistances);

      //compute recall for each query
      L_value_recall[l] = compute_recall(LSHneighbors, groundTruth);

      if (l > 0){
        if(L_value_recall[l] < L_value_recall[l-1]-epsilon){
          fail = true; //if test fails at one point, stop and retry
          break;
        }
      }
    }
    if ( !fail ){
      break; //if test passes one time, it is sufficient
    }
  }
  BOOST_CHECK(fail == false);
   
  //Test: Run LSH with varying hash width, keeping all other parameters 
  //constant. Compute the recall, i.e. the number of reported neighbors that
  //are real neighbors of the query.
  //LSH's property is that (with high probability), increasing the hash width
  //will increase recall. Epsilon ensures that if noise lightly affects the 
  //projections, the test will not fail.
  
  const int Hsize = 7; //number of runs
  const double H_value[] = {0.1, 0.5, 1, 5, 10, 50, 500}; //hash width
  double H_value_recall[Hsize] = {0.0}; //recall of each run

  for (int h=0; h < Hsize; ++h)
  {
    //run LSH with only hashWidth varying (other values default)
    LSHSearch<> lsh_test2(rdata, numProj, numTables, 
            H_value[h], secondHashSize, bucketSize);
    arma::Mat<size_t> LSHneighbors;
    arma::mat LSHdistances;
    lsh_test2.Search(qdata, k, LSHneighbors, LSHdistances);

    //compute recall for each query
    H_value_recall[h] = compute_recall(LSHneighbors, groundTruth);

    if (h > 0)
        BOOST_CHECK(H_value_recall[h] >= H_value_recall[h-1]-epsilon);
    
  }

  //Test: Run LSH with varying number of Projections, keeping all other parameters 
  //constant. Compute the recall, i.e. the number of reported neighbors that
  //are real neighbors of the query.
  //LSH's property is that (with high probability), increasing the number of
  //projections per table will decrease recall. Epsilon ensures that if noise lightly 
  //affects the projections, the test will not fail.
 
  const int Psize = 5; //number of runs
  const int P_value[] = {1, 10, 20, 50, 100}; //number of projections
  double P_value_recall[Psize] = {0.0}; //recall of each run

  for (int p=0; p < Psize; ++p)
  {
    //run LSH with only numProj varying (other values default)
    LSHSearch<> lsh_test3(rdata, P_value[p], numTables, 
            hashWidth, secondHashSize, bucketSize);
    arma::Mat<size_t> LSHneighbors;
    arma::mat LSHdistances;
    lsh_test3.Search(qdata, k, LSHneighbors, LSHdistances);

    //compute recall for each query
    P_value_recall[p] = compute_recall(LSHneighbors, groundTruth);

    if (p > 0) //don't check first run, only that increasing P decreases recall
        BOOST_CHECK(P_value_recall[p] - epsilon <= P_value_recall[p-1]);
  }
  
  //Test: Run a very expensive LSH search, with a large number of hash tables
  //and a large hash width. This run should return an acceptable recall. We set
  //the bar very low (recall >= 50%) to make sure that a test fail means bad
  //implementation.
  
  const int H_exp = 10000; //first-level hash width
  const int K_exp = 1; //projections per table
  const int T_exp = 128; //number of tables
  const double recall_thresh_exp = 0.5;

  LSHSearch<> lsh_test_exp(rdata, K_exp, T_exp, H_exp, secondHashSize, bucketSize);
  arma::Mat<size_t> LSHneighbors_exp;
  arma::mat LSHdistances_exp;
  lsh_test_exp.Search(qdata, k, LSHneighbors_exp, LSHdistances_exp);
  
  const double recall_exp = compute_recall(LSHneighbors_exp, groundTruth);

  BOOST_CHECK(recall_exp >= recall_thresh_exp);

  //Test: Run a very cheap LSH search, with parameters that should cause recall
  //to be very low. Set the threshhold very high (recall <= 25%) to make sure
  //that a test fail means bad implementation.
  //This mainly checks that user-specified parameters are not ignored.
  
  const int H_chp = 1; //small first-level hash width
  const int K_chp = 1000; //large number of projections per table
  const int T_chp = 1; //only one table
  const double recall_thresh_chp = 0.25; //recall threshold

  LSHSearch<> lsh_test_chp(rdata, K_chp, T_chp, H_chp, secondHashSize, bucketSize);
  arma::Mat<size_t> LSHneighbors_chp;
  arma::mat LSHdistances_chp;
  lsh_test_chp.Search(qdata, k, LSHneighbors_chp, LSHdistances_chp);

  const double recall_chp = compute_recall(LSHneighbors_chp, groundTruth);
  BOOST_CHECK(recall_chp <= recall_thresh_chp);
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
