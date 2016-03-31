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
  //const int numTables = 30;
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

  //Run LSH for different number of tables
  const int L_table[] = {1, 8, 16, 32, 64, 128}; //number of tables
  const int Lsize = 6;
  double recall_L[Lsize] = {0.0}; //recall of each LSH run

  for (int l=0; l < Lsize; ++l)
  {
    //run LSH with only numTables varying (other values default)
    LSHSearch<> lsh_test1(rdata, numProj, L_table[l], 
            hashWidth, secondHashSize, bucketSize);
    arma::Mat<size_t> LSHneighbors;
    arma::mat LSHdistances;
    lsh_test1.Search(qdata, k, LSHneighbors, LSHdistances);

    //compute recall for each query
    recall_L[l] = compute_recall(LSHneighbors, groundTruth);

    if (l > 0)
        BOOST_CHECK(recall_L[l] >= recall_L[l-1]-epsilon);
    
  }

  //Test: Run a very expensive LSH search, with a large number of hash tables
  //and a large hash width. This run should return an acceptable recall. We set
  //the bar very low (recall >= 50%) to make sure that a test fail means bad
  //implementation.
  
  const int Ht2 = 10000; //first-level hash width
  const int Kt2 = 128; //projections per table
  const int Tt2 = 128; //number of tables
  const double recall_thresh_t2 = 0.5;

  LSHSearch<> lsh_test2(rdata, Kt2, Tt2, Ht2, secondHashSize, bucketSize);
  arma::Mat<size_t> LSHneighbors;
  arma::mat LSHdistances;
  lsh_test2.Search(qdata, k, LSHneighbors, LSHdistances);
  
  const double recall_t2 = compute_recall(LSHneighbors, groundTruth);

  BOOST_CHECK(recall_t2 >= recall_thresh_t2);

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
