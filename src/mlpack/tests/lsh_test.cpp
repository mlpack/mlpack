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

BOOST_AUTO_TEST_SUITE(LSHTest);

BOOST_AUTO_TEST_CASE(LSHSearchTest)
{
  //kNN and LSH parameters
  const int k = 4;
  const int numProj = 10;
  const double hashWidth = 0;
  const int secondHashSize = 99901;
  const int bucketSize = 500;

  //read iris training and testing data as reference and query
  string iris_train="iris_train.csv";
  string iris_test="iris_test.csv";
  arma::mat rdata;
  arma::mat qdata;
  data::Load(iris_train, rdata, true);
  data::Load(iris_test, qdata, true);
  const int n_queries = qdata.n_cols;


  //Run classic knn on reference data
  AllkNN knn(rdata);
  arma::Mat<size_t> groundTruth;
  arma::mat groundDistances;
  knn.Search(qdata, k, groundTruth, groundDistances);
  

  //Test: Run LSH with varying number of tables, keeping all other parameters 
  //constant. Compute the recall, i.e. the number of reported neighbors that
  //are real neighbors of the query.
  //LSH's property is that (with high probability), increasing the number of
  //tables will increase recall.


  //Run LSH with varying number of tables and compute recall
  const int L[] = {1, 4, 16, 32, 64, 128}; //number of tables
  const int Lsize = 6;
  int recall_L[Lsize] = {0}; //recall of each LSH run

  for (int l=0; l < Lsize; ++l){
      //run LSH with only numTables varying (other values default)
      LSHSearch<> lsh_test(rdata, numProj, L[l], 
              hashWidth, secondHashSize, bucketSize);
      arma::Mat<size_t> LSHneighbors;
      arma::mat LSHdistances;
      lsh_test.Search(qdata, k, LSHneighbors, LSHdistances);

      //compute recall for each query
      for (int q=0; q < n_queries; ++q){
          for (int neigh = 0; neigh < k; ++neigh){
              if (LSHneighbors(neigh, q) == groundTruth(neigh, q))
                  recall_L[l]++;
          }
      }

      if (l > 0)
          BOOST_CHECK(recall_L[l] >= recall_L[l-1]);
  }

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
