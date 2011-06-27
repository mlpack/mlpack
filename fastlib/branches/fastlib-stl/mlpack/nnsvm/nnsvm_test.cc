/**
 * @file nnsvm_test.cc
 *
 * Test file for NNSVM class
 */

#include <fastlib/fastlib.h>
#include <armadillo>
#include <iostream>
#include "nnsvm.h"

#define BOOST_TEST_MODULE NNSVM Test
#include <boost/test/unit_test.hpp>

using namespace mlpack;
using namespace mlpack::nnsvm;

/***
 * simple nonnegative SVM test with small, synthetic dataset
 */
BOOST_AUTO_TEST_CASE(exhaustive_synthetic_test)
{
  // initialize the matrix
  arma::mat data;
  std::cerr << "created the data object\n";

  data << -1.0 <<  1.0 << 1.0 << arma::endr
       << -2.0 <<  2.0 << 1.0 << arma::endr
       << -3.0 <<  3.0 << 1.0 << arma::endr
       << -4.0 <<  4.0 << 1.0 << arma::endr
       <<  1.0 << -1.0 << 0.0 << arma::endr
       <<  2.0 << -2.0 << 0.0 << arma::endr
       <<  3.0 << -3.0 << 0.0 << arma::endr
       <<  4.0 << -4.0 << 0.0 << arma::endr;
  std::cerr << "input the goodies\n";

  // test the linear kernel
  NNSVM<SVMLinearKernel>* nnsvm = new NNSVM<SVMLinearKernel>();

  nnsvm->InitTrain(data, 2);
  double calculatedThreshold = nnsvm->getThreshold();
  index_t calculatedSupportVectorCount = nnsvm->getSupportVectorCount();
  const arma::vec calculatedSupportVectorCoefficients = nnsvm->getSupportVectorCoefficients();
  const arma::vec calculatedWeightVector = nnsvm->getWeightVector();

  // check for correctness on the linear kernel
  BOOST_REQUIRE(calculatedSupportVectorCount == 3);
  BOOST_REQUIRE_CLOSE(calculatedThreshold, -1.0, 1e-5);
  BOOST_REQUIRE_CLOSE(calculatedSupportVectorCoefficients[0], 3.7499785159728178, 1e-5);
  BOOST_REQUIRE_CLOSE(calculatedSupportVectorCoefficients[1], 6.2500214840271884, 1e-5);
  BOOST_REQUIRE_CLOSE(calculatedSupportVectorCoefficients[2], -10.000, 1e-5);
  BOOST_REQUIRE_CLOSE(calculatedWeightVector[0], 0.00000000, 1e-5);
  BOOST_REQUIRE_CLOSE(calculatedWeightVector[1], 0.00000000, 1e-5);
  BOOST_REQUIRE_CLOSE(calculatedWeightVector[2], 0.00000000, 1e-5);
  BOOST_REQUIRE_CLOSE(calculatedWeightVector[3], 0.00017187221748210524, 1e-5);
  BOOST_REQUIRE_CLOSE(calculatedWeightVector[4], 0.00000000, 1e-5);
  BOOST_REQUIRE_CLOSE(calculatedWeightVector[5], 0.00000000, 1e-5);
  BOOST_REQUIRE_CLOSE(calculatedWeightVector[6], 0.00000000, 1e-5);

  // Clean the memory.
  delete nnsvm;
}

///***
// * Test the dual-tree nearest-neighbors method with the naive method.  This
// * uses both a query and reference dataset.
// *
// * Errors are produced if the results are not identical.
// */
//BOOST_AUTO_TEST_CASE(dual_tree_vs_naive_1)
//{
//  arma::mat data_for_tree_;
//
//  // Hard-coded filename: bad!
//  if (data::Load("test_data_3_1000.csv", data_for_tree_) != SUCCESS_PASS)
//  {
//    BOOST_FAIL("Cannot load test dataset test_data_3_1000.csv!");
//  }
//
//  // Set up matrices to work with.
//  arma::mat dual_query(data_for_tree_);
//  arma::mat dual_references(data_for_tree_);
//  arma::mat naive_query(data_for_tree_);
//  arma::mat naive_references(data_for_tree_);
//
//  AllkNN allknn_(dual_query, dual_references, 20, 5);
//  AllkNN naive_(naive_query, naive_references, 1 /* leaf_size ignored */, 5,
//      AllkNN::NAIVE);
//
//  arma::Col<index_t> resulting_neighbors_tree;
//  arma::vec distances_tree;
//  allknn_.ComputeNeighbors(resulting_neighbors_tree, distances_tree);
//
//  arma::Col<index_t> resulting_neighbors_naive;
//  arma::vec distances_naive;
//  naive_.ComputeNeighbors(resulting_neighbors_naive, distances_naive);
//
//  for (index_t i = 0; i < resulting_neighbors_tree.n_elem; i++)
//  {
//    BOOST_REQUIRE(resulting_neighbors_tree[i] == resulting_neighbors_naive[i]);
//    BOOST_REQUIRE_CLOSE(distances_tree[i], distances_naive[i], 1e-5);
//  }
//}
//
///***
// * Test the dual-tree nearest-neighbors method with the naive method.  This uses
// * only a reference dataset.
// *
// * Errors are produced if the results are not identical.
// */
//BOOST_AUTO_TEST_CASE(dual_tree_vs_naive_2)
//{
//  arma::mat data_for_tree_;
//
//  // Hard-coded filename: bad!
//  // Code duplication: also bad!
//  if (data::Load("test_data_3_1000.csv", data_for_tree_) != SUCCESS_PASS)
//  {
//    BOOST_FAIL("Cannot load test dataset test_data_3_1000.csv!");
//  }
//
//  // Set up matrices to work with (may not be necessary with no ALIAS_MATRIX?).
//  arma::mat dual_query(data_for_tree_);
//  arma::mat naive_query(data_for_tree_);
//
//  AllkNN allknn_(dual_query, 20, 1);
//  AllkNN naive_(naive_query, 1 /* leaf_size ignored with naive */, 1,
//      AllkNN::NAIVE);
//
//  arma::Col<index_t> resulting_neighbors_tree;
//  arma::vec distances_tree;
//  allknn_.ComputeNeighbors(resulting_neighbors_tree, distances_tree);
//
//  arma::Col<index_t> resulting_neighbors_naive;
//  arma::vec distances_naive;
//  naive_.ComputeNeighbors(resulting_neighbors_naive, distances_naive);
//
//  for (index_t i = 0; i < resulting_neighbors_tree.n_elem; i++) {
//    BOOST_REQUIRE(resulting_neighbors_tree[i] == resulting_neighbors_naive[i]);
//    BOOST_REQUIRE_CLOSE(distances_tree[i], distances_naive[i], 1e-5);
//  }
//}
//
///***
// * Test the single-tree nearest-neighbors method with the naive method.  This
// * uses only a reference dataset.
// *
// * Errors are produced if the results are not identical.
// */
//BOOST_AUTO_TEST_CASE(single_tree_vs_naive)
//{
//  arma::mat data_for_tree_;
//
//  // Hard-coded filename: bad!
//  // Code duplication: also bad!
//  if (data::Load("test_data_3_1000.csv", data_for_tree_) != SUCCESS_PASS)
//    BOOST_FAIL("Cannot load test dataset test_data_3_1000.csv!");
//
//  // Set up matrices to work with (may not be necessary with no ALIAS_MATRIX?).
//  arma::mat single_query(data_for_tree_);
//  arma::mat naive_query(data_for_tree_);
//
//  AllkNN allknn_(single_query, 20, 5, AllkNN::MODE_SINGLE);
//  AllkNN naive_(naive_query, 1 /* leaf_size ignored with naive */, 5,
//      AllkNN::NAIVE);
//
//  arma::Col<index_t> resulting_neighbors_tree;
//  arma::vec distances_tree;
//  allknn_.ComputeNeighbors(resulting_neighbors_tree, distances_tree);
//
//  arma::Col<index_t> resulting_neighbors_naive;
//  arma::vec distances_naive;
//  naive_.ComputeNeighbors(resulting_neighbors_naive, distances_naive);
//
//  for (index_t i = 0; i < resulting_neighbors_tree.n_elem; i++) {
//    BOOST_REQUIRE(resulting_neighbors_tree[i] == resulting_neighbors_naive[i]);
//    BOOST_REQUIRE_CLOSE(distances_tree[i], distances_naive[i], 1e-5);
//  }
//}
