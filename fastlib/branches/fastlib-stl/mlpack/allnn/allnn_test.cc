/*
 * =====================================================================================
 *
 *       Filename:  allnn_test.cc
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  07/11/2008 03:12:57 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 *
 * =====================================================================================
 */

/**
 * @allnn_test.cc
 * Test file for DiskAllkNN class
 */


#include "allnn.h"
#include "fastlib/base/test.h"
#include "fastlib/fx/io.h"

#include <iostream>

#include <armadillo>
#include <fastlib/base/arma_compat.h>

using namespace mlpack;
using namespace mlpack::allnn;

class TestAllNN {
 public:
  TestAllNN() {
  }
  void Init() {
    if(data::Load("test_data_3_1000.csv", data_for_tree_) != SUCCESS_PASS)
      IO::printFatal("Unable to load test dataset.");
  }
  void Destruct() {
    delete allnn_;
    delete naive_;
  }

  void TestTreeVsNaive1() {
    Init();
    allnn_ = new AllNN(data_for_tree_);
    naive_ = new AllNN(data_for_tree_);

    // run dual-tree allnn 
    arma::Col<index_t> resulting_neighbors_tree;
    arma::vec resulting_distances_tree;
    allnn_->ComputeNeighbors(resulting_distances_tree, resulting_neighbors_tree);

    // run naive allnn
    arma::Col<index_t> resulting_neighbors_naive;
    arma::vec resulting_distances_naive;
    IO::startTimer("allnn/dual_tree_computation");
  
    naive_->ComputeNeighbors(resulting_distances_naive, resulting_neighbors_naive);
    IO::stopTimer("allnn/dual_tree_computation");
    // compare results
    for(index_t i = 0; i < resulting_neighbors_tree.n_elem; i++) {
      TEST_ASSERT(resulting_neighbors_tree[i] == resulting_neighbors_naive[i]);
      TEST_DOUBLE_APPROX(resulting_distances_tree[i], resulting_distances_naive[i], 1e-5);
    }
    
    Destruct();
  }
 
  void TestAll() {
    TestTreeVsNaive1();
 }
 
 private:
  AllNN *allnn_;
  AllNN *naive_;

  arma::mat data_for_tree_;

};

int main(int argc, char *argv[]) {
 TestAllNN test;
  
  AllNN::loadDocumentation();
  IO::parseCommandLine(argc, argv);
  test.TestAll();
  std::cout << IO::getValue<timeval>("allnn/dual_tree_computation").tv_usec << std::endl;
}
