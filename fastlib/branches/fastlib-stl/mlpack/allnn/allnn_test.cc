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

#include <armadillo>
#include <fastlib/base/arma_compat.h>

using namespace mlpack;
using namespace mlpack::allnn;

class TestAllNN {
 public:
  TestAllNN(fx_module *module) {
    module_ = module;
  }
  void Init() {
    data::Load("test_data_3_1000.csv", data_for_tree_);
  }
  void Destruct() {
    delete allnn_;
    delete naive_;
  }

  void TestTreeVsNaive1() {
    Init();
    allnn_ = new AllNN(data_for_tree_, module_);
    naive_ = new AllNN(data_for_tree_, module_);

    // run dual-tree allnn 
    arma::Col<index_t> resulting_neighbors_tree;
    arma::vec resulting_distances_tree;
    allnn_->ComputeNeighbors(resulting_distances_tree, resulting_neighbors_tree);

    // run naive allnn
    arma::Col<index_t> resulting_neighbors_naive;
    arma::vec resulting_distances_naive;
    naive_->ComputeNaive(resulting_distances_naive, resulting_neighbors_naive);

    // compare results
    for(index_t i = 0; i < resulting_neighbors_tree.n_elem; i++) {
      TEST_ASSERT(resulting_neighbors_tree[i] == resulting_neighbors_naive[i]);
      TEST_DOUBLE_APPROX(resulting_distances_tree[i], resulting_distances_naive[i], 1e-5);
    }
    NOTIFY("allnn test 1 passed");
    Destruct();
  }
 
  void TestAll() {
    TestTreeVsNaive1();
 }
 
 private:
  AllNN *allnn_;
  AllNN *naive_;

  arma::mat data_for_tree_;

  fx_module *module_;
};

int main(int argc, char *argv[]) {
 fx_module *fx_root = fx_init(argc, argv, NULL); 
 fx_set_param_int(fx_root, "leaf_size", 20);
 TestAllNN test(fx_root);
 test.TestAll();
}
