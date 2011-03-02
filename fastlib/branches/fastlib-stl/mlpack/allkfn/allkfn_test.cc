/**
 * @allkfn_test.cc
 * Test file for AllkFN class
 */

#include "allkfn.h"
#include "fastlib/base/test.h"

#include <armadillo>
#include <fastlib/base/arma_compat.h>

class TestAllkFN {
 public:
  void Init() {
    allkfn_ = new AllkFN();
    naive_  = new AllkFN();
    if(data::Load("test_data_3_1000.csv", data_for_tree_) != SUCCESS_PASS)
      FATAL("Unable to load test dataset.");
 }

  void Destruct() {
   delete allkfn_; 
   delete naive_;
  }

  void TestTreeVsNaive1() {
    Init();
    arma::mat dual_query(data_for_tree_);
    arma::mat dual_references(data_for_tree_);
    arma::mat naive_query(data_for_tree_);
    arma::mat naive_references(data_for_tree_);
    allkfn_->Init(&dual_query, &dual_references, 20, 5);
    naive_->InitNaive(&naive_query, &naive_references, 5);
 
    arma::Col<index_t> resulting_neighbors_tree;
    arma::vec distances_tree;
    allkfn_->ComputeNeighbors(resulting_neighbors_tree,
                              distances_tree);
    arma::Col<index_t> resulting_neighbors_naive;
    arma::vec distances_naive;
    naive_->ComputeNaive(resulting_neighbors_naive,
                         distances_naive);
    for(index_t i = 0; i < resulting_neighbors_tree.n_elem; i++) {
      TEST_ASSERT(resulting_neighbors_tree[i] == resulting_neighbors_naive[i]);
      TEST_DOUBLE_APPROX(distances_tree[i], distances_naive[i], 1e-5);
    }
    NOTIFY("Allknn test 1 passed");
    Destruct();
  }
   void TestTreeVsNaive2() {
    Init();
    arma::mat dual_references(data_for_tree_);
    arma::mat naive_references(data_for_tree_);
    allkfn_->Init(&dual_references, 20, 5);
    naive_->InitNaive(&naive_references, 5);

    arma::Col<index_t> resulting_neighbors_tree;
    arma::vec distances_tree;
    allkfn_->ComputeNeighbors(resulting_neighbors_tree,
                              distances_tree);
    arma::Col<index_t> resulting_neighbors_naive;
    arma::vec distances_naive;
    naive_->ComputeNaive(resulting_neighbors_naive,
                         distances_naive);
    for(index_t i = 0; i < resulting_neighbors_tree.n_elem; i++) {
      TEST_ASSERT(resulting_neighbors_tree[i] == resulting_neighbors_naive[i]);
      TEST_DOUBLE_APPROX(distances_tree[i], distances_naive[i], 1e-5);
    }
    NOTIFY("Allknn test 2 passed");
    Destruct();
  }
 
  void TestAll() {
    TestTreeVsNaive1();
    TestTreeVsNaive2();
 }
 
 private:
  AllkFN *allkfn_;
  AllkFN *naive_;
  arma::mat data_for_tree_;
};

int main() {
 TestAllkFN test;
 test.TestAll();
}
