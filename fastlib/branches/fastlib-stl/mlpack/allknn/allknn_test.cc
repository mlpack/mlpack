/**
 * @allknn_test.cc
 * Test file for AllkNN class
 */

#include "allknn.h"
#include "fastlib/base/test.h"

#include <armadillo>

using namespace mlpack::allknn;

namespace mlpack {
namespace allknn {

class TestAllkNN {
 public:
  void Init() {
//    allknn_ = new AllkNN();
//    naive_  = new AllkNN();
    data::Load("test_data_3_1000.csv", data_for_tree_);
 }

  void Destruct() {
   delete allknn_; 
   delete naive_;
  }

  void TestDualTreeVsNaive1() {
    Init();
    arma::mat dual_query(data_for_tree_);
    arma::mat dual_references(data_for_tree_);
    arma::mat naive_query(data_for_tree_);
    arma::mat naive_references(data_for_tree_);

    allknn_ = new AllkNN(dual_query, dual_references, 20, 5);
    naive_ = new AllkNN(naive_query, naive_references, 1 /* leaf_size ignored
        */, 5, AllkNN::NAIVE);
 
    arma::Col<index_t> resulting_neighbors_tree;
    arma::vec distances_tree;
    allknn_->ComputeNeighbors(resulting_neighbors_tree,
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

  void TestDualTreeVsNaive2() {
    arma::mat dual_query(data_for_tree_);
    arma::mat naive_query(data_for_tree_);

    allknn_ = new AllkNN(dual_query, 20, 1);
    naive_ = new AllkNN(naive_query, 1 /* leaf_size ignored with naive */, 1, AllkNN::NAIVE);

    arma::Col<index_t> resulting_neighbors_tree;
    arma::vec distances_tree;
    allknn_->ComputeNeighbors(resulting_neighbors_tree,
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

  void TestSingleTreeVsNaive() {
    arma::mat single_query(data_for_tree_);
    arma::mat naive_query(data_for_tree_);

    allknn_ = new AllkNN(single_query, 20, 5, AllkNN::MODE_SINGLE);
    naive_ = new AllkNN(naive_query, 1 /* leaf_size ignored with naive */, 5, AllkNN::NAIVE);

    arma::Col<index_t> resulting_neighbors_tree;
    arma::vec distances_tree;
    allknn_->ComputeNeighbors(resulting_neighbors_tree,
                              distances_tree);
    arma::Col<index_t> resulting_neighbors_naive;
    arma::vec distances_naive;
    naive_->ComputeNaive(resulting_neighbors_naive,
                         distances_naive);
    for(index_t i = 0; i < resulting_neighbors_tree.n_elem; i++) {
      TEST_ASSERT(resulting_neighbors_tree[i] == resulting_neighbors_naive[i]);
      TEST_DOUBLE_APPROX(distances_tree[i], distances_naive[i], 1e-5);
    }
    NOTIFY("Allknn test 3 passed");
    Destruct();
  }

  void TestAll() {
    TestDualTreeVsNaive1();
    TestDualTreeVsNaive2();
    TestSingleTreeVsNaive();
 }
 
 private:
  AllkNN *allknn_;
  AllkNN *naive_;
  arma::mat data_for_tree_;
};

}; // namespace allknn
}; // namespace mlpack

int main(int argc, char *argv[]) {
  fx_root = fx_init(argc, argv, NULL);
  TestAllkNN test;
  test.TestAll();
  fx_done(fx_root);
}
