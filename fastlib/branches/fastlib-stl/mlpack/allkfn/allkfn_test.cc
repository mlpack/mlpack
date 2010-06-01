/**
 * @allkfn_test.cc
 * Test file for AllkFN class
 */

#include "allkfn.h"
#include "fastlib/base/test.h"

class TestAllkFN {
 public:
  void Init() {
    allkfn_ = new AllkFN();
    naive_  = new AllkFN();
    data_for_tree_ = new Matrix();
    data::Load("test_data_3_1000.csv", data_for_tree_);
 }

  void Destruct() {
   delete data_for_tree_;
   delete allkfn_; 
   delete naive_;
  }

  void TestTreeVsNaive1() {
    Init();
    allkfn_->Init(*data_for_tree_, *data_for_tree_, 20, 5);
    naive_->InitNaive(*data_for_tree_, *data_for_tree_, 5);
 
    ArrayList<index_t> resulting_neighbors_tree;
    ArrayList<double> distances_tree;
    allkfn_->ComputeNeighbors(&resulting_neighbors_tree,
                              &distances_tree);
    ArrayList<index_t> resulting_neighbors_naive;
    ArrayList<double> distances_naive;
    naive_->ComputeNaive(&resulting_neighbors_naive,
                         &distances_naive);
    for(index_t i=0; i<resulting_neighbors_tree.size(); i++) {
      TEST_ASSERT(resulting_neighbors_tree[i] == resulting_neighbors_naive[i]);
      TEST_DOUBLE_APPROX(distances_tree[i], distances_naive[i], 1e-5);
    }
    NOTIFY("Allknn test 1 passed");
    Destruct();
  }
   void TestTreeVsNaive2() {
    Init();
    allkfn_->Init(*data_for_tree_, 20, 5);
    naive_->InitNaive(*data_for_tree_, 5);

    ArrayList<index_t> resulting_neighbors_tree;
    ArrayList<double> distances_tree;
    allkfn_->ComputeNeighbors(&resulting_neighbors_tree,
                              &distances_tree);
    ArrayList<index_t> resulting_neighbors_naive;
    ArrayList<double> distances_naive;
    naive_->ComputeNaive(&resulting_neighbors_naive,
                         &distances_naive);
    for(index_t i=0; i<resulting_neighbors_tree.size(); i++) {
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
  Matrix *data_for_tree_;
};

int main() {
 TestAllkFN test;
 test.TestAll();
}
