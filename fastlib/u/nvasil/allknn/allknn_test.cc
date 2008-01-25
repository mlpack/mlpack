/**
 * @allknn_test.cc
 * Test file for AllkNN class
 */

#include "allknn.h"
#include "base/test.h"

class TestAllkNN {
 public:
  void Init() {
    allknn_ = new AllkNN();
    naive_  = new AllkNN();
    data_for_tree_ = new Matrix();
    data_for_naive_= new Matrix();
    data::Load("test_data_3_1000.csv", data_for_tree_);
    data::Load("test_data_3_1000.csv", data_for_naive_);
    allknn_->Init(*data_for_tree_, *data_for_tree_, 20, 5);
    naive_->InitNaive(*data_for_naive_, *data_for_naive_, 5);
  }

  void Destruct() {
   delete data_for_tree_;
   delete data_for_naive_;
   delete allknn_; 
   delete naive_;
  }

  void TestTreeVsNaive() {
    Init();
    ArrayList<index_t> resulting_neighbors_tree;
    ArrayList<double> distances_tree;
    allknn_->ComputeNeighbors(&resulting_neighbors_tree,
                              &distances_tree);
    ArrayList<index_t> resulting_neighbors_naive;
    ArrayList<double> distances_naive;
    naive_->ComputeNaive(&resulting_neighbors_naive,
                         &distances_naive);
    for(index_t i=0; i<resulting_neighbors_tree.size(); i++) {
      TEST_ASSERT(resulting_neighbors_tree[i] == resulting_neighbors_naive[i]);
      TEST_DOUBLE_APPROX(distances_tree[i], distances_naive[i], 1e-5);
    }
    NOTIFY("Allknn test passed");
    Destruct();
  }
  
  void TestAll() {
    TestTreeVsNaive();
  }
 
 private:
  AllkNN *allknn_;
  AllkNN *naive_;
  Matrix *data_for_tree_;
  Matrix *data_for_naive_;
};

int main() {
 TestAllkNN test;
 test.TestAll();
}
