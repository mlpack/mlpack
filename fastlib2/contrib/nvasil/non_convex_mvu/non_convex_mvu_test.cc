/*
 * =====================================================================================
 *
 *       Filename:  non_convex_mvu_test.cc
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  02/05/2008 12:56:45 PM EST
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 *
 * =====================================================================================
 */

#include "non_convex_mvu.h"

class NonConvexMVUTest {
 public:
  
  void Init() {
    engine_ = new NonConvexMVU();
  }
  
  void Destruct() {
    delete engine_;  
  }
  
  void TestInit() {
    NOTIFY("Testing Init() ...\n");
    engine_->Init("test_data_3_1000.csv", 5);
    engine_->coordinates_.Init(1, 1);
    engine_->gradient_.Init(1, 1);
    engine_->previous_gradient_.Init(1, 1);
    engine_->previous_coordinates_.Init(1, 1);
    engine_->ro_bfgs_.Init(1);
    engine_->s_bfgs_.Init();
    engine_->y_bfgs_.Init();
    engine_->lagrange_mult_.Init(30);
    engine_->centering_lagrange_mult_.Init(20);
    NOTIFY("TestInit passed!!\n");
  }
  
  void TestComputeLocalOptimum() {
    NOTIFY("Testing ComputeLocalOptimum() ...\n");
    engine_->Init("test_data_3_1000.csv", 5);
    engine_->set_new_dimension(3);
    engine_->ComputeLocalOptimum();
    NOTIFY("TestComputeLocalOptimum() passed!!\n");
  }
  void TestComputeLocalOptimumBFGS() {
    NOTIFY("Testing ComputeLocalOptimum() ...\n");
    engine_->Init("test_data_3_1000.csv", 5);
    engine_->set_new_dimension(3);
    engine_->set_mem_bfgs(5);
    engine_->ComputeLocalOptimumBFGS();
    NOTIFY("TestComputeLocalOptimum() passed!!\n");
  }
 
  void TestAll() {
    Init();
    TestInit();
    Destruct();
   // Init();
   // TestComputeLocalOptimum();
   //  Destruct();
    Init();
    TestComputeLocalOptimumBFGS();
    Destruct();
  }
  
 private:
  NonConvexMVU *engine_;  
};

int main() {
  NonConvexMVUTest test;
  test.TestAll();
}
