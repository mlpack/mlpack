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
#include "fastlib/base/test.h"

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
    engine_->Init<MaxVariance,
                  EqualityOnNearest,
                  DeterministicGrad>("test_data_3_1000.csv", 5, 0);
    engine_->coordinates_.Init(1, 1);
    engine_->gradient_.Init(1, 1);
    engine_->previous_gradient_.Init(1, 1);
    engine_->previous_coordinates_.Init(1, 1);
    engine_->ro_bfgs_.Init(1);
    engine_->s_bfgs_.Init();
    engine_->y_bfgs_.Init();
    engine_->lagrange_mult_.Init(30);
    NOTIFY("TestInit passed!!\n");
  }
/*  
 void TestFeasibilityGradient(){
    NOTIFY("Testing  Feasibility and Gradient...\n");
    engine_->Init("swiss_roll.csv", 10, 0);
    engine_->set_new_dimension(3);
    engine_->coordinates_.Copy(engine_->data_);
    double distance_constraint, center_constraint;
    // Test to see if it really evaluates the 
    // constraints correctly
    engine_->ComputeFeasibilityError_<1>(
        &distance_constraint, &center_constraint);
    TEST_DOUBLE_APPROX(distance_constraint, 0.0, 1e-5);
    engine_->ComputeFeasibilityError_<1>(
        &distance_constraint, &center_constraint);
    TEST_DOUBLE_APPROX(distance_constraint, 0.0, 1e-5);
    // test to see if the lagrangian evaluates to 0
    TEST_DOUBLE_APPROX(engine_->ComputeLagrangian_<0>(engine_->coordinates_), 
        0, 1e-5);
    
    engine_->gradient_.Copy(engine_->data_);
    engine_->lagrange_mult_.Init(5 *engine_->num_of_points_);
    engine_->lagrange_mult_.SetAll(0.0);
    engine_->sigma_=1;
    engine_->ComputeGradient_<0>(engine_->coordinates_,&engine_->gradient_);
    la::SubFrom(engine_->gradient_, &engine_->coordinates_);
    engine_->ComputeFeasibilityError_<0>(
        &distance_constraint, &center_constraint);
    TEST_DOUBLE_APPROX(distance_constraint, 0.0, 1e-5);

    // These are dummy initializations because of fastlib
    engine_->previous_gradient_.Init(1, 1);
    engine_->previous_coordinates_.Init(1, 1);
    engine_->ro_bfgs_.Init(1);
    engine_->s_bfgs_.Init();
    engine_->y_bfgs_.Init();
    NOTIFY("TestFeasibilityGradient() passed!!\n"); 
  }
  void TestComputeLocalOptimumBFGS1() {
    NOTIFY("Testing ComputeLocalOptimumBFGS() ...\n");
    engine_->Init("test_data_3_100.csv", 5, 0);
    engine_->set_new_dimension(3);
    engine_->set_sigma(100);
    engine_->set_gamma(3);
    engine_->set_eta(0.25);
    engine_->set_distance_tolerance(1e-5);
    engine_->set_mem_bfgs(150);
    engine_->ComputeLocalOptimumBFGS<2>();
    data::Save("results.csv", engine_->coordinates());
    NOTIFY("TestComputeLocalOptimumBFGS1() passed!!\n");
  }
  void TestComputeLocalOptimumBFGS2() {
    NOTIFY("Testing ComputeLocalOptimumBFGS() ...\n");
    engine_->Init("test_data_3_100.csv", 5, 0);
    engine_->set_new_dimension(2);
    engine_->set_sigma(140);
    engine_->set_gamma(4);
    engine_->set_eta(0.999);
    engine_->set_distance_tolerance(1e-5);
    engine_->set_mem_bfgs(150);
    engine_->ComputeLocalOptimumBFGS<2>();
    data::Save("results.csv", engine_->coordinates());
    NOTIFY("TestComputeLocalOptimumBFGS2() passed!!\n");
  }
  void TestComputeLocalOptimumBFGS3() {
    NOTIFY("Testing ComputeLocalOptimumBFGS() ...\n");
    engine_->Init("test_data_3_1000.csv", 5, 0);
    engine_->set_new_dimension(3);
    engine_->set_sigma(1e6);
    engine_->set_gamma(4);
    engine_->set_eta(0.9);
    engine_->set_wolfe_beta(0.8);
    engine_->set_distance_tolerance(5*1e-5);
    engine_->set_mem_bfgs(150);
    engine_->ComputeLocalOptimumBFGS<0>();
    data::Save("results.csv", engine_->coordinates());
    NOTIFY("TestComputeLocalOptimumBFGS3() passed!!\n");
  }
*/  
  void TestComputeLocalOptimumBFGS4() {
    NOTIFY("Testing ComputeLocalOptimumBFGS() ...\n");
    engine_->Init<MaxVariance,
                  EqualityOnNearest,
                  DeterministicGrad>("swiss_roll.csv", 5, 1);
    engine_->set_new_dimension(2);
    engine_->set_sigma(1e3);
    engine_->set_gamma(5);
    engine_->set_eta(0.999999);
    engine_->set_wolfe_beta(0.9);
    engine_->set_distance_tolerance(1e-3);
    engine_->set_mem_bfgs(150);
    engine_->ComputeLocalOptimumBFGS<MaxFurthestNeighbors, 
                                    EqualityOnNearest,
                                    DeterministicGrad>();
    data::Save("results.csv", engine_->coordinates());
    NOTIFY("TestComputeLocalOptimumBFGS4() passed!!\n");
  }

  void TestComputeLocalOptimumBFGS5() {
    NOTIFY("Testing ComputeLocalOptimumBFGS() ...\n");
    engine_->Init<Feasibility,
                  EqualityOnNearest,
                  StochasticGrad>("swiss_roll.csv", 5, 1);
    engine_->set_new_dimension(2);
    engine_->set_sigma(1);
    engine_->set_gamma(1);
    engine_->set_eta(0.999999);
    engine_->set_wolfe_beta(0.5);
    engine_->set_distance_tolerance(1e-3);
    engine_->set_mem_bfgs(150);
    engine_->ComputeLocalOptimumSGD<Feasibility, 
                                    EqualityOnNearest,
                                    StochasticGrad>();
    data::Save("results.csv", engine_->coordinates());
    NOTIFY("TestComputeLocalOptimumBFGS5() passed!!\n");
  }

  void TestAll() {
//    Init();
//    TestInit();
//    Destruct();
 //   Init();
//    TestFeasibilityGradient();
//    Destruct();
    Init();
    TestComputeLocalOptimumBFGS5();
    Destruct();
  }
  
 private:
  NonConvexMVU *engine_;  
};

int main() {
  NonConvexMVUTest test;
  test.TestAll();
}
