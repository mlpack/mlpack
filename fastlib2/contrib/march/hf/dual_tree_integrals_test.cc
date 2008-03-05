#include "dual_tree_integrals.h"
#include <fastlib/base/test.h>

class DualTreeIntegralsTest {
  
public:
  
  void Setup() {
    
    Matrix test_centers;
    data::Load("test_centers.csv", &test_centers);
    
    double test_bandwidth = 0.01;
    
    double test_error = 0.01;
    
    integrals_ = new DualTreeIntegrals();
    integrals_->Init(test_centers, NULL, test_bandwidth, test_error);
    
  } // Setup
  
  
  void Destruct() {
    
    delete integrals_;
    
  } // Destruct
  
  void TestLowerBound() {
    
    Setup();
    
    NONFATAL("TestLowerBound not yet written!\n");
    
    Destruct();
    
  } // TestLowerBound

  void TestUpperBound() {
    
    Setup();
    
    NONFATAL("TestUpperBound not yet written!\n");
    
    Destruct();
    
  } // TestLowerBound  
  
  void TestErfLikeFunction() {
    
    Setup();
    
    NONFATAL("TestErfLikeFunction not yet written!\n");
    // Only waiting for a Matlab test case
    
    printf("test1 = %f\n", integrals_->ErfLikeFunction_(0.5));
    printf("test2 = %f\n", integrals_->ErfLikeFunction_(5.0));

    Destruct();
    
  } // TestErfLikeFunction
  
  void TestComputeSingleIntegralDists() {
    
    Setup();
    
    NONFATAL("TestComputeSingleIntegralDists not yet written!\n");
    
    Destruct();
    
  } // TestComputeSingleIntegralDists
  
  void TestComputeSingleIntegral() {
    
    Setup();
    
    NONFATAL("TestComputeSingleIntegral not yet written!\n");
    
    // Only waiting for a matlab test case
    
    Destruct();
    
  } // TestComputeSingleIntegral
  
  void TestCanApproximate() {
    
    Setup();
    
    NONFATAL("TestCanApproximate not yet written!\n");
    
    // Need to figure out how to distribute error from Dong's code
    // Also, how to determine bounds on the four-way distance
    
    Destruct();
    
  } // TestCanApproximate
  
  void TestComputeIntegralsBaseCase() {
    
    Setup();
    
    NONFATAL("TestComputeIntegralsBaseCase not yet written!\n");
    
    // Should be done, not sure how to test it though
    
    Destruct();
    
  } // TestComputeIntegralsBaseCase
  
  void TestComputeIntegralsRecursion() {
    
    Setup();
    
    NONFATAL("TestComputeIntegralsRecursion not yet written!\n");
    
    Destruct();
    
  } // TestComputeIntegralsRecursion
  
  void TestComputeTwoElectronIntegrals() {
    
    Setup();
    
    NONFATAL("TestComputeTwoElectronIntegrals not yet written!\n");
    
    Destruct();
    
  } // TestComputeTwoElectronIntegrals
  
  void TestAll() {
    
    TestErfLikeFunction();
    
    TestComputeSingleIntegralDists();
    
    TestComputeSingleIntegral();
    
    TestCanApproximate();
    
    TestComputeIntegralsBaseCase();
    
    TestComputeIntegralsRecursion();
    
    TestComputeTwoElectronIntegrals();
    
  } // TestAll
  
private:
  
  DualTreeIntegrals* integrals_;
  
}; //class DualTreeIntegralsTest


int main(int argc, char* argv[]) {
  
  DualTreeIntegralsTest tester;
  tester.TestAll();
  
  return 0;
  
} // main