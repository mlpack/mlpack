#include "dual_tree_integrals.h"
#include <fastlib/base/test.h>

class DualTreeIntegralsTest {
  
public:
  
  void Setup() {
    
    Matrix test_centers;
    data::Load("test_centers.csv", &test_centers);
    
    Matrix test_density;
    data::Load("density_test.csv", &test_density);
    
    double test_bandwidth = 0.01;
    
    double test_error = 0.01;
    
    integrals_ = new DualTreeIntegrals();
    integrals_->Init(test_centers, NULL, test_bandwidth, test_error, 
                     test_density);
                     
  } // Setup
  
  
  void Destruct() { 
    
    delete integrals_;
    
  } // Destruct
  
  bool TestPreOrderTraversalHelper(DualTreeIntegrals::IntegralTree* node) {
  
    printf("node: node_index = %d, start = %d, end = %d\n", 
        node->stat().node_index(), node->begin(), node->end());
  
    if (!(node->is_leaf())) {
      bool left = TestPreOrderTraversalHelper(node->left());
      bool right = TestPreOrderTraversalHelper(node->right());
      
      return ((left && right) 
              && (node->left()->stat().node_index() < 
                  node->right()->stat().node_index()) 
              && (node->right()->stat().node_index() < 
                  node->stat().node_index()));
    }
    else {
      return true;
    }
  
  }
  
  void TestPreOrderTraversal() {
  
    Setup();
    
    bool passed = TestPreOrderTraversalHelper(integrals_->tree_);
    
    Destruct();
    
    TEST_ASSERT(passed);
    
    NONFATAL("TestPreOrderTraversal Passed\n");
  
  } // TestPreOrderTraversal()
  
      
  void TestErfLikeFunction() {
    
    Setup();
    
    // Test values from MATLAB
    TEST_DOUBLE_APPROX(0.3957, integrals_->ErfLikeFunction_(5.0), 0.001);
    TEST_DOUBLE_APPROX(0.8556, integrals_->ErfLikeFunction_(0.5), 0.001);

    Destruct();
    
    NONFATAL("TestErfLikeFunction passed.\n");
    
  } // TestErfLikeFunction
  
  void TestComputeSingleIntegralDists() {
    
    Setup();
    
    NONFATAL("Do single integral MATLAB tests!\n");
    
    // I should also make sure that the integral has the right symmetries
    
    Destruct();
    
  } // TestComputeSingleIntegralDists
  
  void TestComputeSingleIntegral() {
    
    Setup();
    
    NONFATAL("Do single integral MATLAB tests!\n");
    
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
  
  void TestComputeFockMatrix() {
    
    Setup();
    
    NONFATAL("TestComputeTwoElectronIntegrals not yet written!\n");
    
    Destruct();
    
  } // TestComputeTwoElectronIntegrals
  
  void TestAll() {
    
    TestPreOrderTraversal();
    
    TestErfLikeFunction();
    
    TestComputeSingleIntegralDists();
    
    TestComputeSingleIntegral();
    
    TestCanApproximate();
    
    TestComputeIntegralsBaseCase();
    
    TestComputeIntegralsRecursion();
    
    TestComputeFockMatrix();
    
    NONFATAL("Not all tests have been written!");
    
  } // TestAll
  
private:
  
  DualTreeIntegrals* integrals_;
  
}; //class DualTreeIntegralsTest


int main(int argc, char* argv[]) {
  
  fx_init(argc, argv, NULL);
  
  DualTreeIntegralsTest tester;
  tester.TestAll();
  
  fx_done(NULL);
  
  return 0;
  
} // main
