#include "dual_tree_integrals.h"
#include <fastlib/base/test.h>

class DualTreeIntegralsTest {
  
public:
  
  void Setup() {
    
    Matrix test_centers;
    data::Load("test_centers.csv", &test_centers);
    
    double test_bandwidth = 0.01;
    
    integrals_->Init(test_centers, NULL, test_bandwidth);
    
  } // Setup
  
  
  void Destruct() {
    
    delete integrals_;
    
  } // Destruct
  
  void TestAll() {
    
  } // TestAll
  
private:
  
  DualTreeIntegrals* integrals_;
  
}; //class DualTreeIntegralsTest


int main(int argc, char* argv[]) {
  
  DualTreeIntegralsTest tester;
  tester.TestAll();
  
  return 0;
  
} // main