/**
 * @file hf_test.cc
 * 
 * @author Bill March (march@gatech.edu)
 *
 * Unit tests for the Hartree-Fock code.  
 */

#include "fastlib/base/test.h"
#include "hf.h"

/**
 * Unit test class for Hartree-Fock code.
 */
class HartreeFockTest {
  
  // Use the example from Leach's book
  
public:
  
  
  static const index_t num_electrons = 2;
  
  void Init() {
    
    solver_ = new HFSolver();
  
    Matrix overlap;
    Matrix kinetic;
    Matrix potential;
    Matrix two_electron;
    
    Matrix core;
    
    double nuclear_energy = 0.0;
    
    data::Load("overlap_test.csv", &overlap);
    data::Load("kinetic_test.csv", &kinetic);
    data::Load("potential_test.csv", &potential);
    //data::Load("two_electron_test.csv", &two_electron);
    two_electron.Init(2, 2);
    
    data::Load("core_test.csv", &core);
    
    solver_->Init(nuclear_energy, overlap, kinetic, potential, two_electron, 
                 num_electrons);
    
    for (index_t i = 0; i < core.n_rows(); i++) {
      for (index_t j = 0; j < core.n_cols(); j++) {
        TEST_DOUBLE_APPROX(core.ref(i, j), 
                           solver_->one_electron_integrals_.ref(i, j), 0.0001);
      }
    }
    
    
  }
  
  void Destruct() {
   
    delete solver_;
    
  }
  
  void TestOrthogonalizingMatrix() {
    Init();
    
    solver_->FormOrthogonalizingMatrix();
    
    Matrix true_orthogonal;
    data::Load("orthogonalizing_test.csv", &true_orthogonal);
    
    for (index_t i = 0; i < true_orthogonal.n_rows(); i++) {
      for (index_t j = 0; j < true_orthogonal.n_cols(); j++) {
        TEST_DOUBLE_APPROX(true_orthogonal.ref(i, j), 
                           solver_->overlap_matrix_.ref(i, j), 0.0001); 
      }
    }
    
    Destruct();
    
    NONFATAL("Orthogonal matrix correct.\n");
  }
  
  void TestAll() {
   
    TestOrthogonalizingMatrix();  
    
    NONFATAL("All tests passed\n");
    
  }
  
private:
  
  HFSolver* solver_;
  
};
#if 0
class Bob {
public:
  static const double x = .3;
  double y;
  Bob() {
    y = .3;
  }
  double foo() {
    return y += x;
  }
};
#endif

int main(int argc, char *argv[]) {

 
  
  HartreeFockTest tester;
  tester.TestAll();
  
  return 0;
  
}