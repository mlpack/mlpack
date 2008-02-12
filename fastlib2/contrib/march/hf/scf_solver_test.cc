/**
 * @file scf_solver_test.cc
 * 
 * @author Bill March (march@gatech.edu)
 *
 * Unit tests for the SCF solver code.  
 */

#include "fastlib/base/test.h"
#include "scf_solver.h"


/**
 * Unit test class for Hartree-Fock code.
 */
class SCFSolverTest {
  
  // Use the example from Leach's book
  
public:
  
  
  static const index_t num_electrons = 2;
  static const double eps = 0.01;
  
  void Setup() {
    
    solver_ = new SCFSolver();
  
    Matrix overlap;
    Matrix kinetic;
    Matrix potential;
    Vector two_electron;
    
    Matrix core;
    
    double nuclear_energy = 0.0;
    double convergence_tolerance = 1e-5;
    
    data::Load("overlap_test.csv", &overlap);
    data::Load("kinetic_test.csv", &kinetic);
    data::Load("potential_test.csv", &potential);
    
    Matrix two_electron_mat;
    data::Load("two_electron_test.csv", &two_electron_mat);
    DEBUG_ASSERT(two_electron_mat.n_cols() == 1);
    two_electron_mat.MakeColumnVector(0, &two_electron);
    
    data::Load("core_test.csv", &core);
    
    solver_->Init(nuclear_energy, overlap, kinetic, potential, two_electron, 
                 num_electrons, convergence_tolerance, NULL);
    
    for (index_t i = 0; i < core.n_rows(); i++) {
      for (index_t j = 0; j < core.n_cols(); j++) {
        TEST_DOUBLE_APPROX(core.ref(i, j), 
                           solver_->one_electron_integrals_.ref(i, j), eps);
      }
    }
    
    
  }
  
  void Destruct() {
   
    delete solver_;
    
  }
  
  void TestOrthogonalizingMatrix() {
    Setup();
    
    solver_->FormOrthogonalizingMatrix_();
    
    Matrix true_orthogonal;
    data::Load("orthogonalizing_test.csv", &true_orthogonal);
    
    // Is the change-of-basis matrix unique with respect to negaives on the 
    // diagonal?  If not, I need to add absolute values here.
    for (index_t i = 0; i < true_orthogonal.n_rows(); i++) {
      for (index_t j = 0; j < true_orthogonal.n_cols(); j++) {
        TEST_DOUBLE_APPROX(true_orthogonal.ref(i, j), 
                           solver_->overlap_matrix_.ref(i, j), eps); 
      }
    }
    
    Destruct();
    
    NONFATAL("Orthogonal matrix correct.\n");
  }
  
  void TestDensityMatrix() {
    
    Setup();
    
    solver_->FormOrthogonalizingMatrix_();
    
    solver_->ComputeDensityMatrix_();
    
    // Check to see that it worked for real
    Matrix true_density;
    data::Load("density_test.csv", &true_density);
    
    for (index_t i = 0; i < true_density.n_rows(); i++) {
      for (index_t j = 0; j < true_density.n_cols(); j++) {
        TEST_DOUBLE_APPROX(true_density.ref(i, j), 
                           solver_->density_matrix_.ref(i, j), eps);
      }
    }
    
    NONFATAL("Initial Density Matrix Correct.\n");
    
    Destruct();
    
  } // TestDensityMatrix
  
  void TestFillOrbitals() {
    
    Setup();
    
    solver_->number_of_electrons_ = 5;
    solver_->number_of_basis_functions_ = 10;
    
    Vector test_energy_vector;
    test_energy_vector.Init(10);
    test_energy_vector[0] = -5;
    test_energy_vector[1] = 12;
    test_energy_vector[2] = 1;
    test_energy_vector[3] = 7;
    test_energy_vector[4] = 3;
    test_energy_vector[5] = -9;
    test_energy_vector[6] = 4;
    test_energy_vector[7] = 1;
    test_energy_vector[8] = 0;
    test_energy_vector[9] = 2;
    
    solver_->energy_vector_ = test_energy_vector;
    
    ArrayList<index_t> test_indices;
    solver_->FillOrbitals_(&test_indices);
    
    TEST_ASSERT((test_indices[0] == 0) && (test_indices[1] == 5) 
                && (test_indices[2] == 8));
    
    
    NONFATAL("FillOrbitals_ correct\n");

    
    Destruct();
    
  } // TestFillOrbitals
  
  void TestFindIntegralIndex() {
    
    index_t test1 = solver_->FindIntegralIndexHelper_(1, 0);
    
    index_t test2 = solver_->FindIntegralIndexHelper_(5, 4);
    
    index_t test3 = solver_->FindIntegralIndex_(0, 0, 0, 0);
    
    index_t test4 = solver_->FindIntegralIndex_(1, 1, 0, 0);
    
    index_t test5 = solver_->FindIntegralIndex_(1, 0, 1, 0);
    
    index_t test6 = solver_->FindIntegralIndex_(0, 0, 1, 1);
    
    index_t test7 = solver_->FindIntegralIndexHelper_(4, 5);
    
    TEST_ASSERT(test1 == 1);
    TEST_ASSERT(test2 == 19);
    TEST_ASSERT(test3 == 0);
    TEST_ASSERT(test4 == 3);
    TEST_ASSERT(test5 == 2);
    TEST_ASSERT(test6 == test4);
    TEST_ASSERT(test7 == test2);
    
    NONFATAL("FindIntegralIndex correct.\n");

  } // TestFindIntegralIndex
  
  void TestDiagonalizeFockMatrix() {
    
    NONFATAL("TestDiagonalizeFockMatrix not implemented!\n");
    
  } // TestDiagonalizeFockMatrix

  void TestUpdateFockMatrix() {
    
    Setup();
    
    /*index_t test1 = solver_->FindIntegralIndex_(0, 0, 0, 0);
    printf("0,0,0,0 = %d\n", test1);
    index_t test2 = solver_->FindIntegralIndex_(1, 0, 0, 0);
    printf("1,0,0,0 = %d\n", test2);
    index_t test3 = solver_->FindIntegralIndex_(1, 0, 1, 0);
    printf("1,0,1,0 = %d\n", test3);
    index_t test4 = solver_->FindIntegralIndex_(1, 1, 0, 0);
    printf("1,1,0,0 = %d\n", test4);
    index_t test5 = solver_->FindIntegralIndex_(1, 1, 1, 0);
    printf("1,1,1,0 = %d\n", test5);
    index_t test6 = solver_->FindIntegralIndex_(1, 1, 1, 1);
    printf("1,1,1,1 = %d\n", test6);
    */
    
    Matrix true_density;
    data::Load("density_test.csv", &true_density);
    solver_->density_matrix_.CopyValues(true_density);
    
    solver_->UpdateFockMatrix_();
    
    Matrix true_updated_fock;
    data::Load("updated_fock_test.csv", &true_updated_fock);
    
    printf("true_updated_fock\n");
    ot::Print(true_updated_fock);
    printf("fock_matrix_\n");
    ot::Print(solver_->fock_matrix_);
    
    for (index_t i = 0; i < true_updated_fock.n_rows(); i++) {
      for (index_t j = 0; j < true_updated_fock.n_cols(); j++) {
        TEST_DOUBLE_APPROX(true_updated_fock.ref(i, j), 
                           solver_->fock_matrix_.ref(i, j), eps); 
      }
    }
    
    Destruct();
    
    NONFATAL("TestUpdateFockMatrix correct.\n");
    
  } // TestUpdateFockMatrix
  
  void TestTestConvergence() {
    
    // Not quite sure how to do this one
    // Need to make sure I'm storing the energies correctly and 
    // keeping up with the density matrix norm
    
    NONFATAL("TestTestConvergence not implemented!\n");
    
  } // TestTestConvergence
  
  void TestComputeElectronicEnergy() {
    
    Setup();
    
    double true_energy = -3.87;
    
    Matrix true_density;
    data::Load("density_test.csv", &true_density);
    solver_->density_matrix_.CopyValues(true_density);
    
    Matrix true_fock;
    data::Load("updated_fock_test.csv", &true_fock);
    solver_->density_matrix_.CopyValues(true_fock);
    
    double test_energy = solver_->ComputeElectronicEnergy_();
    
    TEST_DOUBLE_APPROX(true_energy, test_energy, eps);
    
    Destruct();
    
    NONFATAL("TestComputeElectronicEnergy correct.\n");
    
  } // TestComputeElectronicEnergy
  
  
  void TestAll() {
   
    TestOrthogonalizingMatrix();  
    
    TestDensityMatrix();
    
    TestFillOrbitals();
    
    TestFindIntegralIndex();
    
    TestDiagonalizeFockMatrix();
    
    TestUpdateFockMatrix();
    
    TestTestConvergence();
    
    TestComputeElectronicEnergy();
    
    NONFATAL("All tests passed\n");
    
  }
  
private:
  
  SCFSolver* solver_;
  
}; // class SCFSolverTest

// There's some kind of weird bug in the apple loader.  eps won't be recognized
// properly without this line.  
const double SCFSolverTest::eps;


int main(int argc, char *argv[]) {

  SCFSolverTest tester;
  tester.TestAll();
  
  return 0;
  
}