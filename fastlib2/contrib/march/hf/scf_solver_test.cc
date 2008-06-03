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
  static const double eps = 0.001;
  
  void Setup() {
    
    solver_ = new SCFSolver();
  
    Matrix basis_centers;
    data::Load("test_centers.csv", &basis_centers);
    
    Matrix density;
    density.Init(basis_centers.n_cols(), basis_centers.n_cols());
    density.SetZero();
    
    Matrix nuclear;
    data::Load("test_nuclear_centers.csv", &nuclear);
    
    Matrix nuclear_mass_mat;
    data::Load("test_nuclear_masses.csv", &nuclear_mass_mat);
    Vector nuclear_mass;
    nuclear_mass_mat.MakeColumnVector(0, &nuclear_mass);
    
    struct datanode* mod = fx_submodule(NULL, "test_scf", "test_scf");
    
    solver_->Init(mod, 2, basis_centers, density, nuclear, nuclear_mass);
    
    solver_->Setup_();
    
    
  }
  
  void Destruct() {
   
    delete solver_;
    
  }
  
  void TestChangeOfBasisMatrix() {
    Setup();
    
    Matrix true_overlap;
    data::Load("test_overlap.csv", &true_overlap);
    
    for (index_t i = 0; i < true_overlap.n_rows(); i++) {
      for (index_t j = 0; j < true_overlap.n_cols(); j++) {
        TEST_DOUBLE_APPROX(true_overlap.ref(i, j), 
                           solver_->overlap_matrix_.ref(i, j), eps); 
      }
    }
    
    Matrix true_change_basis;
    data::Load("test_change_basis.csv", &true_change_basis);
    
    // Is the change-of-basis matrix unique with respect to negatives on the 
    // diagonal?  If not, I need to add absolute values here.
    for (index_t i = 0; i < true_change_basis.n_rows(); i++) {
      for (index_t j = 0; j < true_change_basis.n_cols(); j++) {
        TEST_DOUBLE_APPROX(true_change_basis.ref(i, j), 
                           solver_->change_of_basis_matrix_.ref(i, j), eps); 
      }
    }
    
    
    Destruct();
    
    NONFATAL("Orthogonal matrix correct.\n");
  }
  
  void TestDensityMatrix() {
    
    Setup();
    
    solver_->FormChangeOfBasisMatrix_();
    
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
    
    solver_->FillOrbitals_();
    
    TEST_ASSERT((solver_->occupied_indices_[0] == 0) 
                && (solver_->occupied_indices_[1] == 5) 
                && (solver_->occupied_indices_[2] == 8));
    
    
    NONFATAL("FillOrbitals_ correct\n");

    
    Destruct();
    
  } // TestFillOrbitals
  
    
  void TestDiagonalizeFockMatrix() {
    
    NONFATAL("TestDiagonalizeFockMatrix not implemented!\n");
    
  } // TestDiagonalizeFockMatrix


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
  
  void TestComputeOverlapIntegral() {
  
    Setup();
    
    double dist1 = 0.5;
    
    double test_integral = solver_->ComputeOverlapIntegral_(dist1);
    
    double correct_integral = 0;
    
    Destruct();
    
    NONFATAL("TestComputeOverlapIntegral NOT IMPLEMENTED.\n");
  
  } // TestComputeOverlapIntegral()
  
  void TestComputeKineticIntegral() {
  
    NONFATAL("TestComputeKineticIntegral NOT IMPLEMENTED.\n");
  
  } // TestComputeKineticIntegral()
  
  void TestComputeNuclearIntegral() {
  
    NONFATAL("TestComputeNuclearIntegral NOT IMPLEMENTED.\n");
  
  } // TestComputeNuclearIntegral()
  
  void TestComputeOneElectronMatrices() {
  
    NONFATAL("TestComputeOneElectronMatrices NOT IMPLEMENTED.\n");
  
  } // TestComputeOneElectronMatrices()
  
  void TestComputeNuclearRepulsion() {
    
    NONFATAL("TestComputeNuclearRepulsion NOT IMPLEMENTED.\n");
  
  } // TestComputeNuclearRepulsion()
  
  
  void TestAll() {
   
    TestChangeOfBasisMatrix();  
    
    //TestDensityMatrix();
    
    // TestFillOrbitals();
    
    /*
    TestDiagonalizeFockMatrix();
    
    TestTestConvergence();
    
    TestComputeElectronicEnergy();
    */
    NONFATAL("All tests passed\n");
    
  }
  
private:
  
  SCFSolver* solver_;
  
}; // class SCFSolverTest

// There's some kind of weird bug in the apple loader.  eps won't be recognized
// properly without this line.  
const double SCFSolverTest::eps;


int main(int argc, char *argv[]) {

  fx_init(argc, argv);

  SCFSolverTest tester;
  tester.TestAll();
  
  fx_done();
  
  return 0;
  
}