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
    
    for (index_t i = 0; i < true_change_basis.n_rows(); i++) {
      for (index_t j = 0; j < true_change_basis.n_cols(); j++) {
        TEST_DOUBLE_APPROX(true_change_basis.ref(i, j), 
                           solver_->change_of_basis_matrix_.ref(i, j), eps); 
      }
    }
    
    
    Destruct();
    
    NONFATAL("Overlap and Change of Basis matrices correct.\n");
    
  }
  
  void TestDensityMatrix() {
    
    Setup();
    
    Matrix true_coeffs;
    data::Load("coefficient_test.csv", &true_coeffs);
    solver_->coefficient_matrix_.Destruct();
    solver_->coefficient_matrix_.Copy(true_coeffs);
    
    solver_->nuclear_repulsion_energy_ = 0.0;
    
    solver_->number_of_basis_functions_ = true_coeffs.n_cols();
    
    solver_->number_to_fill_ = 1;
    solver_->occupied_indices_.Destruct();
    solver_->occupied_indices_.Init(1);    
    
    solver_->energy_vector_.Destruct();
    solver_->energy_vector_.Init(solver_->number_of_basis_functions_);
    solver_->energy_vector_[0] = -2.458;
    solver_->energy_vector_[1] = -1.292;
    
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
    
    NONFATAL("ComputeDensityMatrix correct.\n");
    
    Destruct();
    
  } // TestDensityMatrix
  
  void TestFillOrbitals() {
    
    Setup();
    
    solver_->number_of_electrons_ = 6;
    solver_->number_to_fill_ = 3;
    solver_->number_of_basis_functions_ = 10;
    
    solver_->occupied_indices_.Destruct();
    solver_->occupied_indices_.Init(3);
    
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
    
    
    NONFATAL("FillOrbitals_ correct.\n");

    
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
    solver_->density_matrix_.Destruct();
    solver_->density_matrix_.Copy(true_density);
    
    Matrix true_core;
    data::Load("core_test.csv", &true_core);
    solver_->core_matrix_.Destruct();
    solver_->core_matrix_.Copy(true_core);
    
    solver_->nuclear_repulsion_energy_ = 0.0;
    
    Matrix true_fock;
    data::Load("updated_fock_test.csv", &true_fock);
    solver_->fock_matrix_.Destruct();
    solver_->fock_matrix_.Copy(true_fock);
    solver_->number_of_basis_functions_ = true_fock.n_cols();
    
    double test_energy = solver_->ComputeElectronicEnergy_();
    
    TEST_DOUBLE_APPROX(true_energy, test_energy, eps);
    
    Destruct();
    
    NONFATAL("TestComputeElectronicEnergy correct.\n");
    
  } // TestComputeElectronicEnergy
  
  void TestPermuteMatrix() {
  
    Setup();
    
    Matrix permute_me;
    permute_me.Init(2, 5);
    permute_me.set(0, 0, 0);
    permute_me.set(1, 0, 0);
    
    permute_me.set(0, 1, 1);
    permute_me.set(1, 1, 1);
    
    permute_me.set(0, 2, 4);
    permute_me.set(1, 2, 4);
    
    permute_me.set(0, 3, 3);
    permute_me.set(1, 3, 3);
    
    permute_me.set(0, 4, 2);
    permute_me.set(1, 4, 2);
    
    ArrayList<index_t> perm;
    perm.Init(5);
    perm[0] = 0;
    perm[1] = 1;
    perm[2] = 4;
    perm[3] = 3;
    perm[4] = 2;
    
    
    Matrix unpermuted;
    
    solver_->PermuteMatrix_(permute_me, &unpermuted, perm);
    
    
    
    Destruct();
    
    NONFATAL("TestPermuteMatrix correct.\n");
  
  } // TestPermuteMatrix()
  
  void TestComputeKineticIntegral() {
  
    NONFATAL("TestComputeKineticIntegral NOT IMPLEMENTED.\n");
  
  } // TestComputeKineticIntegral()
  
  void TestComputeNuclearIntegral() {
  
    NONFATAL("TestComputeNuclearIntegral NOT IMPLEMENTED.\n");
  
  } // TestComputeNuclearIntegral()
  
  void TestComputeNuclearRepulsion() {
    
    NONFATAL("TestComputeNuclearRepulsion NOT IMPLEMENTED.\n");
  
  } // TestComputeNuclearRepulsion()
  
  void TestComputeOneElectronMatrices() {
  
    Setup();
    
    Matrix true_kinetic;
    data::Load("test_kinetic.csv", &true_kinetic);
    
    for (index_t i = 0; i < true_kinetic.n_rows(); i++) {
      for (index_t j = 0; j < true_kinetic.n_cols(); j++) {
        TEST_DOUBLE_APPROX(true_kinetic.ref(i, j), 
                           solver_->kinetic_energy_integrals_.ref(i, j), eps); 
      }
    }
    
    Matrix true_nuclear;
    data::Load("test_nuclear_integrals.csv", &true_nuclear);
    
    for (index_t i = 0; i < true_nuclear.n_rows(); i++) {
      for (index_t j = 0; j < true_nuclear.n_cols(); j++) {
        TEST_DOUBLE_APPROX(true_nuclear.ref(i, j), 
                           solver_->potential_energy_integrals_.ref(i, j), eps); 
      }
    }
    
    Destruct();
    
    NONFATAL("TestComputeOneElectronMatrices correct.\n");
  
  } // TestComputeOneElectronMatrices()
  
  
  
  void TestAll() {
   
    TestChangeOfBasisMatrix();  
    
    TestDensityMatrix();
    
    TestFillOrbitals();
    
    TestComputeOneElectronMatrices();
    
    // TestDiagonalizeFockMatrix();
    
    // TestTestConvergence();
    
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

  fx_init(argc, argv);

  SCFSolverTest tester;
  tester.TestAll();
  
  fx_done();
  
  return 0;
  
}