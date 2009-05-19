/**
 * @file scf_solver.h
 * 
 * @author Bill March (march@gatech.edu)
 *
 * Contains an algorithm class for the SCF solution for Hartree-Fock.  
 */

#ifndef SCF_SOLVER_H
#define SCF_SOLVER_H

#include "fastlib/fastlib.h"
#include "contrib/march/fock_matrix/fock_impl/eri.h"
#include "contrib/march/fock_matrix/fock_impl/oeints.h"


/**
 * Algorithm class for the SCF part of the HF computation.  This class assumes 
 * the integrals have been computed and does the SVD-like part of the 
 * computation.
 */
template<class CoulombAlg, class ExchangeAlg>
class SCFSolver {
  
 private:
  
  CoulombAlg* coulomb_alg_;
  ExchangeAlg* exchange_alg_;
  
  bool single_fock_alg_;
  
  // Columns are the coordinates of centers of basis functions
  Matrix basis_centers_;
  // Centers of the nuclei
  Matrix nuclear_centers_;  
  
  Vector exponents_;
  
  Vector momenta_;
  
  // Charge of nuclei, needs to be renamed
  Vector nuclear_masses_;
  
  Matrix core_matrix_; // T + V
  Matrix kinetic_energy_integrals_; // T
  Matrix potential_energy_integrals_; // V
  
  Matrix coefficient_matrix_; // C or C'
  
  Matrix overlap_matrix_; // S 
  Matrix change_of_basis_matrix_; // S^{-1/2} 
  Matrix density_matrix_; // D
  
  Matrix coulomb_mat_;
  Matrix exchange_mat_;
  
  Matrix fock_matrix_; // F or F', depending on the basis
  
  Vector energy_vector_; // The diagonal matrix of eigenvalues of F/F'
  
  index_t number_of_basis_functions_; // N
  index_t number_of_electrons_; // K
  index_t number_of_nuclei_;
  // Number of orbitals to fill, i.e. K/2 
  index_t number_to_fill_; 
  
  double nuclear_repulsion_energy_;
  
  double one_electron_energy_;
  double two_electron_energy_;
  
  // The total energy in each iteration
  ArrayList<double> total_energy_;
  
  index_t current_iteration_;
  
  // The density matrix error norms for use in DIIS
  Matrix density_matrix_norms_;
  
  // The frobenius norm of the density matrix after each iteration
  ArrayList<double> iteration_density_norms_;
  
  // Past density matrices and matrix errors for DIIS
  ArrayList<Matrix> density_matrices_;
  ArrayList<Matrix> density_matrix_errors_;
  
  // The total number of matrices to store for DIIS
  index_t diis_count_;
  // The current position in the DIIS arrays
  index_t diis_index_;
  
  // The right hand side of the linear system for the DIIS solution
  Vector diis_rhs_;
  
  Vector basis_energies_;
  
  // Initial size of the density norm and total energy arrays
  //static const index_t expected_number_of_iterations_ = 10;
  
  // Convergence tolerances
  double density_convergence_;
  double energy_convergence_;
  
  // The norm of the difference between this density matrix and the previous one
  double density_matrix_frobenius_norm_;
  
  struct datanode* module_;
  
  ArrayList<index_t> occupied_indices_;
  
  
 public:
    
  SCFSolver() {}
  
  ~SCFSolver() {}
  

  void Init(const Matrix& basis_centers, const Matrix& exp, 
            const Matrix& mom, const Matrix& density, fx_module* mod, 
            const Matrix& nuclear_cent, const Matrix& nuclear_mass, 
            CoulombAlg* coul_alg, ExchangeAlg* exc_alg, index_t num_electrons) {
    
    module_ = mod;
    number_of_electrons_ = num_electrons;
    
    struct datanode* integral_mod = fx_submodule(module_, "integrals");
                                                
    struct datanode* naive_mod = fx_submodule(module_, "naive_integrals");
        
    // Set to 1 to perform no diis iterations
    diis_count_ = fx_param_int(NULL, "diis_states", 1);
    diis_index_ = 0;
    
    density_matrices_.Init(diis_count_);
    density_matrix_errors_.Init(diis_count_);
    density_matrix_norms_.Init(diis_count_ + 1, diis_count_ + 1);
    density_matrix_norms_.SetZero();
    
    diis_rhs_.Init(diis_count_ + 1);
    diis_rhs_.SetZero();
    diis_rhs_[diis_count_] = -1;
    
    nuclear_centers_.Copy(nuclear_cent);
    
    nuclear_masses_.Copy(nuclear_mass.ptr(), nuclear_centers_.n_cols());
    
    exponents_.Copy(exp.ptr(), basis_centers_.n_cols());
    momenta_.Copy(mom.ptr(), basis_centers_.n_cols());
    
    number_of_nuclei_ = nuclear_centers_.n_cols();
    
    number_to_fill_ = (index_t)ceil((double)number_of_electrons_/2);
    occupied_indices_.Init(number_to_fill_);
    
    DEBUG_ASSERT(number_of_nuclei_ == nuclear_masses_.length());
    
    
    number_of_basis_functions_ = basis_centers_.n_cols();
    
    for (index_t i = 0; i < diis_count_; i++) {
      
      density_matrices_[i].Init(number_of_basis_functions_, 
                                number_of_basis_functions_);
                                
      density_matrices_[i].SetZero();
      
      density_matrix_errors_[i].Init(number_of_basis_functions_, 
                                     number_of_basis_functions_);
      density_matrix_errors_[i].SetZero();
      
      density_matrix_norms_.set(diis_count_, i, -1);
      density_matrix_norms_.set(i, diis_count_, -1);
      
    }
    
    
    DEBUG_ASSERT(number_of_basis_functions_ >= number_to_fill_);
    
    // Empty inits to prevent errors on closing
    overlap_matrix_.Init(number_of_basis_functions_, 
                         number_of_basis_functions_);
    kinetic_energy_integrals_.Init(number_of_basis_functions_, 
                                   number_of_basis_functions_);
    potential_energy_integrals_.Init(number_of_basis_functions_, 
                                     number_of_basis_functions_);
    
    coefficient_matrix_.Init(number_of_basis_functions_, 
                             number_of_basis_functions_);
    
    energy_vector_.Init(number_of_basis_functions_);
    
    //total_energy_.Init(expected_number_of_iterations_);
    total_energy_.Init();
    
    //iteration_density_norms_.Init(expected_number_of_iterations_);
    iteration_density_norms_.Init();
    
    density_convergence_ = fx_param_double(module_, "density_convergence", 0.1);
    energy_convergence_ = fx_param_double(module_, "energy_convergence", 0.1);
    
    // Need to double check that this is right
    density_matrix_frobenius_norm_ = DBL_MAX;
    
    current_iteration_ = 0;
    
    basis_energies_.Init(number_of_basis_functions_);
    basis_energies_.SetZero();
    
    coulomb_alg_ = coul_alg;
    exchange_alg_ = exc_alg;
    
    single_fock_alg_ = (coulomb_alg_ == exchange_alg_);

  } // Init()
  
 private:
    
  /**
   * Permutes the matrix mat according to the permutation given.  The permuted 
   * matrix is written to the uninitialized matrix new_mat
   */                               
  void PermuteMatrix_(const Matrix& old_mat, Matrix* new_mat, 
                      const ArrayList<index_t>& perm) {
  
    index_t num_cols = old_mat.n_cols();
    DEBUG_ASSERT(num_cols == perm.size());
    
    new_mat->Init(old_mat.n_rows(), num_cols);
    
    for (index_t i = 0; i < num_cols; i++) {
      
      Vector old_vec;
      old_mat.MakeColumnVector(perm[i], &old_vec);
      Vector new_vec;
      new_mat->MakeColumnVector(i, &new_vec);
      
      new_vec.CopyValues(old_vec);
      
    }
  
  } // PermuteMatrix_()
  
  /** 
   * Given the basis set and nuclear coordinates, compute and store the one 
   * electron matrices. 
   *
   * For now, just using loops.  In the future, it's an N-body problem but 
   * probably a very small fraction of the total running time.  
   */
  void ComputeOneElectronMatrices_();
 
 
  /**
   * Create the matrix S^{-1/2} using the eigenvector decomposition.     *
   */
  void FormChangeOfBasisMatrix_();
  
  
  /**
   * Compute the density matrix.
   * 
   * TODO: Consider an SVD or some eigenvalue solver that will find the 
   * eigenvalues in ascending order.
   */
  void ComputeDensityMatrix_();
  
  /**
   * Pulay's DIIS method, as described by David
   *
   * Need to check convergence and write a function for solving the linear 
   * system.
   *
   * Don't forget to fill in the Init fn. and the linear system solver
   */ 
  void ComputeDensityMatrixDIIS_();
  
  /**
   * Given that the array density_matrices_ and the matrix density_matrix_norms_
   * are full, this performs the DIIS step to get the best linear combination of 
   * the matrices in density_matrices_ and puts it in density_matrix_
   */
  void DIISSolver_();  
  
  /**
    * Given that the Fock matrix has been transformed to the orthonormal basis
   * (F'), this function determines the energy_vector e and the transformed 
   * coefficient matrix C'.  It then untransforms the matrix to get C.
   */
  void DiagonalizeFockMatrix_();  
  
  /**
   * Determine the K/2 lowest energy orbitals.  
   * 
   * TODO: If K is odd, then the last entry here is the orbital that should 
   * have one electron.  I think the closed-shell RHF formulation I'm using
   * forbids an odd number of electons.
   *
   * I should use the built in C++ iterator-driven routines.  Ryan suggested
   * that I write iterators for ArrayLists, since that would open up a lot of
   * functionality, including sorting. 
   *
   * Now that I'm using SVD, the eigenvalues are in order up to signs.  So, I 
   * should be able to use that info to make this code more efficient.
   *
   * Maybe make two passes: 1 for negative e-vals, one for positive.  They're
   * both sorted.  
   */
  void FillOrbitals_() {
    
    double max_energy_kept = -DBL_INF;
    index_t next_to_go = 0;
    
    for (index_t i = 0; i < number_to_fill_; i++) {
      
      occupied_indices_[i] = i;
      if (energy_vector_[i] > max_energy_kept) {
        max_energy_kept = energy_vector_[i];
        next_to_go = i;
      }
      
    }
    
    
    for (index_t i = number_to_fill_; i < number_of_basis_functions_; i++) {    
      
      double this_energy = energy_vector_[i];
      if (this_energy < max_energy_kept) {
        occupied_indices_[next_to_go] = i;
        
        // Find the new index to throw out
        double new_max = -DBL_INF;
        next_to_go = -1;
        for (index_t j = 0; j < number_to_fill_; j++) {
          if (energy_vector_[occupied_indices_[j]] > new_max) {
            new_max = energy_vector_[occupied_indices_[j]];
            next_to_go = j;
          }
        }
        max_energy_kept = new_max;
        DEBUG_ASSERT(!isinf(max_energy_kept));
        DEBUG_ASSERT(next_to_go >= 0);
        DEBUG_ASSERT(next_to_go < number_to_fill_);
      }
      
    }
    
  } //FillOrbitals_
  
  
    
  /**
   * Find the energy of the electrons in the ground state of the current 
   * wavefunction.  
   *
   * The sum needs to be over occupied orbitals, according to Szabo
   * I'm not sure about this, though, since the notation keeps changing
   */
  double ComputeElectronicEnergy_() {
    
    
    double total_energy = 0.0;
    one_electron_energy_ = 0.0;
    two_electron_energy_ = 0.0;
    
    for (index_t i = 0; i < number_of_basis_functions_; i++) {
      
      // for the diagonal entries
      one_electron_energy_ += density_matrix_.ref(i,i) * core_matrix_.ref(i,i);
      two_electron_energy_ += density_matrix_.ref(i,i) * 
                              (fock_matrix_.ref(i,i) - core_matrix_.ref(i,i));
                              
      double current_energy = density_matrix_.ref(i,i) * 
          (core_matrix_.ref(i,i) + fock_matrix_.ref(i,i));
      
      total_energy += current_energy;
          
      basis_energies_[i] = current_energy;
            
      for (index_t j = i+1; j < number_of_basis_functions_; j++) {
      
        // multiply by 2 to get the lower triangle
        one_electron_energy_ += 2 * density_matrix_.ref(i,j) * 
                                core_matrix_.ref(i,j);
        two_electron_energy_ += 2 * density_matrix_.ref(i,j) * 
                                (fock_matrix_.ref(i,j) - core_matrix_.ref(i,j));
        
        double this_energy = 2 * density_matrix_.ref(i, j) * 
            (core_matrix_.ref(i, j) + fock_matrix_.ref(i, j));
            
        total_energy = total_energy + this_energy;
        
        basis_energies_[i] += this_energy;

      } // j
    
    } // i
    
    // Leach says there is a factor of 1/2
    total_energy = (0.5 * total_energy) + nuclear_repulsion_energy_;
    
    // No factor of two because there's no overcounting of electrons
    //one_electron_energy_ = one_electron_energy_;
    two_electron_energy_ = 0.5 * two_electron_energy_;
    
    /*
    printf("one_electron_energy: %g\n", one_electron_energy_);
    printf("two_electron_energy: %g\n", two_electron_energy_);
    */
    
    return total_energy;
    
    
  } // ComputeElectronicEnergy_
  
  /**
    * Determine if the density matrix and total energy have converged.   
   */
  bool TestConvergence_() {
    
    if (unlikely(current_iteration_ < 2)) {
      return false;
    }
    
    double energy_diff = fabs(total_energy_[current_iteration_] - 
                              total_energy_[(current_iteration_ - 1)]);
    
   
    printf("\n");
    printf("current_iteration: %d\n", current_iteration_);
    
    printf("density_matrix_frobenius_norm_: %g\n", 
           density_matrix_frobenius_norm_);
           
    printf("total_energy_[%d]: %g\n", current_iteration_, 
           total_energy_[current_iteration_]);
           
    printf("energy_diff: %g\n", energy_diff);
    
    if (likely(density_matrix_frobenius_norm_ > density_convergence_)) {
      return false;
    }
    
    if (likely(energy_diff > energy_convergence_)) {
      return false;
    }
    
    return true;
    
  } // TestConvergence_
  
  /**
   * Pre- and post-multiply the Fock matrix by the change of basis matrix 
   */
  void TransformFockBasis_() {
    
    Matrix orthogonal_transpose_times_fock;
    la::MulTransAInit(change_of_basis_matrix_, fock_matrix_, 
                           &orthogonal_transpose_times_fock);
    la::MulOverwrite(orthogonal_transpose_times_fock, 
                     change_of_basis_matrix_, &fock_matrix_);
    
  } // TransformFockBasis_
  
 /**
  * Tell the coulomb and exchange algorithms to compute a new Fock matrix after
  * giving them the updated density.
  */
  void UpdateFockMatrix_() {

    // give new density to fock matrix algs
    
    coulomb_alg_.UpdateDensity(density_matrix_);
    
    if (!single_fock_alg_) {
      exchange_alg_.UpdateDensity(density_matrix_);
    }
    
    // compute coulomb
    
    coulomb_alg_.Compute();
    coulomb_alg_.OutputCoulomb(&coulomb_mat_);
    
    // if necessary, compute exchange
    if (single_fock_alg_) {
      
      coulomb_alg_.OutputExchange(&exchange_mat_);
      
    }
    else { 
      
      exchange_alg_.Compute();
      exchange_alg_.OutputExchange(&exchange_mat_);
      
    }
    
    // output results and fill in Fock matrix

    //fock_matrix_ = core_matrix_ + coulomb_mat_ - exchange_mat_;
    la::AddOverwrite(core_matrix_, coulomb_mat_, &fock_matrix_);
    la::SubFrom(exchange_mat_, &fock_matrix_);
    
  } // UpdateFockMatrix_()
  
  /**
    * Does the SCF iterations to find the HF wavefunction
   */
  void FindSCFSolution_() {
    
    bool converged = false;
    
    /*
    printf("Matrix permutation:\n");
    ot::Print(old_from_new_centers_);
    */
    
    while (!converged) {
      
      // Step 4a.
      UpdateFockMatrix_();
      
      // Step 4b.
      // replace this with add back
      /*
      if (unlikely(current_iteration_ >= total_energy_.size())) {
        total_energy_.EnsureSizeAtLeast(2*total_energy_.size());
        iteration_density_norms_.EnsureSizeAtLeast(
            2*iteration_density_norms_.size());
      }
      */
      
      //total_energy_[current_iteration_] = ComputeElectronicEnergy_();
      total_energy_.AddBack(ComputeElectronicEnergy_());
      iteration_density_norms_.AddBack();
      
      // Step 4c.
      TransformFockBasis_();
      
      // Step 4d/e.
      DiagonalizeFockMatrix_();
      
      //Step 4f.
      ComputeDensityMatrixDIIS_();
      
      density_matrix_.PrintDebug("Density Matrix");
      
      // Step 4g.
      converged = TestConvergence_();
      
      current_iteration_++;
      
    } // end while
    
    current_iteration_--;
    
    //total_energy_[current_iteration_] = ComputeElectronicEnergy_();
    
  } // FindSCFSolution_
  
  /**
   * Returns the nuclear repulsion energy for the nuclei given in 
   * nuclear_centers_ and nuclear_masses_
   *
   * I'm only counting each pair once, which I think is correct.  
   */
  double ComputeNuclearRepulsion_();
  
  /**
   * Sets up the matrices for the SCF iterations 
   */
  void Setup_() {
    
    nuclear_repulsion_energy_ = ComputeNuclearRepulsion_();
    
    ComputeOneElectronMatrices_();
    
    FormChangeOfBasisMatrix_();
    
    fock_matrix_.Copy(core_matrix_);
    
    TransformFockBasis_();
    
    DiagonalizeFockMatrix_();
    
    ComputeDensityMatrixDIIS_();
    
  } //Setup_

  /**
   * Save the coefficient matrix, total energy, and energy vector to files.
   *
   * TODO: think about a better way to separate the filled and virtual orbitals
   */
  void OutputResults_() {
    
    const char* coefficients_file = 
        fx_param_str(module_, "C", "coefficients.csv");
    data::Save(coefficients_file, coefficient_matrix_);
    
    const char* energy_file = fx_param_str(module_, "Etot", "total_energy.csv");
    FILE* energy_pointer = fopen(energy_file, "w");
    for (index_t i = 0; i < current_iteration_; i++) {
     
      fprintf(energy_pointer, "Iteration %d:\t %f\n", i, total_energy_[i]);
      
    }
    fclose(energy_pointer);
    
    const char* energy_vector_file = 
        fx_param_str(module_, "Evec", "energy_vector.csv");
    Matrix energy_vector_matrix;
    energy_vector_matrix.AliasColVector(energy_vector_);
    data::Save(energy_vector_file, energy_vector_matrix);
    
    const char* basis_vector_file = fx_param_str(module_, "basis_energy", 
                                                 "basis_energies.csv");
    
    Matrix basis_energy_matrix;
    basis_energy_matrix.AliasColVector(basis_energies_);
    data::Save(basis_vector_file, basis_energy_matrix);
    
    fx_format_result(module_, "density_matrix_norm", "%g", 
                     density_matrix_frobenius_norm_);
    
    /*
    fx_format_result(module_, "fock_max", "%g", fock_max_);
    fx_format_result(module_, "fock_min", "%g", fock_min_);
    fx_format_result(module_, "density_max", "%g", density_max_);
    fx_format_result(module_, "density_min", "%g", density_min_);
    */
    
    fx_format_result(module_, "nuclear_repulsion", "%g", 
                     nuclear_repulsion_energy_);
                     
    fx_format_result(module_, "one_electron_energy", "%g", 
                     one_electron_energy_);
    
    fx_format_result(module_, "two_electron_energy", "%g", 
                     two_electron_energy_);
    
    fx_format_result(module_, "num_iterations", "%d", current_iteration_);
    
    fx_format_result(module_, "total_energy", "%g", 
                     total_energy_[current_iteration_]);
    
  }
  
 public:
  
  /**
   * Compute the Hartree-Fock wavefunction for the given values of the 
   * integrals. 
   */
  void ComputeWavefunction() {
      
    fx_timer_start(module_, "SCF_Setup");
    Setup_();
    fx_timer_stop(module_, "SCF_Setup");
      
    fx_timer_start(module_, "SCF_Iterations");
    FindSCFSolution_();
    fx_timer_stop(module_, "SCF_Iterations");
    
    OutputResults_();
      
  } // ComputeWavefunction
  
  void PrintMatrices() {
    
    // These should be changed to print debug or something
    
    core_matrix_.PrintDebug("Core Matrix");
    
    coefficient_matrix_.PrintDebug("Coefficient matrix");
    
    overlap_matrix_.PrintDebug("Change-of-basis matrix:\n");
    
    density_matrix_.PrintDebug("Density matrix");
    
    fock_matrix_.PrintDebug("Fock matrix");
    
    energy_vector_.PrintDebug("Energy vector");
    
    printf("Total energy:\n");
    ot::Print(total_energy_);  
      
  } // PrintMatrices
  
}; // class HFSolver


#endif // inclusion guards
