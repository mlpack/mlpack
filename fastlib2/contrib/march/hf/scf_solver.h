/**
 * @file scf_solver.h
 * 
 * @author Bill March (march@gatech.edu)
 *
 * Contains an algorithm class for the SCF solution for Hartree-Fock.  
 */

#ifndef SCF_SOLVER_H
#define SCF_SOLVER_H

#include <fastlib/fastlib.h>
#include "dual_tree_integrals.h"

/**
 * Algorithm class for the SCF part of the HF computation.  This class assumes 
 * the integrals have been computed and does the SVD-like part of the 
 * computation.
 */
class SCFSolver {
  
  friend class SCFSolverTest;
  
  FORBID_ACCIDENTAL_COPIES(SCFSolver);
  
 private:
  
  Matrix basis_centers_;
  Matrix nuclear_centers_;  
  
  Vector nuclear_masses_;
  
  Matrix core_matrix_; // T + V
  Matrix kinetic_energy_integrals_; // T
  Matrix potential_energy_integrals_; // V
  
  Matrix coefficient_matrix_; // C or C'
  
  // Consider changing this name to reflect that it's the change of basis matrix
  Matrix overlap_matrix_; // S 
  Matrix change_of_basis_matrix_; // S^{-1/2} 
  Matrix density_matrix_; // D
  Matrix fock_matrix_; // F or F', depending on the basis
  
  Vector energy_vector_; // The diagonal matrix of eigenvalues of F/F'
  
  index_t number_of_basis_functions_; // N
  index_t number_of_electrons_; // K
  index_t number_of_nuclei_; 
  index_t number_to_fill_;
  
  // I think I'll have to compute this in the beginning
  double nuclear_repulsion_energy_;
  
  ArrayList<double> total_energy_;
  index_t current_iteration_;
  
  static const index_t expected_number_of_iterations_ = 40;
  
  double convergence_tolerance_;
  
  // Used for computing the convergence of the density matrix on the fly
  double density_matrix_frobenius_norm_;
  
  struct datanode* module_;
  
  DualTreeIntegrals integrals_;
  
  ArrayList<index_t> occupied_indices_;
  
  ArrayList<index_t> old_from_new_centers_;
  
  double bandwidth_;
  
 public:
    
  SCFSolver() {}
  
  ~SCFSolver() {}
  

  void Init(struct datanode* mod, index_t num_electrons, 
            const Matrix& basis_centers, const Matrix& density, 
            const Matrix& nuclear, const Vector& nuclear_mass) {
    
    module_ = mod;
    number_of_electrons_ = num_electrons;
    
    struct datanode* integral_mod = fx_submodule(module_, "integrals", 
                                                "integrals");
    
    bandwidth_ = fx_param_double(module_, "bandwidth", 0.1);
    
    integrals_.Init(basis_centers, integral_mod, bandwidth_);
    
    // Need to get out the permutation from the integrals_, then use it to 
    // permute the basis centers
    
    integrals_.GetPermutation(&old_from_new_centers_);
    
    PermuteMatrix_(basis_centers, &basis_centers_, old_from_new_centers_);
    PermuteMatrix_(density, &density_matrix_, old_from_new_centers_);
    
    integrals_.GetDensity(density_matrix_);
    
    nuclear_centers_.Copy(nuclear);
    
    nuclear_masses_.Copy(nuclear_mass);
    
    number_of_nuclei_ = nuclear_centers_.n_cols();
    
    number_to_fill_ = (index_t)ceil((double)number_of_electrons_/2);
    occupied_indices_.Init(number_to_fill_);
    
    DEBUG_ASSERT(number_of_nuclei_ == nuclear_masses_.length());
    
    
    number_of_basis_functions_ = basis_centers_.n_cols();
    
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
    
    total_energy_.Init(expected_number_of_iterations_);
    
    convergence_tolerance_ = fx_param_double(module_, "convergence_tolerance", 
                                             0.1);
    
    // Need to double check that this is right
    density_matrix_frobenius_norm_ = DBL_MAX;
    
    current_iteration_ = 0;
    
  } // Init()
  
 private:
 
  double ComputeOverlapIntegral_(double dist);
  
  double ComputeKineticIntegral_(double dist);
  
  double ComputeNuclearIntegral_(const Vector& nuclear_position, 
                                 const Vector& mu, const Vector& nu);
    
  /**
   * Permutes the matrix mat according to the permutation given.  The permuted 
   * matrix is written to new_mat, overwriting whatever was there before
   */                               
  void PermuteMatrix_(const Matrix& old_mat, Matrix* new_mat, 
                      const ArrayList<index_t>& perm) {
  
    index_t num_cols = old_mat.n_cols();
    DEBUG_ASSERT(num_cols == perm.size());
    
    new_mat->Init(old_mat.n_rows(), num_cols);
    
    for (index_t i = 0; i < num_cols; i++) {
      
      Vector old_vec;
      old_mat.MakeColumnVector(i, &old_vec);
      Vector new_vec;
      new_mat->MakeColumnVector(perm[i], &new_vec);
      
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
  void FormChangeOfBasisMatrix_() {
    
    Vector eigenvalues;
    Matrix eigenvectors;
    la::EigenvectorsInit(overlap_matrix_, &eigenvalues, &eigenvectors);
    
#ifdef DEBUG
    
    for (index_t i = 0; i < eigenvalues.length(); i++) {
      DEBUG_ASSERT_MSG(!isnan(eigenvalues[i]), 
                       "Complex eigenvalue in diagonalizing overlap matrix.\n");
    }
    
#endif
    
    for (index_t i = 0; i < eigenvalues.length(); i++) {
      DEBUG_ASSERT(eigenvalues[i] > 0.0);
      eigenvalues[i] = 1/sqrt(eigenvalues[i]);
    }
    
    Matrix sqrt_lambda;
    sqrt_lambda.InitDiagonal(eigenvalues);
    
    Matrix lambda_times_u_transpose;
    la::MulTransBInit(sqrt_lambda, eigenvectors, &lambda_times_u_transpose);
    la::MulInit(eigenvectors, lambda_times_u_transpose, 
                     &change_of_basis_matrix_);
    
  } // FormChangeOfBasisMatrix_()
  
  
  /**
   * Compute the density matrix.
   * 
   * TODO: Consider an SVD or some eigenvalue solver that will find the 
   * eigenvalues in ascending order.
   */
  void ComputeDensityMatrix_() {
    
    FillOrbitals_();
    
    // I MUST find a smarter way to do this for large problems
    // This could also probably be a separate function
    
    // Rows of density_matrix
    for (index_t density_row = 0; density_row < number_of_basis_functions_;
         density_row++) {
      
      // Columns of density matrix
      for (index_t density_column = density_row; 
           density_column < number_of_basis_functions_; density_column++) {
        
        // Occupied orbitals
        double this_sum = 0.0;
        for (index_t occupied_index = 0; 
             occupied_index < occupied_indices_.size(); occupied_index++) {
          
          // By multiplying by 2, I'm assuming all the orbitals are full
          this_sum = this_sum + ( 2 *
              coefficient_matrix_.ref(
                  density_row, occupied_indices_[occupied_index]) * 
              coefficient_matrix_.ref(
                  density_column, occupied_indices_[occupied_index]));
          
        } // occupied_index
        
        // I think this is necessary, but not sure
        if (likely(density_row != density_column)) {
          this_sum = 2 * this_sum;
        }
        
        // Computing the frobenius norm of the difference between this 
        // iteration's density matrix and the previous one for testing 
        // convergence
        if (likely(current_iteration_ > 0)) {
          density_matrix_frobenius_norm_ = density_matrix_frobenius_norm_ + 
              (this_sum - density_matrix_.ref(density_row, density_column)) * 
              (this_sum - density_matrix_.ref(density_row, density_column)); 
        }
        
        density_matrix_.set(density_row, density_column, this_sum);
        if (density_row != density_column) {
          density_matrix_.set(density_column, density_row, this_sum);        
        }
        
      } // density_column
      
    } //density_row
    
    printf("density_matrix_norm: %g\n", density_matrix_frobenius_norm_);
    
    density_matrix_.PrintDebug();
    
  } // ComputeDensityMatrix_
  
  /**
    * Given that the Fock matrix has been transformed to the orthonormal basis
   * (F'), this function determines the energy_vector e and the transformed 
   * coefficient matrix C'.  It then untransforms the matrix to get C.
   */
  void DiagonalizeFockMatrix_() {
    
    energy_vector_.Destruct();
    
    Matrix coefficients_prime;
    la::EigenvectorsInit(fock_matrix_, &energy_vector_, &coefficients_prime);
    
#ifdef DEBUG
    
    for (index_t i = 0; i < energy_vector_.length(); i++) {
      DEBUG_ASSERT_MSG(!isnan(energy_vector_[i]), 
          "Complex eigenvalue in diagonalizing initial Fock matrix.\n");
    }
    
#endif
    
    // 3. Find the untransformed eigenvector matrix
    la::MulOverwrite(change_of_basis_matrix_, coefficients_prime, 
                     &coefficient_matrix_);
    
  } // DiagonalizeFockMatrix_
  
  
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
        DEBUG_ASSERT(next_to_go < number_of_basis_functions_);
      }
      
    }
    
  } //FillOrbitals_
  
  
    
  /**
   * Find the energy of the electrons in the ground state of the current 
   * wavefunction.  
   */
  double ComputeElectronicEnergy_() {
    
    double total_energy = nuclear_repulsion_energy_;
    
    for (index_t mu = 0; mu < number_of_basis_functions_; mu++) {
     
      for (index_t nu = 0; nu < number_of_basis_functions_; nu++) {
       
       // I don't think this is right
       // I guess the density matrix needs to multiply the one electron too?
        total_energy = total_energy + density_matrix_.ref(mu, nu) 
            * (core_matrix_.ref(mu, nu) + fock_matrix_.ref(mu, nu));
        
      }
      
    }
    
    return total_energy;
    
  } // ComputeElectronicEnergy_
  
  /**
    * Determine if the density matrix and total energy have converged.   
   */
  bool TestConvergence_() {
    
    bool is_converged = true;
    
    if (unlikely(current_iteration_ == 0)) {
      return false;
    }
    
    if (likely(density_matrix_frobenius_norm_ > 
                (convergence_tolerance_ * convergence_tolerance_))) {
      is_converged = false;
    }
    
    if (likely((total_energy_[current_iteration_] 
                 - total_energy_[current_iteration_ - 1]) 
                > convergence_tolerance_)) {
      is_converged = false;
    }
    
    density_matrix_frobenius_norm_ = 0.0;
    
    return is_converged;
    
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
  
  void UpdateFockMatrix_() {

    // Needs to call something from the object, preferably updating it first?
    
    integrals_.UpdateMatrices(density_matrix_);
    
    integrals_.ComputeFockMatrix();
    
    la::AddOverwrite(core_matrix_, integrals_.FockMatrix(), &fock_matrix_);

  }
  
  /**
    * Does the SCF iterations to find the HF wavefunction
   */
  void FindSCFSolution_() {
    
    bool converged = false;
    
    while (!converged) {
      
      // Step 4a.
      UpdateFockMatrix_();
      
      // Step 4b.
      if (unlikely(current_iteration_ >= total_energy_.size())) {
        total_energy_.EnsureSizeAtLeast(2*total_energy_.size());
      }
      
      total_energy_[current_iteration_] = ComputeElectronicEnergy_();
      
      // Step 4c.
      TransformFockBasis_();
      
      // Step 4d/e.
      DiagonalizeFockMatrix_();
      
      //Step 4f.
      ComputeDensityMatrix_();
      
      // Step 4g.
      converged = TestConvergence_();
      
      current_iteration_++;
      
    } // end while
    
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
    
    fock_matrix_.Alias(core_matrix_);
    
    TransformFockBasis_();
    
    DiagonalizeFockMatrix_();
    
    ComputeDensityMatrix_();
    
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
    for (index_t i = 0; i < (current_iteration_ - 1); i++) {
     
      fprintf(energy_pointer, "Iteration %d:\t %f\n", i, total_energy_[i]);
      
    }
    fclose(energy_pointer);
    
    const char* energy_vector_file = 
        fx_param_str(module_, "Evec", "energy_vector.csv");
    Matrix energy_vector_matrix;
    energy_vector_matrix.AliasColVector(energy_vector_);
    data::Save(energy_vector_file, energy_vector_matrix);
    
    fx_format_result(module_, "density_matrix_norm", "%g", 
                     density_matrix_frobenius_norm_);
    
    fx_format_result(module_, "num_iterations", "%d", current_iteration_);
    
    fx_format_result(module_, "total_energy", "%g", 
                     total_energy_[current_iteration_-1]);
    
    integrals_.OutputFockMatrix(NULL, NULL, NULL, NULL);
    
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
    
    printf("Core Matrix:\n");
    ot::Print(core_matrix_);
    
    printf("Coefficient matrix:\n");
    ot::Print(coefficient_matrix_);
    
    printf("Change-of-basis matrix:\n");
    ot::Print(overlap_matrix_);
    
    printf("Density matrix:\n");
    ot::Print(density_matrix_);
    
    printf("Fock matrix:\n");
    ot::Print(fock_matrix_);
    
    printf("Energy vector:\n");
    ot::Print(energy_vector_);
    
    printf("Total energy:\n");
    ot::Print(total_energy_);
    
  } // PrintMatrices
  
}; // class HFSolver


#endif // inclusion guards