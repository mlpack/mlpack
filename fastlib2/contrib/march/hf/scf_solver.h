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

/**
 * Algorithm class for the SCF part of the HF computation.  This class assumes 
 * the integrals have been computed and does the SVD-like part of the 
 * computation.
 *
 * For now, this is simply an implementation of the basic algorithm.  In the 
 * future, I should examine how this could be done better.
 */
class SCFSolver {
  
  friend class SCFSolverTest;
  
  FORBID_ACCIDENTAL_COPIES(SCFSolver);
  
 private:
    
  // I can probably be more efficient in terms of storing these matrices
  // I don't want to store many matrices of this size in the final code
    
  Vector two_electron_integrals_;
  
  Matrix one_electron_integrals_; // T + V
  Matrix kinetic_energy_integrals_; // T
  Matrix potential_energy_integrals_; // V
  
  Matrix coefficient_matrix_; // C or C'
  
  // Consider changing this name to reflect that it's the change of basis matrix
  Matrix overlap_matrix_; // S or S^{-1/2}
  Matrix density_matrix_; // D
  Matrix fock_matrix_; // F or F', depending on the basis
  
  Vector energy_vector_; // The diagonal matrix of eigenvalues of F/F'
  
  index_t number_of_basis_functions_; // N
  index_t number_of_electrons_; // K
  
  double nuclear_repulsion_energy_;
  
  ArrayList<double> total_energy_;
  index_t current_iteration_;
  
  static const index_t expected_number_of_iterations_ = 40;
  
  double convergence_tolerance_;
  
  // Used for computing the convergence of the density matrix on the fly
  double density_matrix_frobenius_norm_;
  
  struct datanode* module_;
  
 public:
    
  SCFSolver() {}
  
  ~SCFSolver() {}
  
  /** 
    * Initialize the class with const references to the electron matrices and 
    * the overlap matrix, both of which should have been computed already.
    */
  void Init(double nuclear_energy, const Matrix& overlap_in, 
            const Matrix& kinetic_in, const Matrix& potential_in,
            const Vector& two_electron_in, index_t num_electrons,
            double converged, struct datanode* mod) {
    
    nuclear_repulsion_energy_ = nuclear_energy;
    number_of_electrons_ = num_electrons;
    
    // Read in integrals
    overlap_matrix_.Copy(overlap_in);
    kinetic_energy_integrals_.Copy(kinetic_in);
    potential_energy_integrals_.Copy(potential_in);
    
    // Is this really the best thing to do?
    // Copying this whole thing might be too expensive
    two_electron_integrals_.Copy(two_electron_in);
    
    number_of_basis_functions_ = overlap_matrix_.n_cols();
    
    DEBUG_ASSERT(number_of_basis_functions_ >= number_of_electrons_);
    
    // Form the core Hamiltonian
    la::AddInit(kinetic_energy_integrals_, potential_energy_integrals_, 
                &one_electron_integrals_);
    
    // Empty inits to prevent errors on closing
    coefficient_matrix_.Init(number_of_basis_functions_, 
                             number_of_basis_functions_);
    density_matrix_.Init(number_of_basis_functions_, 
                         number_of_basis_functions_);
    fock_matrix_.Init(number_of_basis_functions_, number_of_basis_functions_);
    
    energy_vector_.Init(number_of_basis_functions_);
    
    total_energy_.Init(expected_number_of_iterations_);
    
    convergence_tolerance_ = converged;
    
    density_matrix_frobenius_norm_ = DBL_MAX;
    
    module_ = mod;
    
  } // Init
  
 private:
  
  /**
   * Finds a pairwise index for use in finding the entire index of an integral.
   */
  index_t FindIntegralIndexHelper_(index_t mu, index_t nu) {
      
    //DEBUG_ASSERT(mu >= nu);
    if (mu >= nu) {
      return (((mu * (mu + 1))/2) + nu);
    }
    else {
      return (((nu * (nu + 1))/2) + mu); 
    }
  } // FindIntegralIndexHelper_
  
  /**
    * Find the index of the two electron integral (mu nu | rho sigma) in the 
   * two-electron integrals array
   */
  index_t FindIntegralIndex_(index_t mu, index_t nu, index_t rho, index_t sig) {
    
    //DEBUG_ASSERT(mu >= nu);
    //DEBUG_ASSERT(rho >= sig);
    
    index_t mu_nu = FindIntegralIndexHelper_(mu, nu);
    index_t rho_sig = FindIntegralIndexHelper_(rho, sig);
    
    //DEBUG_ASSERT(mu_nu >= rho_sig);
    if (mu_nu >= rho_sig) {
      return (FindIntegralIndexHelper_(mu_nu, rho_sig));
    }
    else {
      return (FindIntegralIndexHelper_(rho_sig, mu_nu)); 
    }
      
  }// FindIntegralIndex_
  
  
  /**
   * Create the matrix S^{-1/2} using the eigenvector decomposition.  Overwrites 
   * overlap_matrix_ with S^{-1/2}.
   *
   */
  void FormOrthogonalizingMatrix_() {
    
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
    la::MulOverwrite(eigenvectors, lambda_times_u_transpose, &overlap_matrix_);
    
  } // FormOrthogonalizingMatrix_
  
  
  /**
   * Compute the density matrix.
   * 
   * TODO: Consider an SVD or some eigenvalue solver that will find the 
   * eigenvalues in decending order.
   */
  void ComputeDensityMatrix_() {
    
    ArrayList<index_t> occupied_orbitals;
    
    FillOrbitals_(&occupied_orbitals);
    
    // I MUST find a smarter way to do this for large problems
    // This could also probably be a separate function
    
    // Rows of density_matrix
    for (index_t density_row = 0; density_row < number_of_basis_functions_;
         density_row++) {
      
      // Columns of density matrix
      for (index_t density_column = 0; 
           density_column < number_of_basis_functions_; density_column++) {
        
        // Occupied orbitals
        double this_sum = 0.0;
        for (index_t occupied_index = 0; 
             occupied_index < occupied_orbitals.size(); occupied_index++) {
          
          // By multiplying by 2, I'm assuming all the orbitals are full
          this_sum = this_sum + ( 2 *
              coefficient_matrix_.ref(
                  density_row, occupied_orbitals[occupied_index]) * 
              coefficient_matrix_.ref(
                  density_column, occupied_orbitals[occupied_index]));
          
        } // occupied_index
        
        // Computing the frobenius norm of the difference between this 
        // iteration's density matrix and the previous one for testing 
        // convergence
        if (likely(current_iteration_ > 0)) {
          density_matrix_frobenius_norm_ = density_matrix_frobenius_norm_ + 
              (this_sum - density_matrix_.ref(density_row, density_column)) * 
              (this_sum - density_matrix_.ref(density_row, density_column)); 
        }
        
        density_matrix_.set(density_row, density_column, this_sum);
        
      } // density_column
      
    } //density_row
    
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
    la::MulOverwrite(overlap_matrix_, coefficients_prime, 
                     &coefficient_matrix_);
    
  } // DiagonalizeFockMatrix_
  
  
  /**
   * Determine the K/2 lowest energy orbitals.  
   * 
   * TODO: If K is odd, then the last entry here is the orbital that should 
   * have one electron.
   *
   * I should use the built in C++ iterator-driven routines.  Ryan suggested
   * that I write iterators for ArrayLists, since that would open up a lot of
   * functionality, including sorting.  
   */
  void FillOrbitals_(ArrayList<index_t>* indices) {
    
    index_t number_to_fill = (index_t)ceil((double)number_of_electrons_/2);
    
    indices->Init(number_to_fill);
    
    double max_energy_kept = -DBL_INF;
    index_t next_to_go = 0;
    
    for (index_t i = 0; i < number_of_basis_functions_; i++) {
      
      if (unlikely(i < number_to_fill))  {
        
        (*indices)[i] = i;
        if (energy_vector_[i] > max_energy_kept) {
          max_energy_kept = energy_vector_[i];
          next_to_go = i;
        }
        
      }
      else {
        
        double this_energy = energy_vector_[i];
        if (this_energy < max_energy_kept) {
          (*indices)[next_to_go] = i;
          
          // Find the new index to throw out
          double new_max = -DBL_INF;
          next_to_go = -1;
          for (index_t j = 0; j < number_to_fill; j++) {
            if (energy_vector_[(*indices)[j]] > new_max) {
              new_max = energy_vector_[(*indices)[j]];
              next_to_go = j;
            }
          }
          max_energy_kept = new_max;
          DEBUG_ASSERT(!isinf(max_energy_kept));
          DEBUG_ASSERT(next_to_go >= 0);
        }
        
      }
      
    }
    
  } //FillOrbitals_
  
  
  /**
    * Do step 4a. in Sherrill's notes.  This is a key step that could be turned 
   * into an N-body computation.  
   *
   * TODO: Figure out how to make a matrix that is the density weighted two
   * electron integrals.  That would eliminate at least two of the for loops.
   *
   * BUG: I'm counting each integral more than once.  
   */
  void UpdateFockMatrix_() {
    
    for (index_t mu = 0; mu < number_of_basis_functions_; mu++) {
     
      for (index_t nu = 0; nu < number_of_basis_functions_; nu++) {
       
        double new_value = one_electron_integrals_.ref(mu, nu);
        
        for (index_t rho = 0; rho < number_of_basis_functions_; rho++) {
         
          for (index_t sigma = 0; sigma <= rho; sigma++) {
           
            printf("%d,%d,%d,%d\n", mu, nu, rho, sigma);
            
            index_t first_index = FindIntegralIndex_(mu, nu, rho, sigma);
            index_t second_index = FindIntegralIndex_(mu, rho, nu, sigma);
            
            new_value = new_value + density_matrix_.ref(rho, sigma) * 
                (2 * two_electron_integrals_[first_index] - 
                 two_electron_integrals_[second_index]);
            
          }
          
        }
        
        fock_matrix_.set(mu, nu, new_value);
        
      }
      
    }
    
  } // UpdateFockMatrix_
  
  /**
   * Find the energy of the electrons in the ground state of the current 
   * wavefunction.  
   */
  double ComputeElectronicEnergy_() {
    
    double total_energy = nuclear_repulsion_energy_;
    
    for (index_t mu = 0; mu < number_of_basis_functions_; mu++) {
     
      for (index_t nu = 0; nu < number_of_basis_functions_; nu++) {
       
        total_energy = total_energy + density_matrix_.ref(mu, nu) 
            * (one_electron_integrals_.ref(mu, nu) + fock_matrix_.ref(mu, nu));
        
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
      density_matrix_frobenius_norm_ = 0.0;
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
    la::MulTransAInit(overlap_matrix_, fock_matrix_, 
                           &orthogonal_transpose_times_fock);
    la::MulOverwrite(orthogonal_transpose_times_fock, 
                     overlap_matrix_, &fock_matrix_);
    
  } // TransformFockBasis_
  
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
   * Sets up the matrices for the SCF iterations 
   */
  void Setup_() {
    
    FormOrthogonalizingMatrix_();
    
    // 1. Form the core Fock matrix in transformed basis 
    // F' = S^{-1/2} H S^{-1/2}
    // For now, we assume the initial density matrix is zero
    // In the future, I should support non-zero initialization
   
    fock_matrix_.CopyValues(one_electron_integrals_);
    
    TransformFockBasis_();  // fock_matrix_ should now be F'
   
    // 2. Solve the transformed Fock matrix eigenvalue problem
    
    DiagonalizeFockMatrix_();
    
    ComputeDensityMatrix_();
    
    density_matrix_frobenius_norm_ = 0.0;
    
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
    
  }
  
 public:
  
  /**
   * Compute the Hartree-Fock wavefunction for the given values of the 
   * integrals. 
   */
  void ComputeWavefunction() {
      
    Setup_();
      
    FindSCFSolution_();
    
    OutputResults_();
      
  } // ComputeWavefunction
  
  void PrintMatrices() {
    
    printf("One electron integrals:\n");
    ot::Print(one_electron_integrals_);
    
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