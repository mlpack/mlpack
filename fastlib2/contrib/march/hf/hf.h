/** 
 * @file hf.h
 *
 * @author Bill March (march@gatech.edu)
 * 
 * Contains classes for the Hartree-Fock implementation.  
 */

#ifndef HF_H
#define HF_H
#include <fastlib/fastlib.h>

/**
 * A class that stores the information for a contracted Gaussian basis function.
 *
 * TODO: Should this class also have functions for computations among contracted 
 * functions?
 */
class ContractedGaussian {
  
  FORBID_ACCIDENTAL_COPIES(ContractedGaussian);
  
 private:
  
  // The number of primitive Gaussians that make up this function
  index_t number_of_primitives_;
  ArrayList<double> bandwidths_;
  ArrayList<double> coefficients_;
  
  
 public:
  
  ContractedGaussian() {}
  
  ~ContractedGaussian() {}
  
  void Init(index_t num, const ArrayList<double>& band, 
            const ArrayList<double>& coeff) {
    
    number_of_primitives_ = num;
    bandwidths_.Copy(band);
    coefficients_.Copy(coeff);
    
  }
  
  
};


/**
 * Algorithm class for the basic part of the HF computation.  This class assumes 
 * the integrals have been computed and does the SVD-like part of the 
 * computation.
 *
 * For now, this is simply an implementation of the basic algorithm.  In the 
 * future, I should examine how this could be done better.
 */
class HFSolver {
  friend class HartreeFockTest;
  
  FORBID_ACCIDENTAL_COPIES(HFSolver);
 
  
 private:
 
  // I can probably be more efficient in terms of storing these matrices
  // I don't want to store many matrices of this size in the final code
  
  // This isn't a matrix, it's a rank-four tensor
  // I need to figure out what to do with this
  Matrix two_electron_integrals_;
  
    
  Matrix one_electron_integrals_;
  Matrix kinetic_energy_integrals_;
  Matrix potential_energy_integrals_;

  Matrix coefficient_matrix_;
  Matrix overlap_matrix_;
  Matrix density_matrix_;
  Matrix fock_matrix_;
    
  Vector energy_vector_;
  
  index_t number_of_basis_functions_;
  index_t number_of_electrons_;
  
  double nuclear_repulsion_energy_;
  
 public:
 
  HFSolver() {}
  
  ~HFSolver() {}
  
  /** 
   * Initialize the class with const references to the electron matrices and the 
   * overlap matrix, both of which should have been computed already.
   */
  void Init(double nuclear_energy, const Matrix& overlap_in, 
            const Matrix& kinetic_in, const Matrix& potential_in,
            const Matrix& two_electron_in, index_t num_electrons) {
    
    nuclear_repulsion_energy_ = nuclear_energy;
    number_of_electrons_ = num_electrons;
    
    // Read in integrals
    overlap_matrix_.Copy(overlap_in);
    kinetic_energy_integrals_.Copy(kinetic_in);
    potential_energy_integrals_.Copy(potential_in);
    two_electron_integrals_.Copy(two_electron_in);
    
    number_of_basis_functions_ = overlap_matrix_.n_cols();
    
    // Form the core Hamiltonian
    la::AddInit(kinetic_energy_integrals_, potential_energy_integrals_, 
                &one_electron_integrals_);
    
        
  } // Init
  
  /**
   * Create the matrix S^{-1/2} using the Schur decomposition.  Overwrites 
   * overlap_matrix_ with S^{-1/2}.
   *
   * TODO: go over the linear algrebra and make sure it is efficient
   */
  void FormOrthogonalizingMatrix() {
    
    // Form the orthogonalizing matrix S^{-1/2}
    // Should change this to SchurExpert eventually
    Vector real_eigenvalues;
    Vector imaginary_eigenvalues;
    Matrix schur_form;
    Matrix schur_vectors;
    
    success_t diagonalize = la::SchurInit(overlap_matrix_, &real_eigenvalues,
        &imaginary_eigenvalues, &schur_form, &schur_vectors);
    
    if (diagonalize == SUCCESS_FAIL) {
      // Need to handle this better
      FATAL("Schur Decomposition Failed\n");
    }
    
#ifdef DEBUG
    // Check that the eigenvalues are all real
    for (index_t i = 0; i < imaginary_eigenvalues.length(); i++) {
      DEBUG_ASSERT(imaginary_eigenvalues[i] == 0.0);
    }
    
    // Also check that the Schur form is strictly diagonal
    for (index_t i = 0; i < schur_form.n_rows(); i++) {
      for (index_t j = (i+1); j < schur_form.n_cols(); j++) {
        DEBUG_ASSERT(schur_form.ref(i,j) == 0.0); 
      }
    }
#endif
    
    // Compute lambda^{-1/2}
    for (index_t i = 0; i < real_eigenvalues.length(); i++) {
      real_eigenvalues[i] = 1/sqrt(real_eigenvalues[i]);
    }
    
    Matrix sqrt_lambda;
    sqrt_lambda.InitDiagonal(real_eigenvalues);
    Matrix lambda_times_u_transpose;
    la::MulTransBInit(sqrt_lambda, schur_vectors, &lambda_times_u_transpose);
    la::MulOverwrite(schur_vectors, lambda_times_u_transpose, &overlap_matrix_);
        
  } // FormOrthogonalizingMatrix
  

  /**
   * Compute an initial density matrix
   */

  
  
  
  
}; // class HFSolver



#endif