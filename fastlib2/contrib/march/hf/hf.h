/** 
 * @file hf.h
 *
 * @author Bill March (march@gatech.edu)
 * 
 * Contains classes for the Hartree-Fock implementation.  
 */

#ifndef HF_H
#define HF_H


/**
 * A class that stores the information for a contracted Gaussian basis function.
 *
 * TODO: Should this class also have functions for computations among contracted 
 * functions?
 */
class ContractedGaussian {
  
  FORBID_ACCIDENTAL_COPY(ContractedGaussian);
  
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
 */
class HFSolver {
  
  FORBID_ACCIDENTAL_COPY(HFSolver);
 
 private:
 
  Matrix fock_matrix_;
  Matrix overlap_matrix_;
  
  Vector coefficient_vector_;
  Vector energy_vector_;
  
  index_t number_of_basis_functions_;
  index_t number_of_electrons_;
  
  
 public:
 
  HFSolver() {}
  
  ~HFSolver() {}
  
  /** 
   * Initialize the class with const references to the Fock matrix and the 
   * overlap matrix, both of which should have been computed already.
   */
  void Init(const Matrix& fock_in, const Matrix& overlap_in) {
    
    fock_matrix_.Copy(fock_in);
    overlap_matrix_.Copy(overlap_in);
    
    // I think this will be necessary for the lapack routines 
    coefficient_vector_.Init(number_of_basis_functions_);
    energy_vector_.Init(number_of_basis_functions_);
    
  } // Init
  

  
  
  
  
}; // class HFSolver



#endif