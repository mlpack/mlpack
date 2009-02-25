#ifndef NAIVE_FOCK_MATRIX_H
#define NAIVE_FOCK_MATRIX_H

#include "fastlib/fastlib.h"
#include "contrib/march/fock_matrix/fock_impl/eri.h"


class NaiveFock {

 private:

  Matrix centers_;
  Vector exponents_;
  Vector momenta_;

  ArrayList<BasisShell> shells_;
  
  index_t num_funs_;
  
  fx_module* mod_;
  
  Matrix coulomb_mat_;
  Matrix exchange_mat_;
  Matrix fock_mat_;

 public:

  void Init(const Matrix& centers, const Matrix& exponents, 
            const Matrix& momenta, fx_module* mod_in) {
  
    centers_.Copy(centers);
    
    exponents.MakeColumnVector(0, &exponents_);
    momenta.MakeColumnVector(0, &momenta_);
    
    // This only works for s and p type functions
    num_funs_ = centers_.n_cols() + (index_t)2*la::Dot(momenta_, momenta_);
    
    eri::CreateShells(centers_, exponents_, momenta_, &shells_);
    
    mod_ = mod_in;
    
  
  } // Init()
  
  
  void ComputeFock() {
  
  } // ComputeFock()
  
  
  /**
   * Save the results to matrices for comparison.  Also saves fx results.
   */
  void OutputFock(Matrix* fock_out, Matrix* coulomb_out, Matrix* exchange_out) {
  
    if (fock_out != NULL) {
      fock_out.Copy(fock_mat_);
    }
    
    if (coulomb_out != NULL) {
      coulomb_out.Copy(coulomb_mat_);
    }
    
    if (exchange_out != NULL) {
      exchange_out.Copy(exchange_mat_);
    }
    
    // output results
  
  }
  

}; // NaiveFock




#endif 

