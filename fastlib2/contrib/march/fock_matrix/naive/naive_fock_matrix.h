#ifndef NAIVE_FOCK_MATRIX_H
#define NAIVE_FOCK_MATRIX_H

#include "fastlib/fastlib.h"
#include "contrib/march/fock_matrix/fock_impl/eri.h"


const fx_entry_doc naive_fock_class_entries[] = {
{},
  FX_ENTRY_DOC_DONE
};

const fx_module_doc naive_mod_doc = {
  naive_fock_class_entries, NULL, 
  "A naive fock matrix construction algorithm using 4-way symmetry.\n"
};


class NaiveFockMatrix {

 private:

  Matrix centers_;
  Vector exponents_;
  Vector momenta_;
  Matrix density_;

  ArrayList<BasisShell> shells_;
  
  index_t num_funs_;
  
  index_t num_shells_;
  
  fx_module* mod_;
  
  Matrix coulomb_mat_;
  Matrix exchange_mat_;
  Matrix fock_mat_;

 public:

  void Init(const Matrix& centers, const Matrix& exponents, 
            const Matrix& momenta, const Matrix& density, fx_module* mod_in) {
  
    printf("Init naive\n");
  
    centers_.Copy(centers);
    
    DEBUG_ASSERT(exponents.n_cols() == momenta.n_cols());
    DEBUG_ASSERT(exponents.n_cols() == centers_.n_cols());
    
    exponents_.Copy(exponents.ptr(), centers_.n_cols());
    momenta_.Copy(momenta.ptr(), centers_.n_cols());
    
    density_.Copy(density);
    
    // This only works for s and p type functions
    num_funs_ = centers_.n_cols() + (index_t)2*la::Dot(momenta_, momenta_);
    
    eri::CreateShells(centers_, exponents_, momenta_, &shells_);
    
    num_shells_ = shells_.size();
    
    mod_ = mod_in;
    
    coulomb_mat_.Init(num_funs_, num_funs_);
    exchange_mat_.Init(num_funs_, num_funs_);
    
  
  } // Init()
  
  
  void ComputeFock() {
    
    for (index_t i = 0; i < num_shells_; i++) {
    
      for (index_t j = 0; j <= i; j++) {
      
        double ij_coulomb = 0.0;
        double ij_exchange = 0.0;
      
        for (index_t k = 0; k < num_shells_; k++) {
        
          for (index_t l = 0; l <= k; l++) {
          
            double coulomb_int = eri::ComputeShellIntegrals(shells_[i], 
                                                            shells_[j], 
                                                            shells_[k], 
                                                            shells_[l]);
                                                         
            double exchange_int = eri::ComputeShellIntegrals(shells_[i], 
                                                             shells_[k], 
                                                             shells_[j], 
                                                             shells_[l]);
            
            coulomb_int = density_.ref(k, l) * coulomb_int;
            exchange_int = density_.ref(k,l) * exchange_int;
          
            if (likely(k != l)) {
            
              coulomb_int *= 2;
              exchange_int *= 2;
            
            }
          
            ij_coulomb += coulomb_int;
            ij_exchange += exchange_int;
          
          } // for l
        
        } // for k
        
        coulomb_mat_.set(i, j, ij_coulomb);
        coulomb_mat_.set(j, i, ij_coulomb);
        
        exchange_mat_.set(i, j, ij_exchange);
        exchange_mat_.set(j, i, ij_exchange);
      
      } // for j
      
      
    
    } // for i
    
    la::ScaleInit(-0.5, exchange_mat_, &fock_mat_);
  
    la::AddTo(coulomb_mat_, &fock_mat_);
  
  } // ComputeFock()
  
  
  /**
   * Save the results to matrices for comparison.  Also saves fx results.
   */
  void OutputFock(Matrix* fock_out, Matrix* coulomb_out, Matrix* exchange_out) {
  
    if (fock_out != NULL) {
      fock_out->Copy(fock_mat_);
    }
    
    if (coulomb_out != NULL) {
      coulomb_out->Copy(coulomb_mat_);
    }
    
    if (exchange_out != NULL) {
      exchange_out->Copy(exchange_mat_);
    }
    
    // output results
  
  }
  

}; // NaiveFock




#endif 

