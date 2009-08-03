#ifndef NAIVE_FOCK_MATRIX_H
#define NAIVE_FOCK_MATRIX_H

#include "fastlib/fastlib.h"
#include "contrib/march/fock_matrix/fock_impl/eri.h"


const fx_entry_doc naive_fock_class_entries[] = {
{"naive_time", FX_TIMER, FX_CUSTOM, NULL,
  "The time spent on the naive computation.\n"},
{"num_integrals_computed", FX_RESULT, FX_INT, NULL,
  "The total number of integrals computed.\n"},
{"N", FX_RESULT, FX_INT, NULL, 
"The total number of basis functions, as in the dimension of the Fock matrix.\n"},
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
  
  index_t num_integrals_computed_;
  
  fx_module* mod_;
  
  Matrix coulomb_mat_;
  Matrix exchange_mat_;
  Matrix fock_mat_;

 public:

  void Init(const Matrix& centers, const Matrix& exponents, 
            const Matrix& momenta, const Matrix& density, fx_module* mod_in) {
  
    printf("====Init Naive====\n");
  
    mod_ = mod_in;
    
    centers_.Copy(centers);
    
    DEBUG_ASSERT(exponents.n_cols() == momenta.n_cols());
    DEBUG_ASSERT(exponents.n_cols() == centers_.n_cols());
    
    exponents_.Copy(exponents.ptr(), centers_.n_cols());
    momenta_.Copy(momenta.ptr(), centers_.n_cols());
    
    density_.Copy(density);
    
    // This only works for s and p type functions
    //num_funs_ = centers_.n_cols() + (index_t)2*la::Dot(momenta_, momenta_);
    
    num_funs_ = eri::CreateShells(centers_, exponents_, momenta_, &shells_);
    fx_result_int(mod_, "N", num_funs_);
    
    if (density_.n_cols() != num_funs_) {
      FATAL("Given density matrix does not match basis!\n");
    }
    
    num_shells_ = shells_.size();
    
    
    
    coulomb_mat_.Init(num_funs_, num_funs_);
    exchange_mat_.Init(num_funs_, num_funs_);
    fock_mat_.Init(num_funs_, num_funs_);
    coulomb_mat_.SetZero();
    exchange_mat_.SetZero();
    fock_mat_.SetZero();
    
    num_integrals_computed_ = 0;
    
  
  } // Init()
  
  
  void Compute() {
    
    printf("====Compute Naive====\n");
    
    fx_timer_start(mod_, "naive_time");
    
    for (index_t i = 0; i < num_shells_; i++) {
    
      for (index_t j = 0; j <= i; j++) {
      
        double ij_coulomb = 0.0;
        //double ij_exchange = 0.0;
      
        for (index_t k = 0; k < num_shells_; k++) {
        
          for (index_t l = 0; l <= k; l++) {
            
            index_t num_integrals;
            ArrayList<index_t> perm;
          
            double* integrals = eri::ComputeShellIntegrals(shells_[i], 
                                                           shells_[j], 
                                                           shells_[k], 
                                                           shells_[l],
                                                           &num_integrals,
                                                           &perm);
            
            num_integrals_computed_++;
            
            // now, contract with appropriate density entries and sum into 
            // matrices
            
            double coulomb_int = density_.ref(k, l) * integral;
            double exchange_ik = density_.ref(j, l) * integral;
            double exchange_il = density_.ref(j, k) * integral;
            double exchange_jk = density_.ref(i, l) * integral;
            double exchange_jl = density_.ref(i, k) * integral;
          
          
            if (likely(k != l)) {
            
              coulomb_int *= 2;
            
            }
          
            ij_coulomb += coulomb_int;

            // don't overcount when indices are equal
            exchange_mat_.set(i, k, exchange_mat_.ref(i,k) + exchange_ik);
            if (likely(k != l)) {
              exchange_mat_.set(i, l, exchange_mat_.ref(i,l) + exchange_il);
            }
            if (likely(i != j)) {
              exchange_mat_.set(j, k, exchange_mat_.ref(j,k) + exchange_jk);
            }
            if (likely((k != l) && (i != j))) {
              exchange_mat_.set(j, l, exchange_mat_.ref(j,l) + exchange_jl);
            }

            
          } // for l
        
        } // for k
        
        coulomb_mat_.set(i, j, ij_coulomb);
        coulomb_mat_.set(j, i, ij_coulomb);
        
      } // for j
      
      
    
    } // for i
    
    la::Scale(0.5, &exchange_mat_);
    la::SubOverwrite(exchange_mat_, coulomb_mat_, &fock_mat_);
    
    fx_timer_stop(mod_, "naive_time");
    
    fx_result_int(mod_, "num_integrals_computed", num_integrals_computed_);
    
  } // ComputeFock()
  
  
  
  void UpdateDensity(const Matrix& new_density) {
    
    density_.CopyValues(new_density);
    //fock_mat_.Destruct();
    
    coulomb_mat_.SetZero();
    exchange_mat_.SetZero();
    
  }

  
  /**
   * Save the results to matrices for comparison.  Also saves fx results.
   */
  void OutputFock(Matrix* fock_out, Matrix* coulomb_out, Matrix* exchange_out) {
  
    if (fock_out) {
      fock_out->Copy(fock_mat_);
    }
    
    if (coulomb_out) {
      coulomb_out->Copy(coulomb_mat_);
    }
    
    if (exchange_out) {
      exchange_out->Copy(exchange_mat_);
    }
    
    // output results
  
  }
  
  void OutputCoulomb(Matrix* coulomb_out) {
   
    if (coulomb_out) {
      coulomb_out->Copy(coulomb_mat_); 
    }
    
  }

  void OutputExchange(Matrix* exchange_out) {
    
    if (exchange_out) {
      exchange_out->Copy(exchange_mat_);
    }
    
  }
  

}; // NaiveFock




#endif 

