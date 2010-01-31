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
    
    //printf("num_shells: %d\n", num_shells_);
    
    for (index_t i = 0; i < num_shells_; i++) {
    
      for (index_t j = 0; j <= i; j++) {
      
        Matrix coulomb_ij;
        coulomb_ij.Init(shells_[i].num_functions(), shells_[j].num_functions());
        coulomb_ij.SetZero();
      
        for (index_t k = 0; k < num_shells_; k++) {
        
          for (index_t l = 0; l <= k; l++) {
            
            IntegralTensor integrals;
            eri::ComputeShellIntegrals(shells_[i], shells_[j], 
                                       shells_[k], shells_[l], &integrals);
            //integrals.Print();
            
            num_integrals_computed_++;
            
            // now, contract with appropriate density entries and sum into 
            // matrices
            
            integrals.ContractCoulomb(shells_[k].matrix_indices(),
                                      shells_[l].matrix_indices(),
                                      density_, &coulomb_ij, (k == l));
            //printf("Coulomb integral: (%d, %d, %d, %d): %g\n", i, j, k, l, 
            //       integrals.ref(0, 0, 0, 0) * density_.ref(k, l));
            
            Matrix exchange_ik;
            exchange_ik.Init(shells_[i].num_functions(), 
                             shells_[k].num_functions());
            exchange_ik.SetZero();
            
            Matrix* exchange_jk;
            Matrix* exchange_il;
            Matrix* exchange_jl;
            
            if (i != j) {
              exchange_jk = new Matrix();
              exchange_jk->Init(shells_[j].num_functions(), 
                                shells_[k].num_functions());
              exchange_jk->SetZero();
            }
            else {
              exchange_jk = NULL;
            }
            
            if (l != k) {
              exchange_il = new Matrix();
              exchange_il->Init(shells_[i].num_functions(), 
                                shells_[l].num_functions());
              exchange_il->SetZero();
              
            }
            else {
              exchange_il = NULL;
            }
            
            if (i != j && l != k) {
              exchange_jl = new Matrix();
              exchange_jl->Init(shells_[j].num_functions(), 
                                shells_[l].num_functions());
              exchange_jl->SetZero();
              
            }
            else {
              exchange_jl = NULL; 
            }
            
            integrals.ContractExchange(shells_[i].matrix_indices(),
                                       shells_[j].matrix_indices(),
                                       shells_[k].matrix_indices(),
                                       shells_[l].matrix_indices(),
                                       density_, &exchange_ik, exchange_jk, 
                                       exchange_il, exchange_jl);
            
            eri::AddSubmatrix(shells_[i].matrix_indices(), 
                              shells_[k].matrix_indices(),
                              exchange_ik, &exchange_mat_);
            
            if (exchange_jk) {
              eri::AddSubmatrix(shells_[j].matrix_indices(), 
                                shells_[k].matrix_indices(),
                                *exchange_jk, &exchange_mat_);
	      delete exchange_jk;
            }
            
            if (exchange_il) {
              eri::AddSubmatrix(shells_[i].matrix_indices(), 
                                shells_[l].matrix_indices(),
                                *exchange_il, &exchange_mat_);
	      delete exchange_il;
	    }
            
            if (exchange_jl) {
              eri::AddSubmatrix(shells_[j].matrix_indices(), 
                                shells_[l].matrix_indices(),
                                *exchange_jl, &exchange_mat_);
	      delete exchange_jl;
	    }
                        
          } // for l
        
        } // for k
        
        // write coulomb into the global matrix
        //printf("adding to Coulomb: %d, %d, %g\n", shells_[i].matrix_index(0),
        //       shells_[j].matrix_index(0), coulomb_ij.ref(0,0));
        eri::AddSubmatrix(shells_[i].matrix_indices(), 
                          shells_[j].matrix_indices(),
                          coulomb_ij, &coulomb_mat_);
        
        if (i != j) {
          
          Matrix coulomb_ji;
          la::TransposeInit(coulomb_ij, &coulomb_ji);
          eri::AddSubmatrix(shells_[j].matrix_indices(), 
                            shells_[i].matrix_indices(),
                            coulomb_ji, &coulomb_mat_);
        }
        
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

