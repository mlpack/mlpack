#ifndef CFMM_COULOMB_H
#define CFMM_COULOMB_H

#include "contrib/march/fock_matrix/fock_impl/eri.h"
#include "contrib/dongryel/fast_multipole_method/continuous_fmm.h"

const fx_entry_doc cfmm_mod_entries[] = {
  {"charge_thresh", FX_PARAM, FX_DOUBLE, NULL,
  "The screening threshold for including a charge distribution.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc cfmm_mod_submodules[] = {
  {"multipole_cfmm", &cfmm_fmm_mod_doc,
   "Parameters and results for the multipole portion of the code.\n"},
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc cfmm_mod_doc = {
  cfmm_mod_entries, cfmm_mod_submodules,
  "Algorithm module for CFMM.\n"
};

class CFMMCoulomb {

 public:


 private:

  Matrix centers_;
  
  // should these really be Matrices?
  Vector exponents_;
  Vector momenta_;
  
  Matrix density_;
  
  Matrix charge_centers_;
  Matrix charge_exponents_;
  
  // The density contracted with the integral prefactors
  Matrix charges_;
  
  fx_module* mod_;
  
  // The basis shell list
  ArrayList<BasisShell> shells_;
  
  ArrayList<ShellPair> shell_pairs_;
  
  index_t num_shells_;
  index_t num_funs_;
  index_t num_shell_pairs_;
  
  // the output
  Matrix coulomb_mat_;
  
  // the screening factor for charge distributions
  double charge_thresh_;
  
  Vector multipole_output_;
  
  ContinuousFmm cfmm_algorithm_;
  
  ////////////////// Functions /////////////////////////////
  
  void ScreenCharges_();
  
  void MultipoleInit_();
  
  void MultipoleComputation_();
  
  double NaiveBaseCase_(Vector& q_col, double q_band, Vector& r_col, 
                        double r_band, double r_charge);
                        
  void NaiveComputation_();
  
  void MultipoleCleanup_();

 public:

  void Init(const Matrix& centers, const Matrix& exponents, 
            const Matrix& momenta, const Matrix& density, fx_module* mod_in) {
    
    centers_.Copy(centers);
    
    DEBUG_ASSERT(exponents.n_cols() == momenta.n_cols());
    DEBUG_ASSERT(exponents.n_cols() == centers_.n_cols());
    
    // this is for vectors - change to matrices?
    exponents_.Copy(exponents.ptr(), centers_.n_cols());
    momenta_.Copy(momenta.ptr(), centers_.n_cols());
    
    density_.Copy(density);
    
    
    // This only works for s and p type functions
    num_funs_ = centers_.n_cols() + (index_t)2*la::Dot(momenta_, momenta_);
    
    eri::CreateShells(centers_, exponents_, momenta_, &shells_);
    
    num_shells_ = shells_.size();
    
    mod_ = mod_in;
    
    // set charge_thresh_ from the module
    charge_thresh_ = fx_param_double(mod_, "charge_thresh", 10e-10);
    
    
    
  } // Init()
  
  void ComputeCoulomb();
  
  
  void Output(Matrix* coulomb_out);
  
}; // class CFMMCoulomb



#endif