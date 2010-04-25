#ifndef CFMM_COULOMB_H
#define CFMM_COULOMB_H

#include "contrib/march/fock_matrix/fock_impl/eri.h"
#include "contrib/dongryel/fast_multipole_method/continuous_fmm.h"

const fx_entry_doc cfmm_mod_entries[] = {
  {"charge_thresh", FX_PARAM, FX_DOUBLE, NULL,
  "The screening threshold for including a charge distribution.\n"},
  {"multipole_computation", FX_TIMER, FX_CUSTOM, NULL,
    "Time to do the actual multipole computation, with no charge processing.\n"},
{"cfmm_time", FX_TIMER, FX_CUSTOM, NULL,
  "Total CFMM time (including charge pre- and post-processing.\n"},
{"num_shell_pairs", FX_RESULT, FX_INT, NULL,
  "Number of significant shell pairs.\n"},
{"num_shell_pairs_screened", FX_RESULT, FX_INT, NULL,
  "Number of insignificant shell pairs.\n"},
{"N", FX_RESULT, FX_INT, NULL, 
"The total number of basis functions, as in the dimension of the Fock matrix.\n"},
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
  
  // used with scf computations
  bool multipole_init_called_;
  
  ////////////////// Functions /////////////////////////////
  
  void ScreenCharges_();
  
  void MultipoleInit_();
  
  void MultipoleComputation_();
  
  double NaiveBaseCase_(Vector& q_col, double q_band, Vector& r_col, 
                        double r_band, double r_charge);
                        
  void NaiveComputation_();
  
  void MultipoleCleanup_();
  
  /**
   * Contract the density with the integral GPT factors and normalization
   * constants.
   */
  void ComputeCharges_();

 public:

  void Init(const Matrix& centers, const Matrix& exponents, 
            const Matrix& momenta, const Matrix& density, fx_module* mod_in) {
    
    mod_ = mod_in;
    
    centers_.Copy(centers);
    
    DEBUG_ASSERT(exponents.n_cols() == momenta.n_cols());
    DEBUG_ASSERT(exponents.n_cols() == centers_.n_cols());
    
    // this is for vectors - change to matrices?
    exponents_.Copy(exponents.ptr(), centers_.n_cols());
    momenta_.Copy(momenta.ptr(), centers_.n_cols());
    
    density_.Copy(density);
    
    
    num_funs_ = eri::CreateShells(centers_, exponents_, momenta_, &shells_);
    
    fx_result_int(mod_, "N", num_funs_);
    
    num_shells_ = shells_.size();
    
    // set charge_thresh_ from the module
    charge_thresh_ = fx_param_double(mod_, "charge_thresh", 10e-10);
    
    coulomb_mat_.Init(num_funs_, num_funs_);
    coulomb_mat_.SetZero();
    
    fx_timer_start(mod_, "cfmm_time");
    ScreenCharges_();
    fx_timer_stop(mod_, "cfmm_time");
    
    MultipoleInit_();
    
    multipole_init_called_ = true;
    
  } // Init()
  
  void Destruct() {
    
    centers_.Destruct();
    centers_.Init(1,1);
    
    exponents_.Destruct();
    exponents_.Init(1);
    
    momenta_.Destruct();
    momenta_.Init(1);
    
    density_.Destruct();
    density_.Init(1,1);
    
    shells_.Clear();
    
    coulomb_mat_.Destruct();
    coulomb_mat_.Init(1,1);
    
    // should try to clear up the multipole alg too, but probably not necessary
    
  } // Destruct()
  
  void UpdateDensity(const Matrix& new_density);
  
  void Compute();
  
  
  void OutputCoulomb(Matrix* coulomb_out);
  
  // only here to avoid compilation errors 
  // generates an error message if called
  void OutputExchange(Matrix* exchange_out) {
    FATAL("Tried to compute exchange matrix with CFMM.\n");
  }
  
}; // class CFMMCoulomb



#endif