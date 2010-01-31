/**
 * Prescreening with Schwartz bound
 */

#include "fastlib/fastlib.h"
#include "contrib/march/fock_matrix/fock_impl/eri.h"
#include "contrib/march/fock_matrix/fock_impl/basis_shell.h"
#include "contrib/march/fock_matrix/fock_impl/shell_pair.h"

const fx_entry_doc schwartz_entries[] = {
{"num_prunes", FX_RESULT, FX_INT, NULL, 
  "The number of integral computations pruned.\n"},
{"shell_pair_threshold", FX_PARAM, FX_DOUBLE, NULL, 
 "The threshold for a shell pair to be included.\n"
 "Default: same as integral threshold (i.e. no shell pair screening.)\n"},
{"thresh", FX_PARAM, FX_DOUBLE, NULL, 
  "The threshold to include an integral as significant.  Default: 10e-10.\n"},
{"num_shell_pairs", FX_RESULT, FX_INT, NULL, 
  "The number of significant shell pairs."}, 
{"num_shell_pairs_screened", FX_RESULT, FX_INT, NULL,
  "The number of shell pairs that are screened.\n"},
{"prescreening_time", FX_TIMER, FX_CUSTOM, NULL,
 "Total time for Schwartz prescreening.\n"},
{"shell_screening_time", FX_TIMER, FX_CUSTOM, NULL,
  "Time for selecting important shell pairs.\n"},
{"integral_time", FX_TIMER, FX_CUSTOM, NULL,
  "Time for screening and computing integrals.\n"},
{"num_integrals_computed", FX_RESULT, FX_INT, NULL,
  "The total number of integrals computed in all the iterations.\n"},
{"N", FX_RESULT, FX_INT, NULL, 
"The total number of basis functions, as in the dimension of the Fock matrix.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_module_doc prescreening_mod_doc = {
  schwartz_entries, NULL, "Algorithm module for schwartz prescreening.\n"
};


class SchwartzPrescreening {

 public:
  
  SchwartzPrescreening() {}
  
  ~SchwartzPrescreening() {}
  
  // Change this to match the format of the others
  void Compute();
  
  void OutputFock(Matrix* fock_out, Matrix* coulomb_out, Matrix* exchange_out);
  
  void OutputCoulomb(Matrix* coulomb_out);
  
  void OutputExchange(Matrix* exchange_out);
  
  void UpdateDensity(const Matrix& new_density);
  
  void Init(const Matrix& cent, const Matrix& exp, const Matrix& mom, 
            const Matrix& density_in, fx_module* mod) {
  
    basis_centers_.Copy(cent);
    
    basis_exponents_.Copy(exp.ptr(), basis_centers_.n_cols());
    basis_momenta_.Copy(mom.ptr(), basis_centers_.n_cols());
    
    module_ = mod;

    threshold_ = fx_param_double(module_, "thresh", 10e-10);
    
    shell_pair_threshold_ = fx_param_double(module_, "shell_pair_threshold", 
                                            threshold_);
    
    num_shells_ = basis_centers_.n_cols();
    num_functions_ = eri::CreateShells(basis_centers_, basis_exponents_, 
                                       basis_momenta_, &basis_list_);
    
    fx_result_int(module_, "N", num_functions_);
    
    num_shell_pairs_ = 0;
    
    //shell_pair_list_.Init(num_shells_ + num_shells_*(num_shells_-1)/2);
    //shell_pair_list_.Init(num_shells_*num_shells_);
    
    num_prunes_ = 0;
    
    // Change this to take it as input
    density_matrix_.Copy(density_in);
    
    // Is this correct?
    matrix_size_ = density_matrix_.n_cols();
    if (matrix_size_ != num_functions_) {
      FATAL("Density matrix size does not match basis.");
    }
    
    coulomb_matrix_.Init(matrix_size_, matrix_size_);
    coulomb_matrix_.SetZero();
    exchange_matrix_.Init(matrix_size_, matrix_size_);
    exchange_matrix_.SetZero();
    fock_matrix_.Init(matrix_size_, matrix_size_);
    fock_matrix_.SetZero();
    
    /*
    for (index_t i = 0; i < num_shells_; i++) {
    
      Vector new_cent;
      basis_centers_.MakeColumnVector(i, &new_cent);
      
      basis_list_[i].Init(new_cent, basis_exponents_[i], 
                          (index_t)basis_momenta_[i], i);
    
    } // for i
    */
    
    num_integrals_computed_ = 0;
    
    fx_timer_start(module_, "prescreening_time");
    
    printf("====Screening Shell Pairs====\n");
    fx_timer_start(module_, "shell_screening_time");
    num_shell_pairs_ = eri::ComputeShellPairs(&shell_pair_list_, basis_list_, 
                                              shell_pair_threshold_, 
                                              density_matrix_);
    
    printf("num_shells: %d\n", num_shells_);
    printf("num_shell_pairs: %d, shell_pair_threshold: %g\n", num_shell_pairs_,
           shell_pair_threshold_);
    fx_timer_stop(module_, "shell_screening_time");

    fx_result_int(module_, "num_shell_pairs", num_shell_pairs_);
    fx_result_int(module_, "num_shell_pairs_screened", 
                  num_shells_ * ((num_shells_ - 1) / 2) - num_shell_pairs_ + num_shells_);
    
    fx_timer_stop(module_, "prescreening_time");
    
    first_computation_ = true;
    
  } // Init()
  
  
  
 private:

  fx_module* module_;

  Matrix basis_centers_;
  Vector basis_exponents_;
  Vector basis_momenta_;
  
  
  // J
  Matrix coulomb_matrix_;
  // K
  Matrix exchange_matrix_;
  Matrix fock_matrix_;
  // D
  Matrix density_matrix_;

  // List of all basis shells
  ArrayList<BasisShell> basis_list_;
  
  ArrayList<ShellPair> shell_pair_list_;
  
  index_t num_shells_;
  index_t num_shell_pairs_;
  index_t num_functions_;
  
  index_t num_prunes_;
  
  index_t matrix_size_;
  
  index_t num_integrals_computed_;
  
  // used for resetting during SCF iterations
  bool first_computation_;
  
  // The threshold for ignoring a shell quartet
  double threshold_;
  
  // The threshold for including a shell pair in further computation
  double shell_pair_threshold_;

  /**
   * The result needs to be multiplied by a density matrix bound
   *
   * Maybe the inputs should be shells somehow?  
   */
  //double SchwartzBound_(BasisShell &mu, BasisShell &nu);
                        
  
  /**
   * Inner computation for Schwartz bound
   */
  double ComputeSchwartzIntegral_(BasisShell& mu, BasisShell& nu);
  


}; // class SchwartzPrescreening
