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
 "Default: 0.0 (i.e. no shell pair screening.)\n"},
{"thresh", FX_PARAM, FX_DOUBLE, NULL, 
  "The threshold to include an integral as significant.  Default: 10e-10.\n"},
{"num_shell_pairs", FX_RESULT, FX_INT, NULL, 
  "The number of significant shell pairs."}, 
{"num_shell_pairs_screened", FX_RESULT, FX_INT, NULL,
  "The number of shell pairs that are screened.\n"},
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
  void ComputeFockMatrix(Matrix* fock_out, Matrix* coulomb_out, 
                         Matrix* exchange_out);
  
  void Init(const Matrix& cent, const Matrix& exp, const Matrix& mom, 
            const Matrix& density_in, fx_module* mod) {
  
    basis_centers_.Copy(cent);
    
    basis_exponents_.Copy(exp.ptr(), basis_centers_.n_cols());
    basis_momenta_.Copy(mom.ptr(), basis_centers_.n_cols());
    
    module_ = mod;

    threshold_ = fx_param_double(module_, "thresh", 10e-10);
    
    shell_pair_threshold_ = fx_param_double(module_, "shell_pair_threshold", 
                                            10e-10);
    
    // Set up shells here
    // This is correct even for higher momenta
    num_shells_ = basis_centers_.n_cols();
    basis_list_.Init(num_shells_);

    //printf("num_shells: %d\n", num_shells_);

    num_shell_pairs_ = 0;
    
    //shell_pair_list_.Init(num_shells_ + num_shells_*(num_shells_-1)/2);
    //shell_pair_list_.Init(num_shells_*num_shells_);
    
    num_prunes_ = 0;
    
    // Change this to take it as input
    density_matrix_.Copy(density_in);
    
    // Is this correct?
    matrix_size_ = density_matrix_.n_cols();
    
    coulomb_matrix_.Init(matrix_size_, matrix_size_);
    coulomb_matrix_.SetZero();
    exchange_matrix_.Init(matrix_size_, matrix_size_);
    exchange_matrix_.SetZero();

    //fock_matrix_.Init(matrix_size_, matrix_size_);
    
    
    for (index_t i = 0; i < num_shells_; i++) {
    
      Vector new_cent;
      basis_centers_.MakeColumnVector(i, &new_cent);
      
      basis_list_[i].Init(new_cent, basis_exponents_[i], 
                          (index_t)basis_momenta_[i], i);
    
    } // for i
  
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
  
  index_t num_prunes_;
  
  index_t matrix_size_;
  
  // The threshold for ignoring a shell quartet
  double threshold_;
  
  // The threshold for including a shell pair in further computation
  double shell_pair_threshold_;

  /**
   * The result needs to be multiplied by a density matrix bound
   *
   * Maybe the inputs should be shells somehow?  
   */
  double SchwartzBound_(BasisShell &mu, BasisShell &nu);
                        
  
  /**
   * Inner computation for Schwartz bound
   */
  double ComputeSchwartzIntegral_(BasisShell& mu, BasisShell& nu);
  
  


}; // class SchwartzPrescreening
