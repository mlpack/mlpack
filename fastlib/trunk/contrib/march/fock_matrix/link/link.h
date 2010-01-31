#ifndef LINK_H
#define LINK_H

#include "fastlib/fastlib.h"
#include "contrib/march/fock_matrix/fock_impl/eri.h"
#include "contrib/march/fock_matrix/fock_impl/basis_shell.h"
#include "contrib/march/fock_matrix/fock_impl/shell_pair.h"

const fx_entry_doc link_entries[] = {
  {"shell_pair_cutoff", FX_PARAM, FX_DOUBLE, NULL, 
  "The threshold for a shell pair to be counted as \"significant\".\n"
  "Default: equal to the ERI threshold.\n" },
  {"thresh", FX_PARAM, FX_DOUBLE, NULL, 
   "The threshold to include an integral as significant.  Default: 10e-10.\n"},
  {"num_shell_pairs", FX_RESULT, FX_INT, NULL, 
   "The number of significant shell pairs."},
  {"num_shell_pairs_screened", FX_RESULT, FX_INT, NULL, 
    "The number of possible shell pairs neglected under the threshold.\n"},
  {"num_neglected_sigma", FX_RESULT, FX_INT, NULL, 
    "The total number of sigmas avoided due to sorted loops.\n"},
  {"num_neglected_nu", FX_RESULT, FX_INT, NULL, 
    "The total number of nus avoided due to the prescreening sorts.\n"},
  {"prescreening", FX_TIMER, FX_CUSTOM, NULL, 
    "The time taken to screen shell pairs.\n"},
  {"integrals", FX_TIMER, FX_CUSTOM, NULL, 
  "The time taken to screen and compute all the exchange integrals.\n"},
  {"LinK_time", FX_TIMER, FX_CUSTOM, NULL, 
  "The total time taken for the LinK algorithm, excluding init and output.\n"},
  {"num_integrals_computed", FX_RESULT, FX_INT, NULL, 
  "The total number of integrals computed.\n"},
{"N", FX_RESULT, FX_INT, NULL, 
"The total number of basis functions, as in the dimension of the Fock matrix.\n"},
  FX_ENTRY_DOC_DONE
}; 

const fx_module_doc link_mod_doc = {
  link_entries, NULL, "Algorithm module for LinK Exchange matrix method.\n"
};

class Link {

 private:

  fx_module* module_;

  Matrix basis_centers_;
  
  Vector basis_exponents_;
  
  Vector basis_momenta_;
  
  Matrix exchange_matrix_;
  
  Matrix density_matrix_;
  
  Vector shell_max_;
  
  // The list of shells, constructed from input
  ArrayList<BasisShell> shell_list_;
  
  // The list of shell pairs that meet the cutoff 
  ArrayList<ShellPair> shell_pair_list_;
  
  // significant_mu_pairs_[i] = list of significant nu for shell i
  BasisShell*** significant_nu_for_mu_;

  // num_significant_nu_for_mu_[i] = number of significant nu for mu
  ArrayList<index_t> num_significant_nu_for_mu_;
  
  // used to iterate over the sigmas belonging to a nu
  // needs to be sorted
  ShellPair*** significant_sigma_for_nu_;
  
  // used to index into above sorted list
  ArrayList<index_t> num_significant_sigma_for_nu_;
  
  
  // The length of shell_list_
  index_t num_shells_;
  
  // The number of shell pairs remaining after the initial Schwartz thresholding
  index_t num_shell_pairs_;
  
  // the number of basis functions, also the dimensionality of the density and
  // exchange matrices
  index_t num_functions_;
  
  // The threshold for a significant eri
  double threshold_;
  
  // The screening cutoff for 'significant' shell pairs
  // defaults to be the same as the four index threshold, since I think this is
  // what is done in QChem
  double shell_pair_cutoff_;
  
  // Used in sorting functions 
  index_t current_mu_;
  
  index_t num_neglected_sigma_;
  index_t num_neglected_nu_;
  
  index_t num_integrals_computed_;
  
  index_t num_insignificant_densities_;
  
  // used in the first call to UpdateDensity()
  bool first_computation;
  
  
  ////////////////////////// Functions ///////////////////////////////
 private:

  void PrescreeningLoop_();

 public:
  /**
   * Used for sorting the list of significant nu for each mu in order of 
   * density matrix and maximum Schwartz factor. 
   */
  static bool Prescreening_Sort_(BasisShell* ShellA, BasisShell* ShellB);
  
  /**
   * Used for sorting a list of shell pairs in order of schwartz factor
   */
  static bool ShellPairSort_(ShellPair* shellA, ShellPair* shellB);

  void Init(const Matrix& centers, const Matrix& exp, const Matrix& moment,  
            const Matrix& density_in, fx_module* mod) {
  
    module_ = mod;
    
    threshold_ = fx_param_double(mod, "thresh", 10e-10);
    
    basis_centers_.Copy(centers);
    
    DEBUG_ASSERT(exp.n_cols() == moment.n_cols());
    DEBUG_ASSERT(exp.n_cols() == basis_centers_.n_cols());
    
    basis_exponents_.Copy(exp.ptr(), basis_centers_.n_cols());
    basis_momenta_.Copy(moment.ptr(), basis_centers_.n_cols());
    
    density_matrix_.Copy(density_in);
    
    num_shells_ = basis_centers_.n_cols();
    //shell_list_.Init(num_shells_);
    
    // only works for s and p type functions
    num_functions_ = eri::CreateShells(basis_centers_, basis_exponents_, 
                                       basis_momenta_, &shell_list_);
    
    fx_result_int(module_, "N", num_functions_);
    
    if ((density_matrix_.n_cols() != num_functions_) || 
        (density_matrix_.n_rows() != num_functions_)) {
      
      FATAL("Density matrix must have correct dimensions.\n");
            
    } 
    
    shell_pair_cutoff_ = fx_param_double(module_, "shell_pair_cutoff", 
                                         threshold_);
    
    // change this to use array Lists
    significant_nu_for_mu_ = 
        (BasisShell***)malloc(num_shells_*sizeof(BasisShell**));
        
    significant_sigma_for_nu_ = 
        (ShellPair***)malloc(num_shells_*sizeof(ShellPair**));
        
    num_significant_nu_for_mu_.Init(num_shells_);
    
    num_significant_sigma_for_nu_.Init(num_shells_);
    
    exchange_matrix_.Init(num_functions_, num_functions_);
    exchange_matrix_.SetZero();
    
    num_neglected_sigma_ = 0;
    num_neglected_nu_ = 0;
    
    first_computation = true;
    
    num_integrals_computed_ = 0;
    num_insignificant_densities_ = 0;
     
  }
  
  void Destruct() {
    
    basis_centers_.Destruct();
    basis_centers_.Init(1,1);
    
    basis_exponents_.Destruct();
    basis_exponents_.Init(1);
    
    basis_momenta_.Destruct();
    basis_momenta_.Init(1);
    
    density_matrix_.Destruct();
    density_matrix_.Init(1,1);
    
    shell_list_.Clear();
    
    free(significant_nu_for_mu_);
    free(significant_sigma_for_nu_);
    
    exchange_matrix_.Destruct();
    exchange_matrix_.Init(1,1);
    
  } // Destruct()
    
  void Compute();
  
  void UpdateDensity(const Matrix& new_density);

  void OutputExchange(Matrix* exc_out);

}; // class Link






#endif