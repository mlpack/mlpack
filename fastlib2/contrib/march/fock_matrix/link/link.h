#ifndef LINK_H
#define LINK_H

#include "fastlib/fastlib.h"
#include "contrib/march/fock_matrix/fock_impl/eri.h"
#include "contrib/march/fock_matrix/fock_impl/basis_shell.h"
#include "contrib/march/fock_matrix/fock_impl/shell_pair.h"

const fx_entry_doc link_entries[] = {
  {"shell_pair_cutoff", FX_PARAM, FX_DOUBLE, NULL, 
  "The threshold for a shell pair to be counted as \"significant\".\n"
  "Default: equal to the ERI threshold.\n"},
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
  
  
  // num_shells_for_i_[j] = number of shell pairs for which j is the first 
  // index
  ArrayList<index_t> num_shells_for_i_;
  
  // first_nu_index_[i] = the first place in shell_pair_list_ that nu appears
  ArrayList<index_t> first_nu_index_;
  
  
  
  
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
    shell_list_.Init(num_shells_);
    
    // only works for s and p type functions
    num_functions_ = num_shells_ + (index_t)(2*la::Dot(basis_momenta_, 
                                                       basis_momenta_));
    
    if ((density_matrix_.n_cols() != num_functions_) || 
        (density_matrix_.n_rows() != num_functions_)) {
      
      FATAL("Density matrix must have correct dimensions.\n");
            
    } 
    
    // Change to use the code in eri
    // Fill in shell_list_
    for (index_t i = 0; i < num_shells_; i++) {
    
      Vector new_cent;
      basis_centers_.MakeColumnVector(i, &new_cent);
      
      shell_list_[i].Init(new_cent, basis_exponents_[i], basis_momenta_[i], i);
    
    } // for i
  
    shell_pair_cutoff_ = fx_param_double(module_, "shell_pair_cutoff", 
                                         threshold_);
                                         
    significant_nu_for_mu_ = 
        (BasisShell***)malloc(num_shells_*sizeof(BasisShell**));
        
    significant_sigma_for_nu_ = 
        (ShellPair***)malloc(num_shells_*sizeof(ShellPair**));
        
    num_significant_nu_for_mu_.Init(num_shells_);
    
    num_significant_sigma_for_nu_.Init(num_shells_);
    
    exchange_matrix_.Init(num_functions_, num_functions_);
    exchange_matrix_.SetZero();
     
  }
    
  void ComputeExchangeMatrix();

  void OutputExchangeMatrix(Matrix* exc_out);

}; // class Link






#endif