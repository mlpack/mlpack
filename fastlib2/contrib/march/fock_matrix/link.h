#ifndef LINK_H
#define LINK_H

#include "fastlib/fastlib.h"
#include "eri.h"
#include "basis_shell.h"
#include "shell_pair.h"

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
  
  Matrix density_mat_;
  
  Vector shell_max_;
  
  ArrayList<BasisShell> shell_list_;
  ArrayList<ShellPair> shell_pair_list_;
  
  // 
  //ArrayList<ArrayList> significant_mu_pairs_;
  //ArrayList<BasisShell> significant_nu_for_mu_;
  
  // significant_mu_pairs_[i] = list of significant nu for shell i
  BasisShell*** significant_mu_pairs_;
  
  
  // The length of shell_list_
  index_t num_shells_;
  
  // The number of shell pairs remaining after the initial Schwartz thresholding
  index_t num_shell_pairs_;
  
  // The threshold for a significant eri
  double threshold_;
  
  // The screening cutoff for 'significant' shell pairs
  // defaults to be the same as the four index threshold, since I think this is
  // what is done in QChem
  double shell_pair_cutoff_;
  
  // Used in sorting functions 
  index_t current_mu_;
  
  
  ////////////////////////// Functions ///////////////////////////////
  
  /**
   * Used for sorting the list of significant nu for each mu in order of 
   * density matrix and maximum Schwartz factor. 
   */
  bool Prescreening_Sort_(BasisShell* ShellA, BasisShell* ShellB);

 public:

  void Init(fx_module* mod, double thresh, const Matrix& centers, 
            const Vector& exp, const Vector& moment, const Matrix& density_in) {
  
    module_ = mod;
    
    threshold_ = thresh;
    
    basis_centers_.Copy(centers);
    basis_exponents_.Copy(exp);
    basis_momenta_.Copy(moment);
    density_mat_.Copy(density_in);
    
    num_shells_ = basis_centers_.n_cols();
    shell_list_.Init(num_shells_);
    
    // Fill in shell_list_
    for (index_t i = 0; i < num_shells_; i++) {
    
      Vector new_cent;
      basis_centers_.MakeColumnVector(i, &new_cent);
      
      shell_list_[i].Init(new_cent, basis_exponents_[i], basis_momenta_[i], i);
    
    } // for i
  
    shell_pair_cutoff_ = fx_param_double(module_, "shell_pair_cutoff", 
                                         threshold_);
                                         
    significant_mu_pairs_ = 
        (BasisShell***)malloc(num_shells_*sizeof(BasisShell**));
    
  
  }
    
  void ComputeFockMatrix();

  void OutputExchangeMatrix(Matrix* exc_out);

}; // class Link






#endif