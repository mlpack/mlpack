/**
 * Prescreening with Schwartz bound
 */

#include "schwartz_prescreening.h"

double SchwartzPrescreening::SchwartzBound_(BasisShell &mu, 
                                             BasisShell &nu) {
                                 
  index_t i_funs = mu.num_functions();
                                                                  
  double Q_mu_nu = -DBL_MAX;
  for (index_t i = 0; i < i_funs; i++) {
  
    for (index_t j = 0; j < nu.num_functions(); j++) {
    
      double this_Q = ComputeSchwartzIntegral_();
      
      if (this_Q > Q_mu_nu) {
        
        Q_mu_nu = this_Q;
        
      }
    
    }
  
  }
  
  return sqrt(Q_mu_nu);

} // SchwartzBound_()

double SchwartzPrescreening::ComputeSchwartzIntegral_() {

  
  return 0.0;

} // ComputeSchwartzIntegral_()


void SchwartzPrescreening::ComputeFockMatrix(Matrix* fock_out) {

  // form shell-pairs
  for (index_t i = 0; i < num_shells_; i++) {
  
    BasisShell i_shell = basis_list_[i];
  
    for (index_t j = i; j < num_shells_; j++) {
    
      BasisShell j_shell = basis_list_[j];
      num_shell_pairs_++;
      
      shell_pair_list_[num_shell_pairs_].Init(i, j, i_shell, j_shell);
      
      double this_bound = SchwartzBound_(i_shell, j_shell);
      
      shell_pair_list_[num_shell_pairs_].set_integral_upper_bound(this_bound);
      // set density bound too
      
    } // for j
  
  } //for i
  
  
  // Is there symmetry I can use here?
  for (index_t i = 0; i < num_shells_; i++) {
  
    ShellPair& i_pair = shell_pair_list_[i];
  
    for (index_t j = 0; j < num_shells_; j++) {
    
      ShellPair& j_pair = shell_pair_list_[j];

      // this can be made tighter by considering other density matrix entries
      //double density_bound = max(i_pair.density_mat_upper(), 
      //                           j_pair.density_mat_upper());
      double density_bound = 1.0;
    
      double this_est = i_pair.integral_upper_bound() * 
          j_pair.integral_upper_bound() * density_bound;
          
      if (this_est > threshold_) {
      
        double this_int = eri::ComputeShellIntegrals(i_pair, j_pair);
        
        // extend this for general case
        index_t mind = i_pair.M_index();
        index_t nind = j_pair.N_index();
        
        double cval = coulomb_matrix_.get(mind, nind);
        cval = cval + this_int;
        double eval = exchange_matrix_.get(mind, nind);
        eval = eval - 0.5*this_int;
        
        coulomb_matrix_.set(mind, nind, cval); 
        exchange_matrix_.set(mind, nind, eval);
              
      } 
      else {
        num_prunes_++;
      }
    
    }
    
  } // for i
  
  la::AddOverwrite(coulomb_matrix_, exchange_matrix_, &fock_matrix_);
  
  fock_out->Copy(fock_matrix_);

  printf("num_prunes: %d\n", num_prunes_);

} // ComputeFockMatrix()





