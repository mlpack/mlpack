/**
 * Prescreening with Schwartz bound
 */

#include "schwartz_prescreening.h"

double SchwartzPrescreening::SchwartzBound_(const BasisShell &mu, 
                                            const BasisShell &nu) {
                                            
  double Q_mu_nu = -DBL_MAX;
  for (index_t i = 0; i < mu.num_functions(); i++) {
  
    for (index_t j = 0; j < nu.num_functions(); j++) {
    
      double this_Q = ComputeSchwartzIntegral_();
      
      if (this_Q > Q_mu_nu) {
        
        Q_mu_nu = this_Q;
        
      }
    
    }
  
  }
  
  return math::sqrt(Q_mu_nu * Q_rho_sigma);

} // SchwartzBound_()

double SchwartzPrescreening::ComputeSchwartzIntegral_() {

  


} // ComputeSchwartzIntegral_()


void SchwartzPrescreening::ComputeFockMatrix() {

  

  // form shell-pairs
  for (index_t i = 0; i < num_shells_; i++) {
  
    BasisShell i_shell;
  
    for (index_t j = i; j < num_shells_; j++) {
    
      BasisShell j_shell;
      
      ShellPair new_pair;
      new_pair.Init();
      // add to list of pairs
      
      num_shells_++;
      
      double this_bound = SchwartzBound_(i_shell, j_shell);
      
      new_pair.set_integral_upper_bound(this_bound);
      // set density bound too
      
    } // for j
  
  } //for i
  
  
  // Is there symmetry I can use here?
  for (index_t i = 0; i < num_shells_; i++) {
  
    ShellPair i_pair;
  
    for (index_t j = 0; j < num_shells_; j++) {
    
      ShellPair j_pair;

      // this can be made tighter by considering other density matrix entries
      double density_bound = max(i_pair.density_mat_upper(), j_pair.density_mat_upper());
    
      double this_est = i_pair.integral_upper_bound() * 
          j_pair.integral_upper_bound() * density_bound;
          
      if (this_est > threshold) {
      
        double this_int = eri::ComputeShellIntegrals(i_pair, j_pair);
        // add to fock matrix
      
      } 
      else {
        num_prunes_++;
      }
    
    }
    
  } // for i
  
  
  

} // ComputeFockMatrix()





