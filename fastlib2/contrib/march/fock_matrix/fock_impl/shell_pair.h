#ifndef SHELL_PAIR_H
#define SHELL_PAIR_H

#include "basis_shell.h"
#include "eri.h"

class ShellPair {

 public:

  ShellPair() {}
  
  ~ShellPair() {}
  
  void Init(index_t M_index, index_t N_index, BasisShell& M_shell, 
            BasisShell& N_shell);
            
            
  index_t M_index() {
    return M_index_;
  }
  index_t N_index() {
    return N_index_;
  }
  
  BasisShell& M_Shell() {
    return M_shell_;
  }
  
  BasisShell& N_Shell() {
    return N_shell_;
  }
  
  double integral_factor() {
    return integral_factor_;
  } 
  
  double exponent() {
    return exponent_;
  }
  
  Vector& center() {
    return center_;
  }
  
  
  void set_indices(index_t M_index, index_t N_index) {
  
    DEBUG_ASSERT(M_index <= N_index);
    M_index_ = M_index;
    N_index_ = N_index;
  
  } // set_indices()
  
  void set_integral_upper_bound(double bd) {
    integral_upper_bound_ = bd;
  } 
  
  double integral_upper_bound() {
    return integral_upper_bound_;
  }
  
  double schwartz_factor() {
    return schwartz_factor_;
  }

 private:

  // density matrix bounds
  double density_mat_upper_;
  double density_mat_lower_;
  

  double integral_upper_bound_;
  double integral_lower_bound_;
  
  // The indices of the two shells that make up the pair in the list of shells
  index_t M_index_;
  index_t N_index_;
  
  BasisShell M_shell_;
  BasisShell N_shell_;
  
  index_t num_prunes_;
  
  // for the integrals
  double integral_factor_;
  
  // gamma
  double exponent_;
  
  // the average center weighted by the bandwidths
  Vector center_;
  
  // the Schwartz factor
  double schwartz_factor_;

}; // class ShellPair



#endif