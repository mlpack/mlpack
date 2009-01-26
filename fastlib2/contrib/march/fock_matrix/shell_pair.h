#ifndef SHELL_PAIR_H
#define SHELL_PAIR_H

#include "basis_shell.h"

class ShellPair {

 public:

  void ShellPair() {}
  
  void ~ShellPair() {}
  
  void Init(index_t M_index, index_t N_index, const BasisShell& M_shell, 
            const BasisShell& N_shell) {
  
    DEBUG_ASSERT(M_index <= N_index);
    M_index_ = M_index;
    N_index_ = N_index;
    
    M_shell_.Copy(M_shell);
    N_shell_.Copy(N_shell);
    
    integral_upper_bound_ = DBL_MAX;
    integral_lower_bound_ = -DBL_MAX;
    
  }
  
  index_t M_index() {
    return M_index_;
  }
  index_t N_index() {
    return N_index_;
  }
  
  
  void set_indices(index_t M_index, index_t N_index) {
  
    DEBUG_ASSERT(M_index <= N_index);
    M_index_ = M_index;
    N_index_ = N_index;
  
  } // set_indices()

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

}; // class ShellPair



#endif