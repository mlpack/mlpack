#include "shell_pair.h"

void ShellPair::Init(index_t M_index, index_t N_index, BasisShell& M_shell, 
                BasisShell& N_shell, index_t list_ind) {

  DEBUG_ASSERT(M_index <= N_index);
  M_index_ = M_index;
  N_index_ = N_index;
  
  M_shell_.Copy(M_shell);
  N_shell_.Copy(N_shell);
  
  integral_upper_bound_ = DBL_MAX;
  integral_lower_bound_ = -DBL_MAX;
  
  exponent_ = eri::ComputeGPTCenter(M_shell_.center(), M_shell_.exp(), 
                                    N_shell_.center(), N_shell_.exp(), 
                                    &center_);
  
  //integral_factor_ = eri::IntegralGPTFactor(M_shell_.exp(), M_shell_.center(), 
  //                                          N_shell_.exp(), N_shell_.center());
  
  list_index_ = list_ind;

}