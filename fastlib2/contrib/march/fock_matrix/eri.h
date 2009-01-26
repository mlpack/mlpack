#ifndef ERI_H
#define ERI_H

#include "fastlib/fastlib.h"

namespace eri {

  // The erf-like function
  double F_0_(double z);
  
  // An ERI between four s-type gaussians with arbitrary bandwidth
  // This function does not currently normalize the gaussians
  double SSSSIntegral(double alpha_A, Vector A_vec, double alpha_B, 
                      Vector B_vec, double alpha_C, Vector C_vec, 
                      double alpha_D, Vector D_vec);
                      
  
  // Compute the integral for general basis functions
  // I should add a shell version of this to take advantage of shell symmetry
  double ComputeIntegral(const BasisFunction& mu_fun, 
                         const BasisFunction& nu_fun, 
                         const BasisFunction& rho_fun, 
                         const BasisFunction& sigma_fun);
                        
  // These won't really return doubles, they'll return lists of doubles
  double ComputeShellIntegrals(const BasisShell& mu_fun, 
                               const BasisShell& nu_fun, 
                               const BasisShell& rho_fun, 
                               const BasisShell& sigma_fun);
                               
  double ComputeShellIntegrals(const ShellPair& AB_shell, 
                               const ShellPair& CD_shell);

}