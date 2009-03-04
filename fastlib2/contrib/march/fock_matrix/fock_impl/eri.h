#ifndef ERI_H
#define ERI_H

#include "fastlib/fastlib.h"
#include "basis_shell.h"
#include "shell_pair.h"

class ShellPair;

namespace eri {

  // The erf-like function
  double F_0_(double z);
  
  double ComputeGPTCenter(Vector& A_vec, double alpha_A, Vector& B_vec, 
                          double alpha_B, Vector* p_vec);
                          
  double IntegralPrefactor(double alpha_A, double alpha_B, double alpha_C, 
                           double alpha_D);
                           
  double IntegralGPTFactor(double A_exp, double B_exp, double ab_dist_sq);
                           
  double IntegralGPTFactor(double A_exp, Vector& A_vec, 
                           double B_exp, Vector& B_vec);
                           
  double IntegralMomentumFactor(double alpha_A, double alpha_B, double alpha_C, 
                                double alpha_D, double four_way_dist);
                           
  double IntegralMomentumFactor(double gamma_AB, Vector& AB_center, 
                                double gamma_CD, Vector& CD_center);
                                
  double IntegralMomentumFactor(double alpha_A,  Vector& A_vec, double alpha_B, 
                                Vector& B_vec, double alpha_C, 
                                Vector& C_vec, double alpha_D, 
                                Vector& D_vec);
  
  // An ERI between four s-type gaussians with arbitrary bandwidth
  // This function does not currently normalize the gaussians
  double SSSSIntegral(double alpha_A,  Vector& A_vec, double alpha_B, 
                       Vector& B_vec, double alpha_C,  Vector& C_vec, 
                      double alpha_D,  Vector& D_vec);
                      
  // Compute the integral for general basis functions
  // I should add a shell version of this to take advantage of shell symmetry
/*  double ComputeIntegral(const BasisFunction& mu_fun, 
                         const BasisFunction& nu_fun, 
                         const BasisFunction& rho_fun, 
                         const BasisFunction& sigma_fun);
  */
                        
  // These won't really return doubles, they'll return lists of doubles
  double ComputeShellIntegrals(BasisShell& mu_fun, 
                               BasisShell& nu_fun, 
                               BasisShell& rho_fun, 
                               BasisShell& sigma_fun);
                               
  double ComputeShellIntegrals(ShellPair& AB_shell, 
                               ShellPair& CD_shell);
                          
 /**
  * Used for computing bounds in the multi-tree code
  * Should eventually take a momentum argument
  */
  double DistanceIntegral(double alpha_A, double alpha_B, double alpha_C, 
                          double alpha_D, double AB_dist, double CD_dist, 
                          double four_way_dist);
                               
  /**
   * Computes the Schwartz factor Q_{i j} = (i j|i j)^1/2
   */
  double SchwartzBound(BasisShell& i_shell, BasisShell& j_shell);
  
  
  /**
   * Forms the list of BasisShells from the centers, exponents, and momenta
   */
  void CreateShells(const Matrix& centers, const Vector& exponents, 
                    const Vector& momenta, ArrayList<BasisShell>* shells_out);
                            
  /**
   * Compute the list of significant shell pairs.
   *
   * Currently, shell pairs are screened by the size of their Schwartz factor, 
   * but some implementations may use overlap screening.   
   */
  index_t ComputeShellPairs(ArrayList<ShellPair>* shell_pairs, 
                            ArrayList<BasisShell>& shells_in, 
                            double shell_pair_cutoff);

  /**
   * Compute the list of significant shell pairs.  shell_max[i] is the 
   * largest Schwartz prescreening estimate for shell i.  This is used in the 
   * LinK algorithm. 
   *
   * Currently, shell pairs are screened by the size of their Schwartz factor, 
   * but some implementations may use overlap screening.   
   */
  index_t ComputeShellPairs(ArrayList<ShellPair>* shell_pairs, 
                            ArrayList<BasisShell>& shells_in, 
                            double shell_pair_cutoff, Vector* shell_max, 
                            BasisShell*** sigma_for_nu, 
                            ArrayList<index_t>* num_per_shell);
  

}

#endif
