#include "eri.h"

namespace eri {
double F_0_(double z) {

  if (z == 0) {
    return 1.0;
  }
  else {
    return((1/sqrt(z)) * sqrt(math::PI) * 0.5 * erf(sqrt(z)));
  }
  
} // F_0_

double ComputeGPTCenter(Vector& A_vec, double alpha_A, Vector& B_vec, 
                      double alpha_B, Vector* p_vec) {
 
  double gamma = alpha_A + alpha_B;
  
  Vector A_vec_scaled;
  Vector B_vec_scaled;
  
  la::ScaleInit(alpha_A, A_vec, &A_vec_scaled);
  la::ScaleInit(alpha_B, B_vec, &B_vec_scaled);
  
  la::AddInit(A_vec_scaled, B_vec_scaled, p_vec);
  la::Scale(1/gamma, p_vec);
  
  return gamma;
                                                                
}



double SSSSIntegral(double alpha_A,  Vector& A_vec, double alpha_B, 
                          Vector& B_vec, double alpha_C, 
                          Vector& C_vec, double alpha_D, 
                          Vector& D_vec) {

  double gamma_p = alpha_A + alpha_B;
  double gamma_q = alpha_C + alpha_D;
  
  //printf("gamma_p:%g\n", gamma_p);
  //printf("gamma_q:%g\n", gamma_q);

  double integral = 2*pow(math::PI, 2.5)/
                          (gamma_p*gamma_q*sqrt(gamma_p + gamma_q));
                          
  //printf("constant: %g\n", integral);

  double AB_dist = la::DistanceSqEuclidean(A_vec, B_vec);
  double CD_dist = la::DistanceSqEuclidean(C_vec, D_vec);

  
  Vector A_vec_scaled;
  la::ScaleInit(alpha_A, A_vec, &A_vec_scaled);
  Vector B_vec_scaled;
  la::ScaleInit(alpha_B, B_vec, &B_vec_scaled);
  Vector C_vec_scaled;
  la::ScaleInit(alpha_C, C_vec, &C_vec_scaled);
  Vector D_vec_scaled;
  la::ScaleInit(alpha_D, D_vec, &D_vec_scaled);

  
  Vector AB_vec;
  la::AddInit(A_vec_scaled, B_vec_scaled, &AB_vec);
  la::Scale(1/gamma_p, &AB_vec);
  Vector CD_vec;
  la::AddInit(C_vec_scaled, D_vec_scaled, &CD_vec);
  la::Scale(1/gamma_q, &CD_vec);

  double four_way_dist = la::DistanceSqEuclidean(AB_vec, CD_vec);
  
  double four_way_part = F_0_(four_way_dist * gamma_p * gamma_q/(gamma_p + gamma_q));
  
  integral = integral * four_way_part;

  double K1 = exp(-alpha_A * alpha_B * AB_dist/gamma_p);
  double K2 = exp(-alpha_C * alpha_D * CD_dist/gamma_q);
  
  integral = integral * K1 * K2;
  
  /*
  printf("K1: %g\n", K1);
  printf("K2: %g\n", K2);
  printf("four way part %g\n", four_way_part);
  */
  
  return integral;

}

/**
 * Note, this only works on uncontracted, S-type integrals.  
 */
double ComputeShellIntegrals(BasisShell& mu_fun, BasisShell& nu_fun, 
                                   BasisShell& rho_fun, 
                                   BasisShell& sigma_fun) {
                          
                          
  double this_int;
  
  this_int = SSSSIntegral(mu_fun.exp(), mu_fun.center(), nu_fun.exp(), 
                               nu_fun.center(), rho_fun.exp(), rho_fun.center(), 
                               sigma_fun.exp(), sigma_fun.center());
                              
  //printf("mu norm: %g\n", mu_fun.normalization_constant());
                                 
  this_int = this_int * mu_fun.normalization_constant();
  this_int = this_int * nu_fun.normalization_constant();
  this_int = this_int * rho_fun.normalization_constant();
  this_int = this_int * sigma_fun.normalization_constant();
  
  //printf("this_int: %g\n", this_int);
  
  return this_int;

}

double ComputeShellIntegrals(ShellPair& AB_shell, 
                             ShellPair& CD_shell) {
 
  return ComputeShellIntegrals(AB_shell.M_Shell(), AB_shell.N_Shell(), 
                               CD_shell.M_Shell(), CD_shell.N_Shell());
                             
                                                                                     
}

double SchwartzBound(BasisShell& i_shell, BasisShell& j_shell) {

  double bound = ComputeShellIntegrals(i_shell, j_shell, i_shell, j_shell);

  return sqrt(bound);

}

// I need to order these by size of integral estimate
index_t ComputeShellPairs(ArrayList<ShellPair>* shell_pairs, 
                          ArrayList<BasisShell>& shells_in, 
                          double shell_pair_cutoff) {

  // The size to increase the list by when necessary
  index_t num_to_add = 20;

  index_t num_shells = shells_in.size();
  
  index_t num_shell_pairs = 0;
  
  // What size to init to? 
  shell_pairs->Init(num_shells);
  
  for (index_t i = 0; i < num_shells; i++) {
  
    BasisShell i_shell = shells_in[i];
    
    for (index_t j = i; j < num_shells; j++) {
    
      BasisShell j_shell = shells_in[j];
      
      // Do they use the overlap integral here?
      double this_bound = SchwartzBound(i_shell, j_shell);
      
      if (this_bound > shell_pair_cutoff) {
      
        if (shell_pairs->size() <= num_shell_pairs) {
          shell_pairs->PushBack(num_to_add);
        }
        
        (*shell_pairs)[num_shell_pairs].Init(i, j, i_shell, j_shell);
        (*shell_pairs)[num_shell_pairs].set_integral_upper_bound(this_bound);
        num_shell_pairs++;
              
      }
      /*      else {
        printf("pruned\n");
	}*/
      
    } // for j
    
  } // for i 
  
  return num_shell_pairs;

}

index_t ComputeShellPairs(ArrayList<ShellPair>* shell_pairs, 
                          ArrayList<BasisShell>& shells_in, 
                          double shell_pair_cutoff, Vector* shell_max) {
  
  // The size to increase the list by when necessary
  index_t num_to_add = 20;
  
  index_t num_shells = shells_in.size();
  
  index_t num_shell_pairs = 0;
  
  // What size to init to? 
  shell_pairs->Init(num_shells);

  DEBUG_ASSERT(shell_max != NULL);
  shell_max->Init(num_shells);
  
  for (index_t i = 0; i < num_shells; i++) {
    
    BasisShell i_shell = shells_in[i];
    
    double i_max = -DBL_MAX;
    
    for (index_t j = i; j < num_shells; j++) {
      
      BasisShell j_shell = shells_in[j];
      
      // Do they use the overlap integral here?
      double this_bound = SchwartzBound(i_shell, j_shell);
      
      if (this_bound > shell_pair_cutoff) {
        
        if (shell_pairs->capacity() <= num_shell_pairs) {
          shell_pairs->PushBack(num_to_add);
        }
        
        (*shell_pairs)[num_shell_pairs].Init(i, j, i_shell, j_shell);
        (*shell_pairs)[num_shell_pairs].set_integral_upper_bound(this_bound);
        num_shell_pairs++;
        
        if (this_bound > i_max) {
          i_max = this_bound;
        }
      
      }
      
    } // for j

    DEBUG_ASSERT(i_max > 0.0);
    (*shell_max)[i] = i_max;
    
  } // for i 
  
  return num_shell_pairs;
  
}





} // namespace eri












