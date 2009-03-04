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

/**
 * 2 pi^(5/2)/(gamma_p * gamma_q * sqrt(gamma_p + gamma_q))
 */
double IntegralPrefactor(double gamma_p, double gamma_q) {

  double factor = 2 * pow(math::PI, 2.5);
  factor = factor/(gamma_p * gamma_q * sqrt(gamma_p + gamma_q));
  
  return factor;

}

double IntegralPrefactor(double alpha_A, double alpha_B, double alpha_C, 
                         double alpha_D) {

  double gamma_p = alpha_A + alpha_B;
  double gamma_q = alpha_C + alpha_D;
  
  return IntegralPrefactor(gamma_p, gamma_q);

}

double IntegralGPTFactor(double A_exp, double B_exp, double AB_dist_sq) {

  return (exp(-A_exp * B_exp * AB_dist_sq/(A_exp + B_exp)));

}

/**
 * Denoted K_1 or K_2 in the notes
 */
double IntegralGPTFactor(double A_exp, Vector& A_vec, 
                         double B_exp, Vector& B_vec) {

  double dist_sq = la::DistanceSqEuclidean(A_vec, B_vec);
  
  return IntegralGPTFactor(A_exp, B_exp, dist_sq);

}


double IntegralMomentumFactor(double alpha_A, double alpha_B, double alpha_C, 
                              double alpha_D, double four_way_dist) {

  double gamma_AB = alpha_A + alpha_B;
  double gamma_CD = alpha_C + alpha_D;
  
  return F_0_(four_way_dist * gamma_AB * gamma_CD/(gamma_AB + gamma_CD));

}

/**
 * Depends on the momentum of the four centers
 *
 * In the (ss|ss) case, this will just be the F_0 part
 */
double IntegralMomentumFactor(double gamma_AB, Vector& AB_center, 
                              double gamma_CD, Vector& CD_center) {

  double four_way_dist = la::DistanceSqEuclidean(AB_center, CD_center);

  return F_0_(four_way_dist * gamma_AB * gamma_CD/(gamma_AB + gamma_CD));

}

double IntegralMomentumFactor(double alpha_A,  Vector& A_vec, double alpha_B, 
                              Vector& B_vec, double alpha_C, 
                              Vector& C_vec, double alpha_D, 
                              Vector& D_vec) {

  
  Vector p_vec;
  double gamma_p = ComputeGPTCenter(A_vec, alpha_A, B_vec, 
                                    alpha_B, &p_vec);
  
  Vector q_vec;
  double gamma_q = ComputeGPTCenter(C_vec, alpha_C, D_vec, 
                                    alpha_D, &q_vec);
  
  
  return IntegralMomentumFactor(gamma_p, p_vec, gamma_q, q_vec);

}

double DistanceIntegral(double alpha_A, double alpha_B, double alpha_C, 
                        double alpha_D, double AB_dist, double CD_dist, 
                        double four_way_dist) {

  double ab_fac = IntegralGPTFactor(alpha_A, alpha_B, AB_dist);
  double cd_fac = IntegralGPTFactor(alpha_C, alpha_D, CD_dist);
  
  double prefac = IntegralPrefactor(alpha_A, alpha_B, alpha_C, alpha_D);
  
  double momentum = IntegralMomentumFactor(alpha_A, alpha_B, alpha_C, alpha_D, 
                                           four_way_dist);

  return (ab_fac * cd_fac * prefac * momentum);


} // DistanceIntegral ()



double SSSSIntegral(double alpha_A,  Vector& A_vec, double alpha_B, 
                    Vector& B_vec, double alpha_C, 
                    Vector& C_vec, double alpha_D, 
                    Vector& D_vec) {
  
  double integral = IntegralPrefactor(alpha_A, alpha_B, alpha_C, alpha_D);
  
  integral = integral * IntegralGPTFactor(alpha_A, A_vec, alpha_B, B_vec);
  integral = integral * IntegralGPTFactor(alpha_C, C_vec, alpha_D, D_vec);
  
  integral = integral * IntegralMomentumFactor(alpha_A, A_vec, alpha_B, B_vec, 
                                               alpha_C, C_vec, alpha_D, D_vec);
  
  return integral;

}

void CreateShells(const Matrix& centers, const Vector& exponents, 
                  const Vector& momenta, ArrayList<BasisShell>* shells_out) {
                  
  shells_out->Init(centers.n_cols());
  
  for (index_t i = 0; i < centers.n_cols(); i++) {
  
    Vector new_cent;
    centers.MakeColumnVector(i, &new_cent);
    
    (*shells_out)[i].Init(new_cent, exponents[i], momenta[i], i);
  
  } // for i
                  
} // CreateShells()


/**
 * Note, this only works on uncontracted, S-type integrals.  
 */
double ComputeShellIntegrals(BasisShell& mu_fun, BasisShell& nu_fun, 
                                   BasisShell& rho_fun, 
                                   BasisShell& sigma_fun) {
                          
                          
  double this_int;
  
  // Integral itself
  this_int = SSSSIntegral(mu_fun.exp(), mu_fun.center(), nu_fun.exp(), 
                               nu_fun.center(), rho_fun.exp(), rho_fun.center(), 
                               sigma_fun.exp(), sigma_fun.center());
                              
  //printf("mu norm: %g\n", mu_fun.normalization_constant());
                    
                                 
  // normalization
  this_int = this_int * mu_fun.normalization_constant();
  this_int = this_int * nu_fun.normalization_constant();
  this_int = this_int * rho_fun.normalization_constant();
  this_int = this_int * sigma_fun.normalization_constant();
  
  //printf("this_int: %g\n", this_int);
  
  return this_int;

}

double ComputeShellIntegrals(ShellPair& AB_shell, 
                             ShellPair& CD_shell) {
 
 /*
  return ComputeShellIntegrals(AB_shell.M_Shell(), AB_shell.N_Shell(), 
                               CD_shell.M_Shell(), CD_shell.N_Shell());
                             
   */
   
  // GPT factors
  double integral = AB_shell.integral_factor() * CD_shell.integral_factor();
  
  // prefactor
  integral = integral * IntegralPrefactor(AB_shell.M_Shell().exp(), 
                                          AB_shell.N_Shell().exp(), 
                                          CD_shell.M_Shell().exp(), 
                                          CD_shell.N_Shell().exp());                 
              
  // normalization   
  integral = integral * AB_shell.M_Shell().normalization_constant();
  integral = integral * AB_shell.N_Shell().normalization_constant();
  integral = integral * CD_shell.M_Shell().normalization_constant();
  integral = integral * CD_shell.N_Shell().normalization_constant();

  // momentum term
  integral = integral * IntegralMomentumFactor(AB_shell.exponent(), 
                                               AB_shell.center(), 
                                               CD_shell.exponent(), 
                                               CD_shell.center());

  return integral;

}

double SchwartzBound(BasisShell& i_shell, BasisShell& j_shell) {

  double bound = ComputeShellIntegrals(i_shell, j_shell, i_shell, j_shell);

  return sqrt(bound);

}


index_t ComputeShellPairs(ArrayList<ShellPair>* shell_pairs, 
                          ArrayList<BasisShell>& shells_in, 
                          double shell_pair_cutoff) {

  index_t num_shells = shells_in.size();
  
  index_t num_shell_pairs = 0;
  
  // What size to init to? 
  shell_pairs->Init();
  
  for (index_t i = 0; i < num_shells; i++) {
  
    BasisShell i_shell = shells_in[i];
    
    for (index_t j = i; j < num_shells; j++) {
    
      BasisShell j_shell = shells_in[j];
      
      // Do they use the overlap integral here?
      double this_bound = SchwartzBound(i_shell, j_shell);
      
      if (this_bound > shell_pair_cutoff) {
      
        shell_pairs->PushBack();
        
        (*shell_pairs)[num_shell_pairs].Init(i, j, i_shell, j_shell);
        (*shell_pairs)[num_shell_pairs].set_integral_upper_bound(this_bound);
        num_shell_pairs++;
        
              
      }
      
    } // for j
    
  } // for i 
  
  return num_shell_pairs;

}

// this version is for link
index_t ComputeShellPairs(ArrayList<ShellPair>* shell_pairs, 
                          ArrayList<BasisShell>& shells_in, 
                          double shell_pair_cutoff, Vector* shell_max, 
                          BasisShell*** sigma_for_nu, 
                          ArrayList<index_t>* num_per_shell) {
    
  index_t num_shells = shells_in.size();
  
  index_t num_shell_pairs = 0;
  
  //num_per_shell->Init(num_shells);
  
  // What size to init to? 
  //shell_pairs->Init(num_shells);
  shell_pairs->Init();

  DEBUG_ASSERT(shell_max != NULL);
  shell_max->Init(num_shells);
  
  for (index_t i = 0; i < num_shells; i++) {
    
    BasisShell i_shell = shells_in[i];
    index_t num_for_i = 0;
    
    
    double i_max = -DBL_MAX;
    
    ArrayList<index_t> significant_sig_index;
    significant_sig_index.Init(num_shells);
    
    for (index_t j = i; j < num_shells; j++) {
      
      BasisShell j_shell = shells_in[j];
      
      // Do they use the overlap integral here?
      double this_bound = SchwartzBound(i_shell, j_shell);
      
      if (this_bound > shell_pair_cutoff) {
        
        shell_pairs->PushBack();
        
        (*shell_pairs)[num_shell_pairs].Init(i, j, i_shell, j_shell);
        (*shell_pairs)[num_shell_pairs].set_integral_upper_bound(this_bound);
        num_shell_pairs++;
        
        significant_sig_index[num_for_i] = j;
        
        if (this_bound > i_max) {
          i_max = this_bound;
        }
        
        // track how many shell pairs have i as the first shell
        num_for_i++;
      
      } // if shell pair meets bound
      
      
    } // for j



    DEBUG_ASSERT(i_max > 0.0);
    (*shell_max)[i] = i_max;

    (*num_per_shell)[i] = num_for_i;
    
    sigma_for_nu[i] = (BasisShell**)malloc(num_for_i * sizeof(BasisShell**));

    for (index_t k = 0; k < num_for_i; k++) {
    
      sigma_for_nu[i][k] = shells_in.begin() + significant_sig_index[k];
    
    } // for k
    
  } // for i 
  
  return num_shell_pairs;
  
}




} // namespace eri


