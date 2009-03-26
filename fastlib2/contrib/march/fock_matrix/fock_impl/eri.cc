#include "eri.h"

namespace eri {


double double_factorial(int m) {
  
  DEBUG_ASSERT(m >= 0);

  double ret = (double)m;
  while (m > 1) {
    m = m - 2;
    ret = ret * (double)m;
  }
  
  return ret;

}

double F_0_(double z) {

  if (z == 0.0) {
    // double check that this shouldn't be 2/sqrt(pi)
    return 1.0;
  }
  else {
    return((1/sqrt(z)) * sqrt(math::PI) * 0.5 * erf(sqrt(z)));
  }
  
} // F_0_

// term1 = (m-1)!! sqrt(pi) erf(sqrt(t)) / (2^(m/2 + 1) * t^((m+1)/2)
// term2 = (exp(-t)/(2t)^(m/2)) * sum_{even n < m} ((2 t)^n (m-1)!!/(n+1)!!)
double F_m(double t, int m) {

  if (t == 0.0) {
    return 1/((double)(2*m + 1));
  }
  else {
  
    double term1 = double_factorial(2*m - 1) * sqrt(math::PI) * erf(sqrt(t));
    term1 = term1 / (pow(2, m+1) * pow(t, (double)((m+1)/2)));
    
    double term2 = 0.0;
    int n = 0;
    while (n < 2*m) {
      
      term2 += pow(2*t, n) * double_factorial(2*m - 1) / double_factorial(n+1);
      
    }
    
    term2 *= exp(-t)/(pow(2*t, m)); 

    return (term1 - term2);
     
  }

  

}


double BinomialCoeff(int pow, int mom1, int mom2, double dist1, double dist2) {

  DEBUG_ASSERT(pow >= 0);

  if (pow < 2) {
    return 1.0;
  }
  else {
    // only handles p integrals
    DEBUG_ASSERT(pow == 2);
    
    return dist1 + dist2;
    
  }
  
  // I think I'll need this to work for general momenta, since that's how 
  // kinetic integrals are defined 
  
  /*
  double coeff = 0.0;
  
  int q = min(-pow, pow - (2*mom2));
  
  while (q < min(pow, (2*mom1) - pow)) {
  
    int i = (pow + q) / 2;
    int j = (pow - q) / 2;
  
    coeff += choose(mom1, i) * choose(mom2, j) * pow(dist1, mom1-i) * pow(dist2, mom2-j);
  
    q += 2;
  
  }
  
  return coeff;
   */

}


////////////// Integrals //////////////////////////

double ComputeNormalization(BasisShell& shell) {

  return ComputeNormalization(shell.exp(), shell.total_momentum());

} // ComputeNormalization Basis Shell

double ComputeNormalization(double exp, index_t momentum) {

  double norm = pow(2/math::PI, 0.75); 
  
  double second_part = pow(2, momentum) * 
    pow(exp, ((2*momentum) + 3)*0.25);
  
  norm = norm * second_part;
  
  return norm;     

} // ComputeNormalization numbers


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

  double a_norm = ComputeNormalization(alpha_A, 0);
  double b_norm = ComputeNormalization(alpha_B, 0);
  double c_norm = ComputeNormalization(alpha_C, 0);
  double d_norm = ComputeNormalization(alpha_D, 0);

  double ab_fac = IntegralGPTFactor(alpha_A, alpha_B, AB_dist);
  double cd_fac = IntegralGPTFactor(alpha_C, alpha_D, CD_dist);
  
  double prefac = IntegralPrefactor(alpha_A, alpha_B, alpha_C, alpha_D);
  
  double momentum = IntegralMomentumFactor(alpha_A, alpha_B, alpha_C, alpha_D, 
                                           four_way_dist);

  return (a_norm * b_norm * c_norm * d_norm * ab_fac * cd_fac * prefac * 
          momentum);


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
        
        (*shell_pairs)[num_shell_pairs].Init(i, j, i_shell, j_shell, 
                                             num_shell_pairs);
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
                          ShellPair*** sigma_for_nu, 
                          ArrayList<index_t>* num_sigma_for_nu) {
    
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
        
        (*shell_pairs)[num_shell_pairs].Init(i, j, i_shell, j_shell, 
                                             num_shell_pairs);
        (*shell_pairs)[num_shell_pairs].set_integral_upper_bound(this_bound);
        significant_sig_index[num_for_i] = num_shell_pairs;
        num_shell_pairs++;
        
        
        if (this_bound > i_max) {
          i_max = this_bound;
        }
        
        // track how many shell pairs have i as the first shell
        num_for_i++;
      
      } // if shell pair meets bound
      
      
    } // for j



    DEBUG_ASSERT(i_max > 0.0);
    (*shell_max)[i] = i_max;

    (*num_sigma_for_nu)[i] = num_for_i;
    
    sigma_for_nu[i] = (ShellPair**)malloc(num_for_i * sizeof(ShellPair**));

    for (index_t k = 0; k < num_for_i; k++) {
    
      sigma_for_nu[i][k] = shell_pairs->begin() + significant_sig_index[k];
    
    } // for k
  
    // sort here?
        
  } // for i 
  
  
  
  return num_shell_pairs;
  
}

void TwoElectronIntegral(BasisShell& shellA, BasisShell& shellB, 
                         BasisShell& shellC, BasisShell& shellD, 
                         double**** integrals) {
                      
  // determine the total angular momentum
  index_t total_L = shellA.total_momentum() + shellB.total_momentum() 
                       + shellC.total_momentum() + shellD.total_momentum();
                      
  double alpha_A = shellA.exp();
  double alpha_B = shellB.exp();
  double alpha_C = shellC.exp();
  double alpha_D = shellD.exp();
  
  Vector A_vec = shellA.center();
  Vector B_vec = shellB.center();
  
  Vector p_vec;
  double gamma_p = ComputeGPTCenter(A_vec, alpha_A, B_vec, 
                                    alpha_B, &p_vec);
  
  Vector C_vec = shellC.center();
  Vector D_vec = shellD.center();
  
  Vector q_vec;
  double gamma_q = ComputeGPTCenter(C_vec, alpha_C, D_vec, 
                                    alpha_D, &q_vec);
                                    
  double F_arg = gamma_p * gamma_q;
  F_arg /= (gamma_p + gamma_q);
  
  double dist_sq = la::DistanceSqEuclidean(p_vec, q_vec);
  
  F_arg *= dist_sq;
  
  // compute the necessary values of F_m
  
  // how large an m do we need? depends on largest auxiliary integral
  // it is total_L plus the largest auxiliary integral
  // I think the largest auxiliary value is equal to the total momentum
  ArrayList<double> F_vals;
  F_vals.Init(2*total_L);
  
  for (int i = 0; i < 2*total_L; i++) {
    F_vals[i] = F_m(F_arg, i);
  }
  
  

  // Need to figure out how to code up the regression for integrals 

  // compute the intermediate integrals
  
  // make the last entry the auxiliary index
  // but I don't need all of these integrals, how to know which ones to compute?
  // make the last entry depend on how many of these are necessary
  // will need to keep up with how many in each step
  // include space for s integrals as well (make 3*momentum below)
  int num_intermediates_A = (3*shellA.total_momentum()) + 1;
  int num_intermediates_B = (3*shellB.total_momentum()) + 1;
  int num_intermediates_C = (3*shellC.total_momentum()) + 1;
  int num_intermediates_D = (3*shellD.total_momentum()) + 1;
  
  ArrayList<ArrayList<ArrayList<ArrayList<ArrayList<double> > > > > intermediate_ints;
  intermediate_ints.Init(num_intermediates_A);
  
  for (index_t i = 0; i < num_intermediates_A; i++) {
    intermediate_ints[i].Init(num_intermediates_B);
    
    for (index_t j = 0; j < num_intermediates_B; j++) {
    
      intermediate_ints[i][j].Init(num_intermediates_C);
      
      for (index_t k = 0; k < num_intermediates_C; k++) {
      
        intermediate_ints[i][j][k].Init(num_intermediates_D);
      
        for (index_t l = 0; l < num_intermediates_D; l++) {
        
          // the number of auxiliary integrals needed for this momentum
          // I think that each reduction of momentum requires an increase in m
          // so, the momentum of this integral subtracted from the target 
          // should give the number of auxiliaries
          int num_auxiliaries_needed;
        
          intermediate_ints[i][j][k][l].Init(num_auxiliaries_needed);
        
        }
      
      }
    
    }
    
  }
  
  // sum them into the needed integrals
  
  int num_integrals_A = (2*shellA.total_momentum()) + 1;
  int num_integrals_B = (2*shellB.total_momentum()) + 1;
  int num_integrals_C = (2*shellC.total_momentum()) + 1;
  int num_integrals_D = (2*shellD.total_momentum()) + 1;
  
  // allocate memory to store the output integrals
  // this only works for s and p shells
  // indexed by x, y, z (or is just s if only one entry)
  integrals = (double****)malloc(3 * num_integrals_A * sizeof(double***));
  for (index_t i = 0; i < num_integrals_A; i++) {
    
    integrals[i] = (double***)malloc(num_integrals_B * sizeof(double**));
  
    for (index_t j = 0; j < num_integrals_B; j++) {
      
      integrals[i][j] = (double**)malloc(num_integrals_C * sizeof(double*));
      
      for (index_t k = 0; k < num_integrals_C; k++) {
      
        integrals[i][j][k] = (double*)malloc(num_integrals_D * sizeof(double));
        
      }
      
    }
  
  }


  


  
} // TwoElectronIntegral





} // namespace eri


