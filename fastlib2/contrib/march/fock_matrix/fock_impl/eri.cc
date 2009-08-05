#include "eri.h"
#include "oeints.h"

namespace eri {

  ///////////////////////// constants /////////////////////////////////
  
  /**
   * Stores the quantities n!! needed for integrals
   */
  double double_factorial[MAX_FAC];
  
  ///////////////////////// initialization ///////////////////////////////
  
  void ERIInit() {
    
    init_libint_base();
    
    //double_factorial = (double*)malloc(num_factorials * sizeof(double));
    
    double_factorial[0] = 1.0;
    double_factorial[1] = 1.0;
    for (index_t i = 2; i < MAX_FAC; i++) {
      
      double_factorial[i] = (double)i * double_factorial[i-2];
      
    }
    
  }
  
  void ERIFree() {
    
    //free(double_factorial);
    
  }

  
  ////////////// Helpers ///////////////////////////

  index_t NumFunctions(int momentum) {
    
    if (momentum == 0) {
      return 1;
    }
    else if (momentum == 1) {
      return 3;
    }
    else if (momentum == 2) {
      return 6;
    }
    else if (momentum == 3) {
      return 10;
    }
    else {
      FATAL("Momenta higher than 3 not yet supported.");
    }
    
  }  // NumFunctions
  

  void ArrayListSwap(index_t ind1, index_t ind2, ArrayList<index_t>* perm) {
    
    DEBUG_ASSERT(ind1 < perm->size());
    DEBUG_ASSERT(ind2 < perm->size());
    
    DEBUG_ASSERT(ind1 >= 0);
    DEBUG_ASSERT(ind2 >= 0);
    
    index_t ind2_temp = (*perm)[ind2];
    (*perm)[ind2] = (*perm)[ind1];
    (*perm)[ind1] = ind2_temp;
    
  } // ArrayListSwap
  
  
  // a_ind needs to be the first index after the permutation for 
  index_t IntegralIndex(ArrayList<index_t> indices, ArrayList<index_t> momenta) {
    
    index_t a_ind = indices[0];
    index_t b_ind = indices[1];
    index_t c_ind = indices[2];
    index_t d_ind = indices[3];
    
    index_t B_mom = momenta[1];
    index_t C_mom = momenta[2];
    index_t D_mom = momenta[3];
    
    index_t numB = NumFunctions(B_mom);
    index_t numC = NumFunctions(C_mom);
    index_t numD = NumFunctions(D_mom);
    
    
    index_t result = ((a_ind * numB + b_ind) * numC + c_ind) * numD + d_ind;
    
    return result;
    
  } // IntegralIndex
  
  index_t IntegralIndex(index_t a_ind, int A_mom, index_t b_ind, int B_mom,
                        index_t c_ind, int C_mom, index_t d_ind, int D_mom) {
    
    index_t numB = NumFunctions(B_mom);
    index_t numC = NumFunctions(C_mom);
    index_t numD = NumFunctions(D_mom);
    
    index_t result = ((a_ind * numB + b_ind) * numC + c_ind) * numD + d_ind;
    
    return result;
    
  } // IntegralIndex
  
   
  
  double ComputeNormalization(double exp, int x_mom, int y_mom, int z_mom) {
    
    int total_momentum = x_mom + y_mom + z_mom;
    
    // should be able to do this more efficiently
    double result = pow(2.0/math::PI, 0.75);
    
    result *= pow(2.0, total_momentum); 
    result *= pow(exp, ((2.0 * total_momentum + 3.0)/4.0));
    
    double sqrt_fac = 1.0;
    if (x_mom > 0) {
      sqrt_fac *= double_factorial[2 * x_mom - 1];
    }
    if (y_mom > 0) {
      sqrt_fac *= double_factorial[2 * y_mom - 1];
    }
    if (z_mom > 0) {
      sqrt_fac *= double_factorial[2 * z_mom - 1];
    }
    
    result /= sqrt(sqrt_fac);
    
    return result;
    
  } // ComputeNormalization
  
  
  double ComputeGPTCenter(const Vector& A_vec, double alpha_A, const Vector& B_vec, 
                          double alpha_B, Vector* p_vec) {
    
    double gamma = alpha_A + alpha_B;
    
    la::ScaleInit(alpha_A, A_vec, p_vec);
    la::AddExpert(3, alpha_B, B_vec.ptr(), p_vec->ptr());
    la::Scale(1/gamma, p_vec);
    
    return gamma;
    
  }  
  
  double ComputeShellOverlap(const BasisShell& shellA, const BasisShell& shellB) {
    
    double gamma = shellA.exp() + shellB.exp();
    double overlap = pow(math::PI/gamma, 1.5);
    double dist_sq = la::DistanceSqEuclidean(shellA.center(), shellB.center());
    double exp_fac = exp(-1 * shellA.exp() * shellB.exp() * dist_sq / gamma);
    overlap *= exp_fac;
    
    return overlap;
    
  } // ComputeShellOverlap


  void Compute_F(double* F, int n, double t) {
    
    // copied from eri.cc in libmint
    double EPS = 1.0e-17;
    
    int i, m, k;
    int m2;
    double t2;
    double num;
    double sum;
    double term1, term2;
    static double K = 1.0/M_2_SQRTPI;
    double et;
    
    // forward recurrence?
    if (t>20.0){
      t2 = 2*t;
      et = exp(-t);
      t = sqrt(t);
      F[0] = K*erf(t)/t;
      for(m=0; m<=n-1; m++){
        F[m+1] = ((2*m + 1)*F[m] - et)/(t2);
      }
    }
    // backward?
    else {
      et = exp(-t);
      t2 = 2*t;
      m2 = 2*n;
      num = double_factorial[m2];
      i=0;
      sum = 1.0/(m2+1);
      do{
        i++;
        num = num*t2;
        term1 = num/double_factorial[m2+2*i+2];
        sum += term1;
      } while (fabs(term1) > EPS && i < MAX_FAC);
      F[n] = sum*et;
      for(m=n-1;m>=0;m--){
        F[m] = (t2*F[m+1] + et)/(2*m+1);
      }
    }
    
  } // Compute_F
  
  // IMPORTANT: currently assumes the rows and columns are contiguous
  void AddSubmatrix(const ArrayList<index_t>& rows,
                    const ArrayList<index_t>& cols,
                    const Matrix& submat, Matrix* out_mat) {
    
    Matrix col_slice;
    out_mat->MakeColumnSlice(cols[0], cols.size(), &col_slice);
    
    for (index_t i = 0; i < cols.size(); i++) {
      // for each column, make the subvector and do the addition
      
      Vector out_row;
      col_slice.MakeColumnSubvector(i, rows[0], rows.size(), &out_row);
      
      Vector in_row;
      submat.MakeColumnVector(i, &in_row);
      
      la::AddTo(in_row, &out_row);
      
    }
    
  } // AddSubmatrix
  

  ////////////// External Integrals //////////////////////////

void ComputeShellIntegrals(BasisShell& mu_fun, BasisShell& nu_fun, 
                           BasisShell& rho_fun, BasisShell& sigma_fun,
                           IntegralTensor* integrals) {

  
  ArrayList<BasisShell*> shells;
  shells.Init(4);
  shells[0] = &mu_fun;
  shells[1] = &nu_fun;
  shells[2] = &rho_fun;
  shells[3] = &sigma_fun;
  
  double overlapAB = ComputeShellOverlap(mu_fun, nu_fun);
  double overlapCD = ComputeShellOverlap(rho_fun, sigma_fun);
  
  ComputeERI(shells, overlapAB, overlapCD, integrals);
  
}

void ComputeShellIntegrals(ShellPair& AB_shell, ShellPair& CD_shell,
                           IntegralTensor* integrals) {
 

  ArrayList<BasisShell*> shells;
  shells.Init(4);
  shells[0] = &(AB_shell.M_Shell());
  shells[1] = &(AB_shell.N_Shell());
  shells[2] = &(CD_shell.M_Shell());
  shells[3] = &(CD_shell.N_Shell());
  
  
  ComputeERI(shells, AB_shell.overlap(), CD_shell.overlap(), integrals);
  
}

double SchwartzBound(BasisShell& i_shell, BasisShell& j_shell) {

  IntegralTensor bound;
  ComputeShellIntegrals(i_shell, j_shell, i_shell, j_shell, &bound);

  double* max_bound = std::max_element(bound.ptr(), 
                                       bound.ptr() + bound.num_integrals());
  double* min_bound = std::min_element(bound.ptr(), 
                                       bound.ptr() + bound.num_integrals());
  
  return max(fabs(*max_bound), fabs(*min_bound));

}

  ////////////////// Internal Integrals ///////////////////////
  
  void ComputeERI(const ArrayList<BasisShell*>& shells, double overlapAB, 
                  double overlapCD, IntegralTensor* integrals) {
    
    // should this come in initialized or be initialized here
    // this is new from old
    ArrayList<index_t> perm;
    perm.Init(4);
    perm[0] = 0;
    perm[1] = 1;
    perm[2] = 2;
    perm[3] = 3;
    
        
    // determine the permutation
    
    index_t momA = shells[0]->total_momentum();
    index_t momB = shells[1]->total_momentum();
    index_t momC = shells[2]->total_momentum();
    index_t momD = shells[3]->total_momentum();
    
    
    if (momA < momB) {
      
      //printf("swap1\n");
      ArrayListSwap(0, 1, &perm);
    }
    if (momC < momD) {
      //printf("swap2\n");
      ArrayListSwap(2, 3, &perm);
    }
    
    if (momC + momD < momA + momB) {
      //printf("swap3\n");
      
      ArrayListSwap(0, 2, &perm);
      ArrayListSwap(1, 3, &perm);
      
      std::swap(overlapAB, overlapCD);
      
    }
    
    ArrayList<index_t> anti_perm;
    anti_perm.Init(4);
    
    for (index_t i = 0; i < 4; i++) {
      
      anti_perm[perm[i]] = i;
      
    }
    
    /*
    printf("perm[0]: %d, perm[1]: %d, perm[2]: %d, perm[3]: %d\n", 
           perm[0], perm[1], perm[2], perm[3]);
    printf("a-perm[0]: %d, a-perm[1]: %d, a-perm[2]: %d, a-perm[3]: %d\n", 
           anti_perm[0], anti_perm[1], anti_perm[2], anti_perm[3]);
    */
    
    
    // now call the internal function
    
    Vector A_vec = shells[perm[0]]->center();
    Vector B_vec = shells[perm[1]]->center();
    Vector C_vec = shells[perm[2]]->center();
    Vector D_vec = shells[perm[3]]->center();
    
    double A_exp = shells[perm[0]]->exp();
    double B_exp = shells[perm[1]]->exp();
    double C_exp = shells[perm[2]]->exp();
    double D_exp = shells[perm[3]]->exp();
    
    momA = shells[perm[0]]->total_momentum();
    momB = shells[perm[1]]->total_momentum();
    momC = shells[perm[2]]->total_momentum();
    momD = shells[perm[3]]->total_momentum();
    
    double normA = shells[perm[0]]->normalization_constant(0);
    double normB = shells[perm[1]]->normalization_constant(0);
    double normC = shells[perm[2]]->normalization_constant(0);
    double normD = shells[perm[3]]->normalization_constant(0);

    ComputeERIInternal(A_vec, A_exp, momA, normA, B_vec, B_exp, momB, normB,
                       C_vec, C_exp, momC, normC, D_vec, D_exp, momD, normD,
                       overlapAB, overlapCD,
                       integrals);
    
    // unpermute the integrals
    integrals->UnPermute(anti_perm);
    
  } // ComputeERI
  
  
  
  void ComputeERIInternal(const Vector& A_vec, double A_exp, int A_mom, double normA,
                          const Vector& B_vec, double B_exp, int B_mom, double normB,
                          const Vector& C_vec, double C_exp, int C_mom, double normC,
                          const Vector& D_vec, double D_exp, int D_mom, double normD,
                          double overlapAB, double overlapCD,
                          IntegralTensor* integrals) {
    
    int num_primitives = 1;
    int max_momentum = max(max(A_mom, B_mom), max(C_mom, D_mom));
    int total_momentum = A_mom + B_mom + C_mom + D_mom;
    
    Libint_t tester;
    init_libint(&tester, max_momentum, num_primitives);
    
    // set up tester with overlaps and normalizationss
    
    // the normalization factor used for all integrals in libint
    double norm_denom = normA * normB * normC * normD;
    
    double sqrt_rho_pi = (A_exp + B_exp) * (C_exp + D_exp);
    sqrt_rho_pi /= (A_exp + B_exp + C_exp + D_exp);
    sqrt_rho_pi = sqrt(sqrt_rho_pi / math::PI);
    
    double aux_fac = 2 * sqrt_rho_pi * overlapAB * overlapCD;
    
    printf("aux_fac: %g\n", aux_fac);
    
    double* results = Libint_Eri(A_vec, A_exp, A_mom, B_vec, B_exp, B_mom, 
                                 C_vec, C_exp, C_mom, D_vec, D_exp, D_mom,
                                 aux_fac, &tester);
    
    index_t num_funs_a = NumFunctions(A_mom);
    index_t num_funs_b = NumFunctions(B_mom);
    index_t num_funs_c = NumFunctions(C_mom);
    index_t num_funs_d = NumFunctions(D_mom);
    
    
    
    //printf("results: %p\n", results);
    
    index_t a = 0;
    for (index_t a_ind = 0; a_ind <= A_mom; a_ind++) {
      
      index_t a_x = A_mom - a_ind;
      
      for (index_t a_ind2 = 0; a_ind2 <= a_ind; a_ind2++) {
        
        index_t a_y = a_ind - a_ind2;
        index_t a_z = a_ind2;
        
        double A_norm = ComputeNormalization(A_exp, a_x, a_y, a_z);
        
        index_t b = 0;
        
        for (index_t b_ind = 0; b_ind <= B_mom; b_ind++) {
          
          index_t b_x = B_mom - b_ind;
          
          for (index_t b_ind2 = 0; b_ind2 <= b_ind; b_ind2++) {
            
            index_t b_y = b_ind - b_ind2;
            index_t b_z = b_ind2;
            
            double B_norm = ComputeNormalization(B_exp, b_x, b_y, b_z);
            
            index_t c = 0;
            
            for (index_t c_ind = 0; c_ind <= C_mom; c_ind++) {
              
              index_t c_x = C_mom - c_ind;
              
              for (index_t c_ind2 = 0; c_ind2 <= c_ind; c_ind2++) {
                
                index_t c_y = c_ind - c_ind2;
                index_t c_z = c_ind2;
                
                double C_norm = ComputeNormalization(C_exp, c_x, c_y, c_z);
                
                index_t d = 0;
                
                for (index_t d_ind = 0; d_ind <= D_mom; d_ind++) {
                  
                  index_t d_x = D_mom - d_ind;
                  
                  
                  for (index_t d_ind2 = 0; d_ind2 <= d_ind; d_ind2++) {
                    
                    index_t d_y = d_ind - d_ind2;
                    index_t d_z = d_ind2;
                    
                    double D_norm = ComputeNormalization(D_exp, d_x, d_y, d_z);
                    
                    index_t integral_ind = IntegralIndex(a, A_mom, 
                                                         b, B_mom, 
                                                         c, C_mom, 
                                                         d, D_mom);
                    
                    results[integral_ind] = results[integral_ind] * A_norm 
                    * B_norm * C_norm * D_norm / norm_denom;
                    
                    printf("results[%d]: %g\n", integral_ind, 
                           results[integral_ind]);
                    
                    
                    d++;
                    
                  } // d_ind2
                  
                }// d_ind
                
                c++;
                
              }// c_ind2
              
            }// c_ind
            
            b++;
            
          }// b_ind2
          
        } // b_ind
        
        a++;
        
      } //a_ind2
      
    }// a_ind
    
    
    integrals->Init(num_funs_a, num_funs_b, num_funs_c, num_funs_d, results);
    free_libint(&tester);
    
    /*
    for (index_t a = 0; a < integrals->dim_a(); a++) {
      
      for (index_t b = 0; b < integrals->dim_b(); b++) {
        
        for (index_t c = 0; c < integrals->dim_c(); c++) {
          
          for (index_t d = 0; d < integrals->dim_d(); d++) {
            
            printf("integrals[%d, %d, %d, %d] = %g\n", a, b, c, d, 
                   integrals->ref(a, b, c, d));
            
          }
        }
      }
    }
    
    printf("\n\n");
    */
    
    
  } // ComputeERIInternal()
  
  
  double* Libint_Eri(const Vector& A_vec, double A_exp, int A_mom, 
                     const Vector& B_vec, double B_exp, int B_mom,
                     const Vector& C_vec, double C_exp, int C_mom,
                     const Vector& D_vec, double D_exp, int D_mom, 
                     double aux_fac, Libint_t* libint) {
    
    // assuming libint has been initialized, but PrimQuartet hasn't been set up
    
    // fill in the other stuff
    Vector AB;
    la::SubInit(B_vec, A_vec, &AB);
    
    Vector CD;
    la::SubInit(D_vec, C_vec, &CD);
    
    libint->AB[0] = AB[0];
    libint->AB[1] = AB[1];
    libint->AB[2] = AB[2];
    
    libint->CD[0] = CD[0];
    libint->CD[1] = CD[1];
    libint->CD[2] = CD[2];
    
    
    Vector P_vec;
    double gamma = ComputeGPTCenter(A_vec, A_exp, B_vec, B_exp, &P_vec);
    
    Vector Q_vec;
    double eta = ComputeGPTCenter(C_vec, C_exp, D_vec, D_exp, &Q_vec);
    
    Vector W_vec;
    double gamma_p_eta = ComputeGPTCenter(P_vec, gamma, Q_vec, eta, &W_vec);
    
    //goes in U[0]
    Vector PA;
    la::SubInit(A_vec, P_vec, &PA);
    
    libint->PrimQuartet[0].U[0][0] = PA[0];
    libint->PrimQuartet[0].U[0][1] = PA[1];
    libint->PrimQuartet[0].U[0][2] = PA[2];
    
    // goes in U[2]
    Vector QC;
    la::SubInit(C_vec, Q_vec, &QC);
    
    libint->PrimQuartet[0].U[2][0] = QC[0];
    libint->PrimQuartet[0].U[2][1] = QC[1];
    libint->PrimQuartet[0].U[2][2] = QC[2];
    
    // goes in U[4]
    Vector WP;
    la::SubInit(P_vec, W_vec, &WP);
    
    libint->PrimQuartet[0].U[4][0] = WP[0];
    libint->PrimQuartet[0].U[4][1] = WP[1];
    libint->PrimQuartet[0].U[4][2] = WP[2];
    
    // goes in U[5]
    Vector WQ;
    la::SubInit(Q_vec, W_vec, &WQ);
    
    libint->PrimQuartet[0].U[5][0] = WQ[0];
    libint->PrimQuartet[0].U[5][1] = WQ[1];
    libint->PrimQuartet[0].U[5][2] = WQ[2];
    
    
    int total_momentum = A_mom + B_mom + C_mom + D_mom;
    
    double rho = gamma * eta / (gamma_p_eta);
    
    // fill in PrimQuartet
    // will continue to assume uncontracted bases
    // I think this means that I only need libint->PrimQuartet[0]
    
    double PQ_dist_sq = la::DistanceSqEuclidean(P_vec, Q_vec);
    double T = rho * PQ_dist_sq;
    
    // TODO: Libmints uses am + 1 here (probably for derivatives)
    Compute_F(libint->PrimQuartet[0].F, total_momentum, T);
    
    la::Scale(total_momentum, aux_fac, libint->PrimQuartet[0].F);
    
    libint->PrimQuartet[0].oo2z = 1.0/(2.0 * gamma);
    libint->PrimQuartet[0].oo2n = 1.0/(2.0 * eta);
    libint->PrimQuartet[0].oo2zn = 1.0 / (2.0 * (gamma + eta));
    libint->PrimQuartet[0].poz = rho / gamma;
    libint->PrimQuartet[0].pon = rho / eta;
    libint->PrimQuartet[0].oo2p = 1.0 / (2.0 * rho);
    
    // TODO: modify the array F to hold the auxiliary integrals given in 
    // eqns 14 - 16 of libint programmers manual
    // this means the overlaps
    // what about normalizations?  
    
    
    
    // is it safe to leave the rest unspecified?
    // the docs say they are only used by libderiv and libr12
    
    //// compute the integrals
    
    //integrals = (double*)malloc(num_integrals * sizeof(double));
  
    double* integrals;
    if (total_momentum) {
      integrals = (build_eri[A_mom][B_mom][C_mom][D_mom](libint, 1));
    }
    else {
      integrals = libint->PrimQuartet[0].F;
    }
      
    // integrals get renormalized outside
    
    return integrals;
    
  } // Libint_Eri()
  
  
  
  ////////////// Creating shells and shell pairs ///////////////////////
  
index_t CreateShells(const Matrix& centers, const Vector& exponents, 
                     const Vector& momenta, ArrayList<BasisShell>* shells_out) {
  
  index_t num_functions = 0;
  
  shells_out->Init(centers.n_cols());
  
  for (index_t i = 0; i < centers.n_cols(); i++) {
    
    Vector new_cent;
    centers.MakeColumnVector(i, &new_cent);
    
    (*shells_out)[i].Init(new_cent, exponents[i], momenta[i], num_functions);
    
    num_functions += (*shells_out)[i].num_functions();
    
  } // for i
  
  return num_functions;
  
} // CreateShells()

  


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
      //double this_bound = oeints::ComputeOverlapIntegral(i_shell, j_shell);
      
      if (this_bound > shell_pair_cutoff) {
      
        shell_pairs->PushBack();
        
        (*shell_pairs)[num_shell_pairs].Init(i, j, i_shell, j_shell, 
                                             num_shell_pairs);
        (*shell_pairs)[num_shell_pairs].set_integral_upper_bound(this_bound);
        (*shell_pairs)[num_shell_pairs].set_schwartz_factor(this_bound);
        num_shell_pairs++;
        
              
      }
      
    } // for j
    
  } // for i 
  
  //printf("num_shell_pairs: %d\n", num_shell_pairs);
  
  return num_shell_pairs;

}

// this version is for link
index_t ComputeShellPairs(ArrayList<ShellPair>* shell_pairs, 
                          ArrayList<BasisShell>& shells_in, 
                          double shell_pair_cutoff, Vector* shell_max, 
                          ShellPair**** sigma_for_nu, 
                          ArrayList<index_t>* num_sigma_for_nu) {
    
  index_t num_shells = shells_in.size();
  
  index_t num_shell_pairs = 0;
  
  //num_per_shell->Init(num_shells);
  
  // What size to init to? 
  //shell_pairs->Init(num_shells);
  shell_pairs->Init();

  ArrayList<ArrayList<index_t> > significant_sig_index;
  significant_sig_index.Init(num_shells);

  DEBUG_ASSERT(shell_max != NULL);
  shell_max->Init(num_shells);
  
  for (index_t i = 0; i < num_shells; i++) {
    
    BasisShell i_shell = shells_in[i];
    index_t num_for_i = 0;
    
    
    double i_max = -DBL_MAX;
    
    //ArrayList<index_t> significant_sig_index;
    significant_sig_index[i].Init(num_shells);
    
    for (index_t j = i; j < num_shells; j++) {
      
      BasisShell j_shell = shells_in[j];
      
      // Do they use the overlap integral here?
      double this_bound = SchwartzBound(i_shell, j_shell);
      
      if (this_bound > shell_pair_cutoff) {
        
        shell_pairs->PushBack();
        
        (*shell_pairs)[num_shell_pairs].Init(i, j, i_shell, j_shell, 
                                             num_shell_pairs);
        (*shell_pairs)[num_shell_pairs].set_integral_upper_bound(this_bound);
        significant_sig_index[i][num_for_i] = num_shell_pairs;
        (*shell_pairs)[num_shell_pairs].set_schwartz_factor(this_bound);
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


  } // for i


  // do two loops to avoid invalidating the pointers into shell pair list when 
  // calling PushBack
  for (index_t i = 0; i < num_shells; i++) {

    index_t num_for_i = (*num_sigma_for_nu)[i];
    
    (*sigma_for_nu)[i] = (ShellPair**)malloc(num_for_i * sizeof(ShellPair**));

    for (index_t k = 0; k < num_for_i; k++) {
    
      (*sigma_for_nu)[i][k] = shell_pairs->begin() + significant_sig_index[i][k];
    
    } // for k
  
    // sort here?
        
  } // for i 
  
  
  return num_shell_pairs;
  
}


} // namespace eri


