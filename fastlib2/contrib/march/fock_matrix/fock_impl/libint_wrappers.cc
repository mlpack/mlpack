/*
 *  libint_wrappers.cc
 *  
 *
 *  Created by William March on 6/17/09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

#include "libint_wrappers.h"

namespace libint_wrappers {
  
  /**
   * Stores the quantities n!! needed for integrals
   */
  double double_factorial[MAX_FAC];
  
  
  void ERIInit() {
    
    init_libint_base();
    
    // PSI just uses 100
    index_t num_factorials = MAX_FAC;
    
    //double_factorial = (double*)malloc(num_factorials * sizeof(double));
    
    double_factorial[0] = 1.0;
    double_factorial[1] = 1.0;
    for (index_t i = 2; i < num_factorials; i++) {
      
      double_factorial[i] = (double)i * double_factorial[i-2];
      
    }
    
  }
  
  void ERIFree() {
   
    //free(double_factorial);
    
  }
  
  
  
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
  
  
  double ComputeNormalization(double exp, int x_mom, int y_mom, int z_mom) {
    
    int total_momentum = x_mom + y_mom + z_mom;
    
    // should be able to do this more efficiently
    double result = pow(2.0/math::PI, 0.75);
    
    result *= pow(2.0, total_momentum); 
    result *= pow(exp, ((2.0 * total_momentum + 3.0)/4.0));
    result /= sqrt(double_factorial[2*x_mom - 1] 
                   * double_factorial[2*y_mom - 1] 
                   * double_factorial[2*z_mom - 1]);
    
    return result;
    
  } // ComputeNormalization
  
  
  
  index_t IntegralIndex(int a_ind, int A_mom, int b_ind, int B_mom, 
                        int c_ind, int C_mom, int d_ind, int D_mom) {
    
    index_t result = ((a_ind * B_mom + b_ind) * C_mom + c_ind) * D_mom + d_ind;
    
    return result;
    
  } // IntegralIndex
  
  
  
  double* ComputeERI(const Vector& A_vec, double A_exp, int A_mom, 
                     const Vector& B_vec, double B_exp, int B_mom,
                     const Vector& C_vec, double C_exp, int C_mom,
                     const Vector& D_vec, double D_exp, int D_mom) {
    
    int num_primitives = 1;
    int max_momentum = max(max(A_mom, B_mom), max(C_mom, D_mom));
    
    Libint_t tester;
    init_libint(&tester, max_momentum, num_primitives);
    
    double* results = Libint_Eri(A_vec, A_exp, A_mom, B_vec, B_exp, B_mom, 
                                 C_vec, C_exp, C_mom, D_vec, D_exp, D_mom,
                                 tester);
    
    index_t num_integrals = NumFunctions(A_mom) * NumFunctions(B_mom)
                            * NumFunctions(C_mom) * NumFunctions(D_mom);
    
    // the normalization factor used for all integrals in libint
    double norm_denom = ComputeNormalization(A_exp, A_mom, 0, 0)
                        * ComputeNormalization(B_exp, B_mom, 0, 0)
                        * ComputeNormalization(C_exp, C_mom, 0, 0)
                        * ComputeNormalization(D_exp, D_mom, 0, 0);
    
    
    for (index_t a_ind = 0; a_ind < A_mom; a_ind++) {
      
      index_t a_x = A_mom - a_ind;
      
      for (index_t a_ind2 = 0; a_ind2 <= a_ind; a_ind2++) {
       
        index_t a_y = a_ind - a_ind2;
        index_t a_z = a_ind2;
        
        double A_norm = ComputeNormalization(A_exp, a_x, a_y, a_z);
     
        for (index_t b_ind = 0; b_ind < B_mom; b_ind++) {
          
          index_t b_x = B_mom - b_ind;
          
          for (index_t b_ind2 = 0; b_ind2 <= b_ind; b_ind2++) {
            
            index_t b_y = b_ind - b_ind2;
            index_t b_z = b_ind2;
            
            double B_norm = ComputeNormalization(B_exp, b_x, b_y, b_z);
            
            for (index_t c_ind = 0; c_ind < C_mom; c_ind++) {
              
              index_t c_x = C_mom - c_ind;
              
              for (index_t c_ind2 = 0; c_ind2 <= c_ind; c_ind2++) {
                
                index_t c_y = c_ind - c_ind2;
                index_t c_z = c_ind2;
                
                double C_norm = ComputeNormalization(C_exp, c_x, c_y, c_z);
                
                for (index_t d_ind = 0; d_ind < D_mom; d_ind++) {
                  
                  index_t d_x = D_mom - d_ind;
                  
                  index_t integral_ind = IntegralIndex(a_ind, A_mom, 
                                                       b_ind, B_mom, 
                                                       c_ind, C_mom, 
                                                       d_ind, D_mom);
                  
                  for (index_t d_ind2 = 0; d_ind2 <= d_ind; d_ind2++) {
                    
                    index_t d_y = d_ind - d_ind2;
                    index_t d_z = d_ind2;
                    
                    double D_norm = ComputeNormalization(D_exp, d_x, d_y, d_z);
                    
                    
                    results[integral_ind] = results[integral_ind] * A_norm 
                        * B_norm * C_norm * D_norm / norm_denom;
                    
                    
                  } // d_ind2
                  
                }// d_ind
                
              }// c_ind2
              
            }// c_ind
            
          }// b_ind2
          
        } // b_ind
        
      } //a_ind2
     
    }// a_ind
    
    
    free_libint(&tester);
    
    return results;
    
    
  } // ComputeERI()
  
  
  // copied from development verion of PSI3 on 6/17/09
  // part of libmint, from eri.cc
  void Compute_F(double *F, int n, double t)
  {
    
    int i, m, k;
    int m2;
    double t2;
    double num;
    double sum;
    double term1, term2;
    static double K = 1.0/M_2_SQRTPI;
    double et;
    
    // from macro in libmint's eri.cc 
    double EPS = 1.0e-17;
    
    // long range approx
    if (t>20.0){
      t2 = 2*t;
      et = exp(-t);
      t = sqrt(t);
      F[0] = K*erf(t)/t;
      for(m=0; m<=n-1; m++){
        F[m+1] = ((2*m + 1)*F[m] - et)/(t2);
      }
    }
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
  } // Compute_F()
  
  double* Libint_Eri(const Vector& A_vec, double A_exp, int A_mom, 
                    const Vector& B_vec, double B_exp, int B_mom,
                    const Vector& C_vec, double C_exp, int C_mom,
                    const Vector& D_vec, double D_exp, int D_mom,
                    Libint_t& libint) {
    
    // assuming libint has been initialized, but PrimQuartet hasn't been set up
    
    double* integrals; 
    
    
    // fill in the other stuff
    Vector AB;
    la::SubInit(A_vec, B_vec, &AB);

    Vector CD;
    la::SubInit(C_vec, D_vec, &CD);
    
    libint.AB[0] = AB[0];
    libint.AB[1] = AB[1];
    libint.AB[2] = AB[2];

    libint.CD[0] = CD[0];
    libint.CD[1] = CD[1];
    libint.CD[2] = CD[2];
    
    double gamma = A_exp + B_exp;
    double eta = C_exp + D_exp;
    
    Vector P_vec;
    la::ScaleInit(A_exp, A_vec, &P_vec);
    la::AddExpert(3, B_exp, B_vec.ptr(), P_vec.ptr());
    la::Scale(1.0/gamma, &P_vec);

    Vector Q_vec;
    la::ScaleInit(C_exp, C_vec, &Q_vec);
    la::AddExpert(3, D_exp, D_vec.ptr(), Q_vec.ptr());
    la::Scale(1.0/eta, &Q_vec);
    
    Vector W_vec;
    la::ScaleInit(eta, Q_vec, &W_vec);
    la::AddExpert(3, gamma, P_vec.ptr(), W_vec.ptr());
    la::Scale(1.0/(gamma + eta), &W_vec);
    
    //goes in U[0]
    Vector PA;
    la::SubInit(P_vec, A_vec, &PA);
    
    libint.PrimQuartet[0].U[0][0] = PA[0];
    libint.PrimQuartet[0].U[0][1] = PA[1];
    libint.PrimQuartet[0].U[0][2] = PA[2];
    
    // goes in U[2]
    Vector QC;
    la::SubInit(Q_vec, C_vec, &QC);
    
    libint.PrimQuartet[0].U[2][0] = QC[0];
    libint.PrimQuartet[0].U[2][1] = QC[1];
    libint.PrimQuartet[0].U[2][2] = QC[2];
    
    // goes in U[4]
    Vector WP;
    la::SubInit(W_vec, P_vec, &WP);
    
    libint.PrimQuartet[0].U[4][0] = WP[0];
    libint.PrimQuartet[0].U[4][1] = WP[1];
    libint.PrimQuartet[0].U[4][2] = WP[2];
    
    // goes in U[5]
    Vector WQ;
    la::SubInit(W_vec, Q_vec, &WQ);
    
    libint.PrimQuartet[0].U[5][0] = WQ[0];
    libint.PrimQuartet[0].U[5][1] = WQ[1];
    libint.PrimQuartet[0].U[5][2] = WQ[2];
    
    
    int total_momentum = A_mom + B_mom + C_mom + D_mom;
    
    double rho = gamma * eta / (gamma + eta);
    
    // fill in PrimQuartet
    // will continue to assume uncontracted bases
    // I think this means that I only need libint.PrimQuartet[0]
    
    double PQ_dist_sq = la::DistanceSqEuclidean(P_vec, Q_vec);
    double T = rho * PQ_dist_sq;
    Compute_F(libint.PrimQuartet[0].F, total_momentum, T);
    
    libint.PrimQuartet[0].oo2z = 1.0/(2.0 * gamma);
    libint.PrimQuartet[0].oo2n = 1.0/(2.0 * eta);
    libint.PrimQuartet[0].oo2zn = 1.0 / (2.0 * (gamma + eta));
    libint.PrimQuartet[0].poz = rho / gamma;
    libint.PrimQuartet[0].pon = rho / eta;
    libint.PrimQuartet[0].oo2p = 1.0 / (2.0 * rho);
    
    
    
    // is it safe to leave the rest unspecified?
    // the docs say they are only used by libderiv and libr12
    
    //// compute the integrals
    
    // this isn't right
    //index_t num_integrals = 3 * (A_mom + B_mom + C_mom + D_mom);
    //num_integrals = (num_integrals==0) ? 1 : num_integrals;
  
    //integrals = (double*)malloc(num_integrals * sizeof(double));
    
    // need to enforce that the integrals are in the right order
    integrals = build_eri[A_mom][B_mom][C_mom][D_mom](&libint, 1);
    
    // renormalize the integrals 
    
    
    
    return integrals;
    
  } // Libint_Eri()
  
  
  
} // eri



