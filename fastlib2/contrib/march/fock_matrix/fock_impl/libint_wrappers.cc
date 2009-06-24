/*
 *  libint_wrappers.cc
 *  
 *
 *  Created by William March on 6/17/09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

#include "libint_wrappers.h"

namespace eri {
  
  // copied from development verion of PSI3 on 6/17/09
  // part of libmint, from eri.cc
  void Compute_F(double *F, int n, double t)
  {
    // this is a macro in libmints/wavefunction.h
    // I think it stands for max factorial
    int MAX_FAC = 100;
    
    // very inefficient, needs to be stored somewhere
    double* df;
    // needs to go up to 2*(n+MAX_FAC+1)
    df = (double*)malloc(2 * (n + MAX_FAC + 1) * sizeof(double));    
    df[0] = 1.0;
    df[1] = 1.0;
    for(index_t a = 2; a < 2 * (n + MAX_FAC + 1); a++) {
      df[a] = a * df[a-2];
    }
    
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
      num = df[m2];
      i=0;
      sum = 1.0/(m2+1);
      do{
        i++;
        num = num*t2;
        term1 = num/df[m2+2*i+2];
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



