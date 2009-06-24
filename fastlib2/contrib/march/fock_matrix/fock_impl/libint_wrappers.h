/*
 *  libint_wrappers.h
 *  
 *
 *  Created by William March on 6/17/09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

#include "fastlib/fastlib.h"
#include "../../libint/include/libint/libint.h"
#include "../../libint/include/libint/hrr_header.h"
#include "../../libint/include/libint/vrr_header.h"


namespace eri {
  
  void Compute_F(double* F, int n, double t);
  
  double* Libint_Eri(const Vector& A_vec, double A_exp, int A_mom, 
                    const Vector& B_vec, double B_exp, int B_mom,
                    const Vector& C_vec, double C_exp, int C_mom,
                    const Vector& D_vec, double D_exp, int D_mom,
                    Libint_t& libint);
  
}