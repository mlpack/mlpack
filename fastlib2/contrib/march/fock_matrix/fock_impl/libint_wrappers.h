/*
 *  libint_wrappers.h
 *  
 *
 *  Created by William March on 6/17/09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef LIBINT_WRAPPERS_H
#define LIBINT_WRAPPERS_H

#include "fastlib/fastlib.h"
#include "../../libint/include/libint/libint.h"
#include "../../libint/include/libint/hrr_header.h"
#include "../../libint/include/libint/vrr_header.h"

// copied from libint or libmint
#define MAX_FAC 100

namespace libint_wrappers {
  
  
  /**
   * Overall initializer for ERI's.  Computes the double factorials and 
   * initializes Libint with init_libint_base()
   *
   * Call this function before using any other functions in namespace eri
   *
   * Only needs to be called once per program.
   */
  void ERIInit();
  
  /**
   * Frees the double factorial array.  
   */
  void ERIFree();
  
  /**
   * Computes the number of functions in a shell of the given momentum.
   */
  index_t NumFunctions(int momentum);
  
  /**
   * Returns the index of the given integral in the array returned from LIBINT.
   */
  index_t IntegralIndex(int a_ind, int A_mom, int b_ind, int B_mom, 
                        int c_ind, int C_mom, int d_ind, int D_mom);
  
  /**
   * Normalization function for higher momenta
   */
  double ComputeNormalization(double exp, int x_mom, int y_mom, int z_mom);
  
  
  /**
   * This is the function to call from outside.
   *
   * This currently assumes that the momenta obey the conditions:
   * A_mom >= B_mom
   * C_mom >= D_mom
   * A_mom + B_mom <= C_mom + D_mom
   *
   * IMPORTANT: must have called ERIInit() before calling this function
   */
  double* ComputeERI(const Vector& A_vec, double A_exp, int A_mom, 
                     const Vector& B_vec, double B_exp, int B_mom,
                     const Vector& C_vec, double C_exp, int C_mom,
                     const Vector& D_vec, double D_exp, int D_mom);
  
  void Compute_F(double* F, int n, double t);
  
  double* Libint_Eri(const Vector& A_vec, double A_exp, int A_mom, 
                    const Vector& B_vec, double B_exp, int B_mom,
                    const Vector& C_vec, double C_exp, int C_mom,
                    const Vector& D_vec, double D_exp, int D_mom,
                    Libint_t& libint);
  
}


#endif
