/*
 *  eri_bounds.h
 *  
 *
 *  Created by William March on 8/28/09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef ERI_BOUNDS_H
#define ERI_BOUNDS_H

#include "contrib/march/fock_matrix/fock_impl/eri.h"
#include "basis_shell_tree.h"

namespace eri_bounds {
  
  /**
   * Returns the bound for the incomplete gamma function
   */
  double BoundFm(double T, int m);
  
  /**
   * The upper bound of the prefactor
   */
  double BoundPrefactor(BasisShellTree* a_shells, BasisShellTree* b_shells,
                        BasisShellTree* c_shells, BasisShellTree* d_shells);
  
  double MaxCartesianDistance(const DHrectBound<2>& bound1,
                              const DHrectBound<2>& bound2);
  
  // all of these return bounds on the absolute values of the integrals
  // they do not include the prefactor Kf
  double BoundSSSS(const DRange& a_exp, const DRange& b_exp, 
                   const DRange& c_exp, const DRange& d_exp, 
                   const DHrectBound<2>& a_bound, const DHrectBound<2>& b_bound, 
                   const DHrectBound<2>& c_bound, const DHrectBound<2>& d_bound,
                   const DHrectBound<2>& p_bound, const DHrectBound<2>& q_bound, 
                   double T, int aux_fac);
  
  double BoundPSSS(const DRange& a_exp, const DRange& b_exp, 
                   const DRange& c_exp, const DRange& d_exp, 
                   const DHrectBound<2>& a_bound, const DHrectBound<2>& b_bound, 
                   const DHrectBound<2>& c_bound, const DHrectBound<2>& d_bound,
                   const DHrectBound<2>& p_bound, const DHrectBound<2>& q_bound, 
                   double T, int aux_fac);
  
  double BoundPPSS(const DRange& a_exp, const DRange& b_exp, 
                   const DRange& c_exp, const DRange& d_exp, 
                   const DHrectBound<2>& a_bound, const DHrectBound<2>& b_bound, 
                   const DHrectBound<2>& c_bound, const DHrectBound<2>& d_bound,
                   const DHrectBound<2>& p_bound, const DHrectBound<2>& q_bound, 
                   double T, int aux_fac);
  
  double BoundPSPS(const DRange& a_exp, const DRange& b_exp, 
                   const DRange& c_exp, const DRange& d_exp, 
                   const DHrectBound<2>& a_bound, const DHrectBound<2>& b_bound, 
                   const DHrectBound<2>& c_bound, const DHrectBound<2>& d_bound,
                   const DHrectBound<2>& p_bound, const DHrectBound<2>& q_bound, 
                   double T, int aux_fac);
  
  double BoundPPPS(const DRange& a_exp, const DRange& b_exp, 
                   const DRange& c_exp, const DRange& d_exp, 
                   const DHrectBound<2>& a_bound, const DHrectBound<2>& b_bound, 
                   const DHrectBound<2>& c_bound, const DHrectBound<2>& d_bound,
                   const DHrectBound<2>& p_bound, const DHrectBound<2>& q_bound, 
                   double T, int aux_fac);
  
  double BoundPPPP(const DRange& a_exp, const DRange& b_exp, 
                   const DRange& c_exp, const DRange& d_exp, 
                   const DHrectBound<2>& a_bound, const DHrectBound<2>& b_bound, 
                   const DHrectBound<2>& c_bound, const DHrectBound<2>& d_bound,
                   const DHrectBound<2>& p_bound, const DHrectBound<2>& q_bound, 
                   double T, int aux_fac);
  
  
  /**
   * Computes the necessary vectors and the prefactor.  Handles any necessary
   * permutations
   */
  double BoundIntegrals(BasisShellTree* a_shells, BasisShellTree* b_shells,
                        BasisShellTree* c_shells, BasisShellTree* d_shells);
  
} // eri_bounds



#endif

