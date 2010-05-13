/*
 *  eri_bounds.cc
 *  
 *
 *  Created by William March on 8/28/09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

#include "eri_bounds.h"

namespace eri_bounds {

  // TODO: fill me in!
  double BoundFm(double T, int m) {
  
    double m_over_T = (double)m/T;
    
    if (m_over_T <= 1) {
      return pow(m_over_T, (double)m) * exp(-(double)m);
    }
    else {
      return exp(-T);
    }
    
    return 1.0;
    
  } // BoundFm
  
  // TODO: fill me in!
  double BoundPrefactor(BasisShellTree* a_shells, BasisShellTree* b_shells,
                        BasisShellTree* c_shells, BasisShellTree* d_shells) {

    double a_min = a_shells->exponents().lo;
    double b_min = b_shells->exponents().lo;
    double c_min = c_shells->exponents().lo;
    double d_min = d_shells->exponents().lo;
    
    double E1 = exp(-a_min * b_min 
                    / (a_min + b_min)
                    * a_shells->bound().MinDistanceSq(b_shells->bound()));
    double E2 = exp(-c_min * d_min 
                    / (c_min + d_min)
                    * c_shells->bound().MinDistanceSq(d_shells->bound()));
    
    double denom = (a_min + b_min) * (c_min + d_min) 
                   * sqrt(a_min + b_min + c_min + d_min);
    
    double sqrt_pi = sqrt(math::PI);
    double retval = 2 * sqrt_pi * sqrt_pi * sqrt_pi * sqrt_pi * sqrt_pi;
    double normalizations = a_shells->normalizations().hi 
                            * b_shells->normalizations().hi 
                            * c_shells->normalizations().hi
                            * d_shells->normalizations().hi;
    retval *= E1 * E2 * normalizations / denom;
    
    DEBUG_ASSERT(retval >= 0.0);
    
    return retval;
    
  } // BoundPrefactor()
  
  // I actually shouldn't bound these distances separately
  // I should compute all three throught the bound, and take the max
  // this should be tighter
  double MaxCartesianDistance(const DHrectBound<2>& bound1,
                              const DHrectBound<2>& bound2) {
    
    double retval = -DBL_MAX;
    for (index_t i = 0; i < 3; i++) {
      
      double this_val = max(fabs(bound1.get(i).hi - bound2.get(i).lo),
                            fabs(bound2.get(i).hi - bound1.get(i).lo));
      
      if (this_val > retval) {
        retval = this_val;
      }
      
    } // for i
    
    DEBUG_ASSERT(retval >= 0.0);
    return retval;
    
  } // MaxCartesianDistance
  
  double BoundSSSS(const DRange& a_exp, const DRange& b_exp, 
                   const DRange& c_exp, const DRange& d_exp, 
                   const DHrectBound<2>& a_bound, const DHrectBound<2>& b_bound, 
                   const DHrectBound<2>& c_bound, const DHrectBound<2>& d_bound,
                   const DHrectBound<2>& p_bound, const DHrectBound<2>& q_bound, 
                   double T, int aux_fac) {
    
    return BoundFm(T, aux_fac);
    
  } // BoundSSSS
  
  // pass T around, since it will be needed many times
  double BoundPSSS(const DRange& a_exp, const DRange& b_exp, 
                   const DRange& c_exp, const DRange& d_exp, 
                   const DHrectBound<2>& a_bound, const DHrectBound<2>& b_bound, 
                   const DHrectBound<2>& c_bound, const DHrectBound<2>& d_bound,
                   const DHrectBound<2>& p_bound, const DHrectBound<2>& q_bound, 
                   double T, int aux_fac) {
    
    double term1 = b_exp.hi / (b_exp.hi + a_exp.lo) 
                    * MaxCartesianDistance(b_bound, a_bound);
    term1 *= BoundFm(T, 0 + aux_fac);
    DEBUG_ASSERT(term1 >= 0.0);
    
    double term2 = (c_exp.hi + d_exp.hi) / (a_exp.lo + b_exp.lo + c_exp.hi + d_exp.hi);
    term2 *= MaxCartesianDistance(q_bound, p_bound) * BoundFm(T, 1 + aux_fac);
    DEBUG_ASSERT(term2 >= 0.0);
    
    return term1 + term2;
    
  } // BoundPSSS()
  
  double BoundPPSS(const DRange& a_exp, const DRange& b_exp, 
                   const DRange& c_exp, const DRange& d_exp, 
                   const DHrectBound<2>& a_bound, const DHrectBound<2>& b_bound, 
                   const DHrectBound<2>& c_bound, const DHrectBound<2>& d_bound,
                   const DHrectBound<2>& p_bound, const DHrectBound<2>& q_bound, 
                   double T, int aux_fac) { 
    
    double term1 = a_exp.hi / (a_exp.hi + b_exp.lo);
    term1 *= MaxCartesianDistance(a_bound, b_bound);
    term1 *= BoundPSSS(a_exp, b_exp, c_exp, d_exp, 
                       a_bound, b_bound, c_bound, d_bound, p_bound, q_bound,
                       T, aux_fac);
    DEBUG_ASSERT(term1 >= 0.0);
    
    double term2_expbound = (c_exp.hi + d_exp.hi)
                            /(a_exp.lo + b_exp.lo + c_exp.hi + d_exp.hi);
    double term2 = term2_expbound;
    term2 *= MaxCartesianDistance(q_bound, p_bound);
    term2 *= BoundPSSS(a_exp, b_exp, c_exp, d_exp, 
                       a_bound, b_bound, c_bound, d_bound, p_bound, q_bound,
                       T, 1+aux_fac);
    DEBUG_ASSERT(term2 >= 0.0);
    
    double term3 = 0.5 / (a_exp.lo + b_exp.lo);
    term3 *= (BoundFm(T, aux_fac) + term2_expbound * BoundFm(T, 1+aux_fac));
    DEBUG_ASSERT(term3 >= 0.0);
    
    return term1 + term2 + term3;
    
  } // BoundPPSS()
  
  double BoundPSPS(const DRange& a_exp, const DRange& b_exp, 
                   const DRange& c_exp, const DRange& d_exp, 
                   const DHrectBound<2>& a_bound, const DHrectBound<2>& b_bound, 
                   const DHrectBound<2>& c_bound, const DHrectBound<2>& d_bound,
                   const DHrectBound<2>& p_bound, const DHrectBound<2>& q_bound, 
                   double T, int aux_fac) {
    
    double term1 = d_exp.hi / (d_exp.hi + c_exp.lo);
    term1 *= MaxCartesianDistance(d_bound, c_bound);
    term1 *= BoundPSSS(a_exp, b_exp, c_exp, d_exp, 
                       a_bound, b_bound, c_bound, d_bound, p_bound, q_bound,
                       T, aux_fac);
    DEBUG_ASSERT(term1 >= 0.0);
    
    double big_exp_bound = (a_exp.hi + b_exp.hi)
                           /(a_exp.hi + b_exp.hi + c_exp.lo + d_exp.lo);
    
    double term2 = big_exp_bound * MaxCartesianDistance(p_bound, q_bound);
    term2 *= BoundPSSS(a_exp, b_exp, c_exp, d_exp, 
                       a_bound, b_bound, c_bound, d_bound, p_bound, q_bound,
                       T, 1+aux_fac);
    DEBUG_ASSERT(term2 >= 0.0);
    
    double term3 = 0.5 / (a_exp.lo + b_exp.lo + c_exp.lo + d_exp.lo);
    term3 *= BoundFm(T, 1 + aux_fac);
    DEBUG_ASSERT(term3 >= 0.0);
    
    return term1 + term2 + term3;
    
  } // BoundPSPS
  
  // TODO: fill me in
  double BoundPPPS(const DRange& a_exp, const DRange& b_exp, 
                   const DRange& c_exp, const DRange& d_exp, 
                   const DHrectBound<2>& a_bound, const DHrectBound<2>& b_bound, 
                   const DHrectBound<2>& c_bound, const DHrectBound<2>& d_bound,
                   const DHrectBound<2>& p_bound, const DHrectBound<2>& q_bound, 
                   double T, int aux_fac) {
    
    double term1 = d_exp.hi / (c_exp.lo + d_exp.hi);
    term1 *= MaxCartesianDistance(d_bound, c_bound);
    term1 *= BoundPPSS(a_exp, b_exp, c_exp, d_exp, 
                       a_bound, b_bound, c_bound, d_bound, p_bound, q_bound,
                       T, aux_fac);
    DEBUG_ASSERT(term1 >= 0.0);
    
    double term2 = (a_exp.hi + b_exp.hi) / (a_exp.hi + b_exp.hi + c_exp.lo + d_exp.lo);
    term2 *= MaxCartesianDistance(p_bound, q_bound);
    term2 *= BoundPPSS(a_exp, b_exp, c_exp, d_exp, 
                       a_bound, b_bound, c_bound, d_bound, p_bound, q_bound,
                       T, 1+aux_fac);
    DEBUG_ASSERT(term2 >= 0.0);
    
    double term3 = 0.5 / (a_exp.lo + b_exp.lo + c_exp.lo + d_exp.lo);
    // swap a and b to compute the sp|ss integral in the bound
    term3 *= (BoundPSSS(b_exp, a_exp, c_exp, d_exp, 
                        b_bound, a_bound, c_bound, d_bound, p_bound, q_bound,
                        T, 1+aux_fac)
              + BoundPSSS(a_exp, b_exp, c_exp, d_exp, 
                          a_bound, b_bound, c_bound, d_bound, p_bound, q_bound,
                          T, 1+aux_fac));
    DEBUG_ASSERT(term3 >= 0.0);
    
    return term1 + term2 + term3;
    
  } // BoundPPPS
  
  // TODO: fill me in
  double BoundPPPP(const DRange& a_exp, const DRange& b_exp, 
                   const DRange& c_exp, const DRange& d_exp, 
                   const DHrectBound<2>& a_bound, const DHrectBound<2>& b_bound, 
                   const DHrectBound<2>& c_bound, const DHrectBound<2>& d_bound,
                   const DHrectBound<2>& p_bound, const DHrectBound<2>& q_bound, 
                   double T, int aux_fac) {
    
    double term1 = c_exp.hi / (c_exp.hi + d_exp.lo);
    term1 *= MaxCartesianDistance(c_bound, d_bound);
    term1 *= BoundPPPS(b_exp, a_exp, c_exp, d_exp, 
                       b_bound, a_bound, c_bound, d_bound, p_bound, q_bound,
                       T, aux_fac);
    DEBUG_ASSERT(term1 >= 0.0);
    
    double term2 = (a_exp.hi + b_exp.hi) / (a_exp.hi + b_exp.hi + c_exp.lo + d_exp.lo);
    term2 *= MaxCartesianDistance(p_bound, q_bound);
    term2 *= BoundPPPS(a_exp, b_exp, c_exp, d_exp, 
                       a_bound, b_bound, c_bound, d_bound, p_bound, q_bound,
                       T, 1+aux_fac);
    DEBUG_ASSERT(term2 >= 0.0);
    
    double term3 = 0.5 / (a_exp.lo + b_exp.lo + c_exp.lo + d_exp.lo);
    // swap a and b to compute the sp|ps integral in the bound
    term3 *= (BoundPSPS(b_exp, a_exp, c_exp, d_exp, 
                        b_bound, a_bound, c_bound, d_bound, p_bound, q_bound,
                        T, 1+aux_fac)
              + BoundPSPS(a_exp, b_exp, c_exp, d_exp, 
                          a_bound, b_bound, c_bound, d_bound, p_bound, q_bound,
                          T, 1+aux_fac));
    DEBUG_ASSERT(term3 >= 0.0);
    
    double term4 = (a_exp.hi + b_exp.hi) / (a_exp.hi + b_exp.hi + c_exp.lo + d_exp.lo);
    term4 *= BoundPPSS(a_exp, b_exp, c_exp, d_exp, 
                       a_bound, b_bound, c_bound, d_bound, p_bound, q_bound,
                       T, aux_fac);
    term4 += BoundPPSS(a_exp, b_exp, c_exp, d_exp, 
                       a_bound, b_bound, c_bound, d_bound, p_bound, q_bound,
                       T, 1+aux_fac);
    term4 *= 0.5 / (c_exp.lo + d_exp.lo);
    DEBUG_ASSERT(term4 >= 0.0);
    
    return term1 + term2 + term3 + term4;
    
  } // BoundPPPP()
  
  
  
  double BoundIntegrals(BasisShellTree* a_shells, BasisShellTree* b_shells,
                        BasisShellTree* c_shells, BasisShellTree* d_shells) {
    
    // each node should only have a single momentum
    
    // IMPORTANT: only considering s and p functions for now
    
    if (d_shells->momenta().lo > c_shells->momenta().hi) {
      return BoundIntegrals(a_shells, b_shells, d_shells, c_shells);
    }
    else if (b_shells->momenta().lo > a_shells->momenta().hi) {
      return BoundIntegrals(b_shells, a_shells, c_shells, d_shells);
    }
    else if (d_shells->momenta().lo + c_shells->momenta().hi
             > b_shells->momenta().lo + a_shells->momenta().hi) {
      return BoundIntegrals(c_shells, d_shells, a_shells, b_shells);
    }
    
    // all momenta should be in correct order now
    
    double pre_val = BoundPrefactor(a_shells, b_shells, c_shells, d_shells);
    
    double f_val;
    DEBUG_ASSERT(pre_val >= 0.0);
    
    int total_momentum = a_shells->momenta().lo + b_shells->momenta().lo 
    + c_shells->momenta().lo + d_shells->momenta().lo;
    
    if (pre_val > 0.0) { 
      // if it's less, we won't bother with bounding the rest, one of the 
      // exponential factors is too small for it to matter.
      DRange& a_exp = a_shells->exponents();
      DRange& b_exp = b_shells->exponents();
      DRange& c_exp = c_shells->exponents();
      DRange& d_exp = d_shells->exponents();
      
      DHrectBound<2>& a_bound = a_shells->bound();
      DHrectBound<2>& b_bound = b_shells->bound();
      DHrectBound<2>& c_bound = c_shells->bound();
      DHrectBound<2>& d_bound = d_shells->bound();
      
      DHrectBound<2> p_bound;
      p_bound.WeightedAverageBoxesInit(a_exp.lo, a_exp.hi, a_bound,
                                       b_exp.lo, b_exp.hi, b_bound);
      
      DHrectBound<2> q_bound;
      q_bound.WeightedAverageBoxesInit(c_exp.lo, c_exp.hi, c_bound,
                                       d_exp.lo, d_exp.hi, d_bound);
      
      // T = (a + b)(c+d)/(a+b+c+d) PQ^2
      // minimize or maximize?
      // minimized in previous version
      // does this hold for F_m for m > 0?
      double T = (a_exp.lo + b_exp.lo) * (c_exp.lo + d_exp.lo) 
                 / (a_exp.lo + b_exp.lo + c_exp.lo + d_exp.lo);
      T *= p_bound.MinDistanceSq(q_bound);
      
      if (total_momentum == 4) {
        f_val = BoundPPPP(a_exp, b_exp, c_exp, d_exp, 
                          a_bound, b_bound, c_bound, d_bound, p_bound, q_bound,
                          T, 0);
      }
      else if (total_momentum == 3) {
        f_val = BoundPPPS(a_exp, b_exp, c_exp, d_exp, 
                          a_bound, b_bound, c_bound, d_bound, p_bound, q_bound,
                          T, 0);
      }
      else if (total_momentum == 2) {
        
        if (c_shells->momenta().lo == 1) {
          f_val = BoundPSPS(a_exp, b_exp, c_exp, d_exp, 
                            a_bound, b_bound, c_bound, d_bound, p_bound, q_bound,
                            T, 0);
        }
        else {
          f_val = BoundPPSS(a_exp, b_exp, c_exp, d_exp, 
                            a_bound, b_bound, c_bound, d_bound, p_bound, q_bound,
                            T, 0);
        }
        
      }
      else if (total_momentum == 1) {
        f_val = BoundPSSS(a_exp, b_exp, c_exp, d_exp, 
                          a_bound, b_bound, c_bound, d_bound, p_bound, q_bound,
                          T, 0);
      }
      else if (total_momentum == 0) {
        f_val = BoundFm(T, 0); 
      }
      else {
        FATAL("Unaccounted for total momentum.");
      }
      
    }
    else {
      f_val = 1.0;
    }
    double retval = pre_val * f_val;
    DEBUG_ASSERT(retval >= 0.0);
    return retval;
    
  } // BoundIntegrals
  
  
  
} // namespace