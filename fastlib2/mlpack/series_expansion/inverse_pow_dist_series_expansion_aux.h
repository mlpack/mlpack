/** @file inverse_pow_dist_series_expansion_aux.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef INVERSE_POW_DIST_SERIES_EXPANSION_AUX
#define INVERSE_POW_DIST_SERIES_EXPANSION_AUX

#include <iostream>

#include "fastlib/fastlib.h"

/** @brief Series expansion class for the inverse distance power
 *         functions.
 */
class InversePowDistSeriesExpansionAux {
  
 private:

  /** @brief The inverse power $1 / r^{\lambda}$ for which this object
   *         is relevant.
   */
  double lambda_;

  int dim_;

  int max_order_;

  /** @brief These are the $(-1)^n (n C a) (a C b) / (2^a n!)$
   *         constants. $n$ index the most outer ArrayList, and (a, b)
   *         are the indices on each matrix.
   */
  ArrayList<Matrix> precomputed_constants_;
  
  OT_DEF_BASIC(InversePowDistSeriesExpansionAux) {
    OT_MY_OBJECT(dim_);
    OT_MY_OBJECT(max_order_);
    OT_MY_OBJECT(precomputed_constants_);
  }

 public:

  int get_dimension() const { 
    return dim_;
  }

  int get_max_order() const {
    return max_order_;
  }

  void set_max_order(int new_order) {
    max_order_ = new_order;
  }

  void ComputeConstants() {
    
    double n_factorial = 1.0;
    precomputed_constants_.Init(max_order_);

    for(index_t n = 0; n < precomputed_constants_.size(); n++) {
      
      // Allocate $(n + 1)$ by $(n + 1)$ matrix.
      precomputed_constants_[n].Init(n + 1, n + 1);

      // The reference to the matrix.
      Matrix &n_th_order_matrix = precomputed_constants_[n];

      double two_raised_to_a = 1.0;
      for(index_t a = 0; a <= n; a++) {

	for(index_t b = 0; b <= a; b++) {
	  n_th_order_matrix.set(a, b, math::BinomialCoefficient(n, a) * 
				math::BinomialCoefficient(a, b) / 
				(two_raised_to_a * n_factorial));
	  if(n % 2 == 1) {
	    n_th_order_matrix.set(a, b, -n_th_order_matrix.get(a, b));
	  }
	}

	two_raised_to_a *= 2.0;
      }

      n_factorial *= (n + 1);
    }
  }

  /** @brief Initialize the auxiliary object with precomputed
   *         quantities for order up to max_order for the given
   *         dimensionality.
   */
  void Init(int max_order, int dim) {
    max_order_ = max_order;
    dim_ = dim;
  }

  /** @brief This function assumes that the base is a root of unity,
   *         meaning its magnitude is exactly 1.
   */
  static void PowWithRootOfUnity(const std::complex<double> &base, int power, 
				 std::complex<double> &result) {
    
    // Extract the arg part (angle in radians).
    double complex_arg = atan2(base.imag(), base.real());

    result.real() = cos(power * complex_arg);
    result.imag() = sin(power * complex_arg);
  }
};

#endif
