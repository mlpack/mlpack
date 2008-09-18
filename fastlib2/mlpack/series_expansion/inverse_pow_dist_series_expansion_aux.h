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
  ArrayList<Matrix> multiplicative_constants_;

  /** @brief These are the $A(n, m, \lambda) = (-1)^n (n - m)! 2^m
   *         \left( \frac{\lambda}{2} \right)_m$.
   */
  ArrayList<Matrix> a_constants_;
  
  /** @brief These are the $T(m, k, \lambda) = (-1)^{m + k} (\lambda -
   *         1)_{2k} m choose k$.
   */
  Matrix t_constants_;

  /** @brief Precomputed factorial values.
   */
  Vector factorials_;

  OT_DEF_BASIC(InversePowDistSeriesExpansionAux) {
    OT_MY_OBJECT(dim_);
    OT_MY_OBJECT(max_order_);
    OT_MY_OBJECT(multiplicative_constants_);
    OT_MY_OBJECT(factorials_);
  }

  double Factorial_(int index) {
    
    if(index < 0) {
      return 0;
    }
    else {
      return factorials_[index];
    }
  }

  void ComputeAConstants_() {

    a_constants_.Init(max_order_);

    for(index_t n = 0; n < a_constants_.size(); n++) {
      
      // Allocate $(n + 1)$ by $(n + 1)$ matrix.
      a_constants_[n].Init(n + 1, n + 1);
      a_constants_[n].SetZero();

      // The reference to the matrix.
      Matrix &n_th_order_matrix = a_constants_[n];

      double two_raised_to_m = 1.0;
      for(index_t m = 0; m <= n; m++) {

	for(index_t lambda_index = 0; lambda_index <= m; lambda_index++) {
	  n_th_order_matrix.set
	    (m, lambda_index, Factorial_(n - m) * two_raised_to_m * 
	     PochammerValue(lambda_ / 2.0 + n, m));
	  if(n % 2 == 1) {
	    n_th_order_matrix.set(m, lambda_index,
				  -n_th_order_matrix.get(m, lambda_index));
	  }
	}

	two_raised_to_m *= 2.0;
      }
    }    
  }

  void ComputeTConstants_() {

    t_constants_.Init(max_order_ + 1, max_order_ + 1);
    t_constants_.SetZero();

    for(index_t m = 0; m < t_constants_.n_rows(); m++) {
      for(index_t k = 0; k <= m; k++) {

	t_constants_.set(m, k, PochammerValue(lambda_ - 1, 2 * k) *
			 math::BinomialCoefficient(m, k));
	if((m + k) % 2 == 1) {
	  t_constants_.set(m, k, t_constants_.get(m, k));
	}
      }
    }
  }

  void ComputeFactorials_() {

    factorials_.Init(max_order_ + 1);
    factorials_[0] = 1;
    for(index_t i = 1; i < factorials_.length(); i++) {
      factorials_[i] *= factorials_[i - 1] * i;      
    }
  }

  void ComputeMultiplicativeConstants_() {
    
    multiplicative_constants_.Init(max_order_);

    for(index_t n = 0; n < multiplicative_constants_.size(); n++) {
      
      // Allocate $(n + 1)$ by $(n + 1)$ matrix.
      multiplicative_constants_[n].Init(n + 1, n + 1);
      multiplicative_constants_[n].SetZero();

      // The reference to the matrix.
      Matrix &n_th_order_matrix = multiplicative_constants_[n];

      double two_raised_to_a = 1.0;
      for(index_t a = 0; a <= n; a++) {

	for(index_t b = 0; b <= a; b++) {
	  n_th_order_matrix.set(a, b, math::BinomialCoefficient(n, a) * 
				math::BinomialCoefficient(a, b) / 
				(two_raised_to_a * factorials_[n]));
	  if(n % 2 == 1) {
	    n_th_order_matrix.set(a, b, -n_th_order_matrix.get(a, b));
	  }
	}

	two_raised_to_a *= 2.0;
      }
    }
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

  /** @brief Initialize the auxiliary object with precomputed
   *         quantities for order up to max_order for the given
   *         dimensionality.
   */
  void Init(int max_order, int dim) {
    max_order_ = max_order;
    dim_ = dim;

    // Compute factorial values.
    ComputeFactorials_();

    // Compute multiplicative constants that must be multiplied by
    // evaluating a far field/local expansion.
    ComputeMultiplicativeConstants_();

    ComputeAConstants_();

    ComputeTConstants_();
  }

  double PochammerValue(double base, int index) const {
    
    double result = 1.0;

    for(index_t i = 0; i < index; i++) {
      result *= (base + i);
    }
    return result;
  }

  /** @brief Evaluates the Geganbauer polynomials with the given
   *         argument at several values efficiently using a recurrence
   *         relationship.
   *
   *  @param argument              The argument to the polynomials.
   *  @param evaluated_polynomials The table of evaluated polynomial values.
   */
  void GegenbauerPolynomials(double argument, 
			     Matrix &evaluated_polynomials) {
    
    // lambda_index = lambda / 2 + 1, ...
    for(index_t lambda_index = 0; lambda_index < 
	  evaluated_polynomials.n_cols(); lambda_index++) {
      
      double effective_lambda = lambda_ / 2.0 + lambda_index;
      evaluated_polynomials.set(0, lambda_index, 1.0);
      evaluated_polynomials.set(1, lambda_index, 2 * effective_lambda *
				argument);

      // Apply the recurrence.
      for(index_t n = 2; n < evaluated_polynomials.n_rows(); n++) {
	evaluated_polynomials.set
	  (n, lambda_index, 2 * (n + effective_lambda - 1) * argument *
	   evaluated_polynomials.get(n - 1, lambda_index) -
	   (n + 2 * effective_lambda - 2) *
	   evaluated_polynomials.get(n - 2, lambda_index));
      }
    }
  }

  /** @brief Computes the required P factor.
   *
   *  @param evaluated_polynomial The table of evaluated Gegenbauer
   *         polynomials using $cos(theta)$.
   */
  void ComputePFactor(double radius, double theta, double phi,
		      int n, int m, double lambda,
		      const Matrix &evaluated_polynomials,
		      std::complex<double> &result) {
    
    // Common factor.
    double common_factor = pow(sin(theta), m) / pow(radius, n + lambda);

    // Compute the real part.
    result.real() = common_factor * cos(m * phi) * 
      evaluated_polynomials.get(n - m, m);

    // Then compute the imaginary part.
    result.imag() = common_factor * sin(m * phi) *
      evaluated_polynomials.get(n - m, m);
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
