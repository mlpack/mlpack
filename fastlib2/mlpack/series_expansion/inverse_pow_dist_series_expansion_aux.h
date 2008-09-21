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

    a_constants_.Init(2 * (max_order_ + 1));

    for(index_t n = 0; n < a_constants_.size(); n++) {
      
      // Allocate $(n + 1)$ by $(n + 1)$ matrix.
      a_constants_[n].Init(n + 1, 2 * max_order_ + 1);
      a_constants_[n].SetZero();

      // The reference to the matrix.
      Matrix &n_th_order_matrix = a_constants_[n];

      for(index_t m = 0; m < a_constants_[n].n_rows(); m++) {

	for(index_t lambda_index = 0; lambda_index < a_constants_[n].n_cols(); 
	    lambda_index++) {
	  n_th_order_matrix.set
	    (m, lambda_index, Factorial_(n - m) * pow(2, m) * 
	     PochammerValue(lambda_ / 2.0 + lambda_index, m));
	  if(n % 2 == 1) {
	    n_th_order_matrix.set(m, lambda_index,
				  -n_th_order_matrix.get(m, lambda_index));
	  }
	}
      }
    }
  }

  void ComputeTConstants_() {

    t_constants_.Init(2 * (max_order_ + 1), 2 * (max_order_ + 1));
    t_constants_.SetZero();

    for(index_t m = 0; m < t_constants_.n_rows(); m++) {
      for(index_t k = 0; k <= m; k++) {

	t_constants_.set(m, k, PochammerValue(lambda_ - 1, 2 * k) *
			 math::BinomialCoefficient(m, k));
	if((m + k) % 2 == 1) {
	  t_constants_.set(m, k, -t_constants_.get(m, k));
	}
      }
    }
  }

  void ComputeFactorials_() {

    factorials_.Init(2 * (max_order_ + 1));
    factorials_[0] = 1;
    for(index_t i = 1; i < factorials_.length(); i++) {
      factorials_[i] = factorials_[i - 1] * i;      
    }
  }

  void ComputeMultiplicativeConstants_() {
    
    multiplicative_constants_.Init(2 * (max_order_ + 1));

    for(index_t n = 0; n < multiplicative_constants_.size(); n++) {
      
      // Allocate $(n + 1)$ by $(n + 1)$ matrix.
      multiplicative_constants_[n].Init(n + 1, n + 1);
      multiplicative_constants_[n].SetZero();

      // The reference to the matrix.
      Matrix &n_th_order_matrix = multiplicative_constants_[n];

      for(index_t a = 0; a <= n; a++) {

	for(index_t b = 0; b <= a; b++) {
	  n_th_order_matrix.set(a, b, math::BinomialCoefficient(n, a) * 
				math::BinomialCoefficient(a, b) / 
				(pow(2, a) * factorials_[n]));
	  if(n % 2 == 1) {
	    n_th_order_matrix.set(a, b, -n_th_order_matrix.get(a, b));
	  }
	}
      }
    }
  }

 public:

  const ArrayList<Matrix> *get_multiplicative_constants() const {
    return &multiplicative_constants_;
  }

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
  void Init(double incoming_lambda, int max_order, int dim) {
    
    lambda_ = incoming_lambda;
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
			     Matrix &evaluated_polynomials) const {

    // First initialize the matrix to zeros.
    evaluated_polynomials.SetZero();

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
	  (n, lambda_index, 
	   (2 * (n + effective_lambda - 1) * argument *
	    evaluated_polynomials.get(n - 1, lambda_index) -
	    (n + 2 * effective_lambda - 2) *
	    evaluated_polynomials.get(n - 2, lambda_index)) / ((double) n));
      }
    }
  }

  /** @brief Computes the required P factor.
   *
   *  @param evaluated_polynomial The table of evaluated Gegenbauer
   *         polynomials using $cos(theta)$.
   */
  void ComputePFactor(double radius, double theta, double phi,
		      int n, int m, int k, double lambda,
		      const Matrix &evaluated_polynomials,
		      std::complex<double> &result) const {
    
    // Common factor.
    double common_factor = pow(sin(theta), m) / pow(radius, n + lambda);

    // Compute the real part.
    result.real() = common_factor * cos(m * phi) * 
      evaluated_polynomials.get(n - m, k + m);

    // Then compute the imaginary part.
    result.imag() = common_factor * sin(m * phi) *
      evaluated_polynomials.get(n - m, k + m);
  }

  /** @brief Converts a 3-D coordinate into its spherical coordinate
   *         representation.
   */
  static void ConvertCartesianToSpherical(double x, double y, double z,
					  double *radius, double *theta,
					  double *phi) {

    *radius = sqrt(math::Sqr(x) + math::Sqr(y) + math::Sqr(z));
    *theta = atan2(sqrt(math::Sqr(x) + math::Sqr(y)), z);
    *phi = atan2(y, x);
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

  /** @brief This function comptues the partial derivative factor.
   */
  void ComputePartialDerivativeFactor
  (int n, int a, int b, double radius, double theta, double phi,
   const Matrix &evaluated_polynomials, std::complex<double> &result) const {
    
    int m = std::min(b, a - b);
    std::complex<double> tmp;
    result.real() = result.imag() = 0;

    for(index_t k = 0; k <= m; k++) {
      double common_factor = t_constants_.get(m, k) * 
	a_constants_[n - 2 * k].get(abs(a - 2 * b), k);
      int sign = 2 * b - a;
      if(sign > 0) {
	sign = 1;
      }
      else if(sign < 0) {
	sign = -1;
      }
      else {
	sign = 0;
      }

      ComputePFactor(radius, theta, sign * phi, n - 2 * k, abs(a - 2 * b), k,
		     lambda_ + 2 * k, evaluated_polynomials, tmp);

      result.real() += common_factor * tmp.real();
      result.imag() += common_factor * tmp.imag();
    }
  }

};

#endif
