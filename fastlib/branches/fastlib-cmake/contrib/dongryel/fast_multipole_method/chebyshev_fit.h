#ifndef CHEBYSHEV_FIT_H
#define CHEBYSHEV_FIT_H

#include "fastlib/fastlib.h"

class ChebyshevFit {

 public:

  static const double pi_ = 3.141592653589793;

  int total_num_coeffs_;

  DRange approximation_interval_;

  Vector coefficients_;

 public:

  template<typename TKernel>
  void Init(const DRange &interval_in, int order, TKernel *kernel) {
    
    // Copy the interval.
    approximation_interval_ = interval_in;

    // Allocate the coefficients.
    total_num_coeffs_ = order;
    coefficients_.Init(order);
    coefficients_.SetZero();

    // Temporary space needed for computation.
    Vector tmp_vector;
    tmp_vector.Init(order);
    tmp_vector.SetZero();

    // Transform the interval to [-1, 1]
    DRange transformed_interval;
    transformed_interval.lo = 0.5 * (interval_in.hi - interval_in.lo);
    transformed_interval.hi = 0.5 * (interval_in.hi + interval_in.lo);

    for(index_t k = 0; k < order; k++) {
      double grid_val = cos(pi_ * (k + 0.5) / ((double) order));
      tmp_vector[k] = 
	kernel->EvalUnnorm(grid_val * transformed_interval.lo +
			   transformed_interval.hi);      
    }

    double factor = 2.0 / ((double) order);
    for(index_t j = 0; j < order; j++) {
      double sum = 0.0;
      for(index_t k = 0; k < order; k++) {
	sum += tmp_vector[k] * cos(pi_ * j * (k + 0.5) / ((double) order));
      }
      coefficients_[j] = factor * sum;
    }
  }

  double Evaluate(double x, int order) {
    
    double d = 0.0;
    double dd = 0.0;
    double sv = 0.0;
    if((x - approximation_interval_.lo) *
       (x - approximation_interval_.hi) > 0.0) {

      printf("x not in range in Chebyshev!\n");
      return 0;
    }
    double y = (2.0 * x - approximation_interval_.lo -
		approximation_interval_.hi) / 
      (approximation_interval_.hi - approximation_interval_.lo);
    double y2 = 2.0 * y;

    for(index_t j = order - 1; j > 0; j--) {
      sv = d;
      d = y2 * d - dd + coefficients_[j];
      dd = sv;
    }
    return y * d - dd + 0.5 * coefficients_[0];    
  }

  void ConvertToTaylorExpansion(int order, Vector *result) {

    Vector dd;
    result->Init(order);
    result->SetZero();
    dd.Init(order);
    dd.SetZero();
    
    double sv;
    ((*result)[0]) = coefficients_[order - 1];
    
    for(index_t j = order - 2; j > 0; j--) {
      for(index_t k = order - j; k > 0; k--) {
	sv = ((*result)[k]);
	((*result)[k]) = 2.0 * ((*result)[k - 1]) - dd[k];
	dd[k] = sv;
      }
      sv = (*result)[0];
      ((*result)[0]) = -dd[0] + coefficients_[j];
      dd[0] = sv;
    }

    for(index_t j = order - 1; j > 0; j--) {
      ((*result)[j]) = ((*result)[j - 1]) - dd[j];
    }
    ((*result)[0]) = -dd[0] + 0.5 * coefficients_[0];    
  }

  void ConvertToTaylorExpansionOriginalVariable(int order, Vector *result) {
    
    ConvertToTaylorExpansion(order, result);

    double cnst = 2.0 / approximation_interval_.width();
    double fac = cnst;
    for(index_t j = 1; j < result->length(); j++) {
      ((*result)[j]) *= fac;
      fac *= cnst;
    }
    cnst = approximation_interval_.mid();
    
    for(index_t j = 0; j <= result->length() - 2; j++) {
      for(index_t k = result->length() - 2; k >= j; k--) {
	((*result)[k]) -= cnst * ((*result)[k + 1]);
      }
    }
  }

  /** @brief Computes an estimate of the truncation error after the
   *         expansion is cut-off after the prescribed order.
   */
  double TruncationError(int order) {

    double error = 0;

    for(index_t m = order + 1; m < coefficients_.length(); m++) {

      error += fabs(coefficients_[m]);
    }

    return error;
  }

};

#endif
