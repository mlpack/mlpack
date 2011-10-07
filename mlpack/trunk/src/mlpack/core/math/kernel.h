/**
 * @file kernel.h
 *
 * Common statistical kernels.
 */

#ifndef MATH_KERNEL_H
#define MATH_KERNEL_H

#include "math_lib.h"

#include <math.h>

/* More to come soon - Gaussian, Epanechnikov, etc. */

/**
 * Standard multivariate Gaussian kernel.
 *
 */
class GaussianKernel {
 private:
  double neg_inv_bandwidth_2sq_;
  double bandwidth_sq_;

 public:
  static const bool HAS_CUTOFF = false;

 public:
  double bandwidth_sq() const {
    return bandwidth_sq_;
  }

  void Init(double bandwidth_in, size_t dims) {
    Init(bandwidth_in);
  }

  /**
   * Initializes to a specific bandwidth.
   *
   * @param bandwidth_in the standard deviation sigma
   */
  void Init(double bandwidth_in) {
    bandwidth_sq_ = bandwidth_in * bandwidth_in;
    neg_inv_bandwidth_2sq_ = -1.0 / (2.0 * bandwidth_sq_);
  }

  /**
   * Evaluates an nonnormalized density, given the distance between
   * the kernel's mean and a query point.
   */
  double EvalUnnorm(double dist) const {
    return EvalUnnormOnSq(dist * dist);
  }

  /**
   * Evaluates an nonnormalized density, given the square of the
   * distance.
   */
  double EvalUnnormOnSq(double sqdist) const {
    double d = exp(sqdist * neg_inv_bandwidth_2sq_);
    return d;
  }

  /** Nonnormalized range on a range of squared distances. */
  Range RangeUnnormOnSq(const Range& range) const {
    return Range(EvalUnnormOnSq(range.hi), EvalUnnormOnSq(range.lo));  //!! TODO explain
  }

  /**
   * Gets the maximum unnormalized value.
   */
  double MaxUnnormValue() {
    return 1;
  }

  /**
   * Divide by this constant when you're done.
   */
  double CalcNormConstant(size_t dims) const {
    // Changed because * faster than / and 2 * M_PI opt out.  RR
    //return pow((-math::PI/neg_inv_bandwidth_2sq_), dims/2.0);
    return pow(2 * M_PI * bandwidth_sq_, dims / 2.0);
  }
};

#endif
