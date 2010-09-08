/**
 * @file kernel.h
 *
 * Common statistical kernels.
 */

#ifndef CORE_METRIC_KERNELS_KERNEL_H
#define CORE_METRIC_KERNELS_KERNEL_H

#include "abstract_kernel.h"
#include "core/math/math_lib.h"
#include "core/math/range.h"
#include "core/math/geometry.h"
#include "core/math/discrete.h"

namespace core {
namespace metric_kernels {

/**
 * Standard multivariate Gaussian kernel.
 *
 */
class GaussianKernel: public AbstractKernel {
  private:
    double neg_inv_bandwidth_2sq_;
    double bandwidth_sq_;

  public:
    static const bool HAS_CUTOFF = false;

  public:
    double bandwidth_sq() const {
      return bandwidth_sq_;
    }

    void Init(double bandwidth_in, int dims) {
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
     * Evaluates an unnormalized density, given the distance between
     * the kernel's mean and a query point.
     */
    double EvalUnnorm(double dist) const {
      return EvalUnnormOnSq(dist * dist);
    }

    /**
     * Evaluates an unnormalized density, given the square of the
     * distance.
     */
    double EvalUnnormOnSq(double sqdist) const {
      double d = exp(sqdist * neg_inv_bandwidth_2sq_);
      return d;
    }

    /** Unnormalized range on a range of squared distances. */
    core::math::Range RangeUnnormOnSq(const core::math::Range& range) const {
      return core::math::Range(EvalUnnormOnSq(range.hi), EvalUnnormOnSq(range.lo));
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
    double CalcNormConstant(int dims) const {
      return pow(2 * math::PI * bandwidth_sq_, dims / 2.0);
    }
};

class GaussianStarKernel {
  private:
    double neg_inv_bandwidth_2sq_;
    double factor_;
    double bandwidth_sq_;
    double critical_point_sq_;
    double critical_point_value_;

  public:
    static const bool HAS_CUTOFF = false;

  public:
    double bandwidth_sq() const {
      return bandwidth_sq_;
    }

    /**
     * Initializes to a specific bandwidth.
     *
     * @param bandwidth_in the standard deviation sigma
     */
    void Init(double bandwidth_in, int dims) {
      bandwidth_sq_ = bandwidth_in * bandwidth_in;
      neg_inv_bandwidth_2sq_ = -1.0 / (2.0 * bandwidth_sq_);
      factor_ = pow(2.0, -dims / 2.0 - 1);
      critical_point_sq_ = 4 * bandwidth_sq_ * (dims / 2.0 + 2) * math::LN_2;
      critical_point_value_ = EvalUnnormOnSq(critical_point_sq_);
    }

    /**
     * Evaluates an unnormalized density, given the distance between
     * the kernel's mean and a query point.
     */
    double EvalUnnorm(double dist) const {
      return EvalUnnormOnSq(dist * dist);
    }

    /**
     * Evaluates an unnormalized density, given the square of the
     * distance.
     */
    double EvalUnnormOnSq(double sqdist) const {
      double d =
        factor_ * exp(sqdist * neg_inv_bandwidth_2sq_ * 0.5)
        - exp(sqdist * neg_inv_bandwidth_2sq_);
      return d;
    }

    /** Unnormalized range on a range of squared distances. */
    core::math::Range RangeUnnormOnSq(const core::math::Range& range) const {
      double eval_lo = EvalUnnormOnSq(range.lo);
      double eval_hi = EvalUnnormOnSq(range.hi);
      if (range.lo < critical_point_sq_) {
        if (range.hi < critical_point_sq_) {
          // Strictly under critical point.
          return core::math::Range(eval_lo, eval_hi);
        }
        else {
          // Critical point is included
          return core::math::Range(std::min(eval_lo, eval_hi), critical_point_value_);
        }
      }
      else {
        return core::math::Range(eval_hi, eval_lo);
      }
    }

    /**
     * Divide by this constant when you're done.
     *
     * @deprecated -- this function is very confusing
     */
    double CalcNormConstant(int dims) const {
      return pow(math::PI_2*bandwidth_sq_, dims / 2) / 2;
    }
};

/**
 * Multivariate Epanechnikov kernel.
 *
 * To use, first get an unnormalized density, and divide by the
 * normalizeation factor.
 */
class EpanKernel: public AbstractKernel {
  private:
    double inv_bandwidth_sq_;
    double bandwidth_sq_;

  public:
    static const bool HAS_CUTOFF = true;

  public:
    void Init(double bandwidth_in, int dims) {
      Init(bandwidth_in);
    }

    /**
     * Initializes to a specific bandwidth.
     */
    void Init(double bandwidth_in) {
      bandwidth_sq_ = (bandwidth_in * bandwidth_in);
      inv_bandwidth_sq_ = 1.0 / bandwidth_sq_;
    }

    /**
     * Evaluates an unnormalized density, given the distance between
     * the kernel's mean and a query point.
     */
    double EvalUnnorm(double dist) const {
      return EvalUnnormOnSq(dist * dist);
    }

    /**
     * Evaluates an unnormalized density, given the square of the
     * distance.
     */
    double EvalUnnormOnSq(double sqdist) const {
      if (sqdist < bandwidth_sq_) {
        return 1 - sqdist * inv_bandwidth_sq_;
      }
      else {
        return 0;
      }
    }

    /** Unnormalized range on a range of squared distances. */
    core::math::Range RangeUnnormOnSq(const core::math::Range& range) const {
      return core::math::Range(
               EvalUnnormOnSq(range.hi), EvalUnnormOnSq(range.lo));
    }

    /**
     * Gets the maximum unnormalized value.
     */
    double MaxUnnormValue() {
      return 1.0;
    }

    /**
     * Divide by this constant when you're done.
     */
    double CalcNormConstant(int dims) const {
      return 2.0 * math::SphereVolume(sqrt(bandwidth_sq_), dims)
             / (dims + 2.0);
    }

    /**
     * Gets the squared bandwidth.
     */
    double bandwidth_sq() const {
      return bandwidth_sq_;
    }

    /**
    * Gets the reciproccal of the squared bandwidth.
     */
    double inv_bandwidth_sq() const {
      return inv_bandwidth_sq_;
    }
};
};
};

#endif
