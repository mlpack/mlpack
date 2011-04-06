/** @file kernel.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 *
 *  Common statistical kernels.
 */

#ifndef CORE_METRIC_KERNELS_KERNEL_H
#define CORE_METRIC_KERNELS_KERNEL_H

#include <armadillo>
#include "core/math/math_lib.h"
#include "core/math/range.h"
#include "boost/math/constants/constants.hpp"

namespace core {
namespace metric_kernels {

/** @brief Standard multivariate Gaussian kernel.
 */
class GaussianKernel {
  private:
    double neg_inv_bandwidth_2sq_;
    double bandwidth_sq_;

  public:
    static const bool HAS_CUTOFF = false;

  public:

    template<typename PointType>
    void DrawRandomVariate(
      int num_dimensions_in, PointType *random_variate) const {

      // Draw random $D$ Gaussian variates and scale it by the inverse
      // of the bandwidth.
      random_variate->Init(num_dimensions_in);
      for(int i = 0; i < num_dimensions_in; i++) {
        (*random_variate)[i] =
          core::math::RandGaussian(sqrt(1.0 / bandwidth_sq_));
      }
    }

    std::string name() const {
      return std::string("gaussian");
    }

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
      return core::math::Range(
               EvalUnnormOnSq(range.hi), EvalUnnormOnSq(range.lo));
    }

    /**
     * Gets the maximum unnormalized value.
     */
    double MaxUnnormValue() const {
      return 1.0;
    }

    /**
     * Divide by this constant when you're done.
     */
    double CalcNormConstant(int dims) const {
      return pow(
               2 * boost::math::constants::pi<double>() * bandwidth_sq_,
               dims / 2.0);
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
      critical_point_sq_ = 4 * bandwidth_sq_ * (dims / 2.0 + 2) *
                           boost::math::constants::ln_two<double>();
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
      if(range.lo < critical_point_sq_) {
        if(range.hi < critical_point_sq_) {
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
     */
    double CalcNormConstant(int dims) const {
      return pow(
               boost::math::constants::pi<double>() * 0.5 * bandwidth_sq_,
               dims / 2) / 2;
    }
};

/** @brief Multivariate Epanechnikov kernel.
 *
 * To use, first get an unnormalized density, and divide by the
 * normalizeation factor.
 */
class EpanKernel {
  private:
    double inv_bandwidth_sq_;
    double bandwidth_sq_;

  public:
    static const bool HAS_CUTOFF = true;

  public:

    template<typename PointType>
    void DrawRandomVariate(
      int num_dimensions_in, PointType *random_variate) const {

      // Not implemented yet - implement random binning technique from
      // Rahimi's paper.
    }

    std::string name() const {
      return std::string("epan");
    }

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
      if(sqdist < bandwidth_sq_) {
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
    double MaxUnnormValue() const {
      return 1.0;
    }

    /**
     * Divide by this constant when you're done.
     */
    double CalcNormConstant(int dims) const {
      return 2.0 * core::math::SphereVolume<double>(sqrt(bandwidth_sq_), dims)
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
}
}

#endif
