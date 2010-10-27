/** @file lmetric.h
 *
 *  An implementation of general L_p metric.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_METRIC_KERNELS_LMETRIC_H
#define CORE_METRIC_KERNELS_LMETRIC_H

#include <armadillo>
#include "core/math/math_lib.h"
#include "core/metric_kernels/abstract_metric.h"

namespace core {
namespace metric_kernels {

template<int t_pow>
class LMetricDistanceSqTrait {
  public:
    static double Compute(
      const core::metric_kernels::AbstractMetric &metric_in,
      const core::table::AbstractPoint &a,
      const core::table::AbstractPoint &b) {

      return core::math::Pow<2, t_pow>(metric_in.DistanceIneq(a, b));
    }
};

template<>
class LMetricDistanceSqTrait<2> {
  public:
    static double Compute(
      const core::metric_kernels::AbstractMetric &metric_in,
      const core::table::AbstractPoint &a,
      const core::table::AbstractPoint &b) {

      return metric_in.DistanceIneq(a, b);
    }
};

/**
 * An L_p metric for vector spaces.
 *
 * A generic Metric class should simply compute the distance between
 * two points.  An LMetric operates for integer powers on arma::vec spaces.
 */
template<int t_pow>
class LMetric: public core::metric_kernels::AbstractMetric {
  public:

    /**
     * Computes the distance metric between two points.
     */
    double Distance(
      const core::table::AbstractPoint& a,
      const core::table::AbstractPoint& b) const {
      return core::math::Pow<1, t_pow>(DistanceIneq(a, b));
    }

    double DistanceIneq(
      const core::table::AbstractPoint &a,
      const core::table::AbstractPoint &b) const {

      double distance_ineq = 0;
      for(int i = 0; i < a.length(); i++) {
        distance_ineq += core::math::Pow<t_pow, 1>(a[i] - b[i]);
      }
      return distance_ineq;
    }

    /**
     * Computes the distance metric between two points, raised to a
     * particular power.
     *
     * This might be faster so that you could get, for instance, squared
     * L2 distance.
     */
    double DistanceSq(
      const core::table::AbstractPoint &a,
      const core::table::AbstractPoint &b) const {

      return core::metric_kernels::LMetricDistanceSqTrait<t_pow>::Compute(
               *this, a, b);
    }
};
};
};

#endif
