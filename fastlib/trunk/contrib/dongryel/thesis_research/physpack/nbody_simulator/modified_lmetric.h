/** @file modified_lmetric.h
 *
 *  An implementation of general modified L_p metric that returns an
 *  epsilon for very small distance values.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef PHYSPACK_NBODY_SIMULATOR_MODIFIED_LMETRIC_H
#define PHYSPACK_NBODY_SIMULATOR_MODIFIED_LMETRIC_H

#include <armadillo>
#include "core/math/math_lib.h"
#include "core/metric_kernels/abstract_metric.h"

namespace physpack {
namespace nbody_simulator {

template<int t_pow>
class ModifiedLMetricDistanceSqTrait {
  public:
    static double Compute(
      const core::metric_kernels::AbstractMetric &metric_in,
      const core::table::AbstractPoint &a,
      const core::table::AbstractPoint &b) {

      return core::math::Pow<2, t_pow>(metric_in.DistanceIneq(a, b));
    }
};

template<>
class ModifiedLMetricDistanceSqTrait<2> {
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
class ModifiedLMetric: public core::metric_kernels::AbstractMetric {
  public:

    /**
     * Computes the distance metric between two points.
     */
    double Distance(
      const core::table::AbstractPoint& a,
      const core::table::AbstractPoint& b) const {
      return std::max(
               core::math::Pow<1, t_pow>(DistanceIneq(a, b)),
               sqrt(std::numeric_limits<double>::epsilon()));
    }

    double DistanceIneq(
      const core::table::AbstractPoint &a,
      const core::table::AbstractPoint &b) const {
      double distance_ineq = 0;
      for(int i = 0; i < a.length(); i++) {
        distance_ineq += core::math::Pow<t_pow, 1>(a[i] - b[i]);
      }
      return std::max(distance_ineq, std::numeric_limits<double>::epsilon());
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

      return std::max(
               physpack::nbody_simulator::ModifiedLMetricDistanceSqTrait <
               t_pow >::Compute(
                 *this, a, b), std::numeric_limits<double>::epsilon());
    }
};
};
};

#endif
