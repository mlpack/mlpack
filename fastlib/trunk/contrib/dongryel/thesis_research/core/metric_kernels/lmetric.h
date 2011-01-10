/** @file lmetric.h
 *
 *  An implementation of general L_p metric.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_METRIC_KERNELS_LMETRIC_H
#define CORE_METRIC_KERNELS_LMETRIC_H

#include "core/table/dense_point.h"
#include "core/math/math_lib.h"

namespace core {
namespace metric_kernels {

/** @brief A trait class for computing a squared distance.
 */
template<int t_pow>
class LMetricDistanceSqTrait {
  public:
    template<typename LMetricType, typename PointType>
    static double Compute(
      const LMetricType &metric_in,
      const PointType &a, const PointType &b) {
      return core::math::Pow<2, t_pow>(metric_in.DistanceIneq(a, b));
    }
};

/** @brief Template specialization for computing a squared distance
 *         under L2 metric, which avoids a square root operation.
 */
template<>
class LMetricDistanceSqTrait<2> {
  public:
    template<typename LMetricType, typename PointType>
    static double Compute(
      const LMetricType &metric_in,
      const PointType &a, const PointType &b) {
      return metric_in.DistanceIneq(a, b);
    }
};

/** @brief An L_p metric for vector spaces.
 *
 * A generic Metric class should simply compute the distance between
 * two points.  An LMetric operates for integer powers on arma::vec spaces.
 */
template<int t_pow>
class LMetric {
  public:

    /** @brief Computes the distance metric between two points.
     */
    template<typename PointType>
    double Distance(
      const PointType &a, const PointType &b) const {
      return core::math::Pow<1, t_pow>(DistanceIneq(a, b));
    }

    template<typename PointType>
    double DistanceIneq(
      const PointType &a, const PointType &b) const {
      double distance_ineq = 0;
      int length = core::table::LengthTrait<PointType>::length(a);
      for(int i = 0; i < length; i++) {
        distance_ineq += core::math::Pow<t_pow, 1>(a[i] - b[i]);
      }
      return distance_ineq;
    }

    /** @brief Computes the distance metric between two points, raised
     *         to a particular power.
     *
     * This might be faster so that you could get, for instance, squared
     * L2 distance.
     */
    template<typename PointType>
    double DistanceSq(
      const PointType &a, const PointType &b) const {

      return core::metric_kernels::LMetricDistanceSqTrait<t_pow>::Compute(
               *this, a, b);
    }
};
};
};

#endif
