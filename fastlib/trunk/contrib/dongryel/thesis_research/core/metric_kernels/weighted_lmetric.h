/** @file weighted_lmetric.h
 *
 *  An implementation of general weighted L_p metric.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_METRIC_KERNELS_WEIGHTED_LMETRIC_H
#define CORE_METRIC_KERNELS_WEIGHTED_LMETRIC_H

#include <boost/serialization/serialization.hpp>
#include "core/table/dense_point.h"
#include "core/math/math_lib.h"
#include "core/metric_kernels/lmetric.h"

namespace core {
namespace metric_kernels {

/** @brief An L_p metric for vector spaces.
 *
 * A generic Metric class should simply compute the distance between
 * two points.  An WeightedLMetric operates for integer powers on
 * arma::vec spaces.
 */
template<int t_pow>
class WeightedLMetric {

  private:

    // For boost serialization.
    friend class boost::serialization::access;

    std::vector<double> scales_;

  public:

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
    }

    void set_scales(const std::vector<double> &scales_in) {
      scales_.resize(scales_in.size());
      for(unsigned int i = 0; i < scales_in.size(); i++) {
        scales_[i] = scales_in[i];
      }
    }

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
        distance_ineq +=
          core::math::Pow<t_pow, 1>((a[i] - b[i]) / scales_[i]);
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
