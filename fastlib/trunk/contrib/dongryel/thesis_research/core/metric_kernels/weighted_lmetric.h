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

    /** @brief Returns the identifier.
     */
    std::string name() const {
      return std::string("weighted_lmetric");
    }

    /** @brief Sets the scaling factor for each dimension.
     */
    template<typename TableType>
    void set_scales(const TableType &scales_in) {
      scales_.resize(scales_in.n_entries());
      for(int i = 0; i < scales_in.n_entries(); i++) {
        arma::vec scale_per_dimension;
        scales_in.get(i, &scale_per_dimension);
        scales_[i] = scale_per_dimension[0];
      }
    }

    /** @brief Computes the distance metric between two points.
     */
    double Distance(
      const arma::vec &a, const arma::vec &b) const {
      return core::math::Pow<1, t_pow>(DistanceIneq(a, b));
    }

    double DistanceIneq(
      const arma::vec &a, const arma::vec &b) const {
      double distance_ineq = 0;
      for(unsigned int i = 0; i < a.n_elem; i++) {
        distance_ineq +=
          core::math::Pow<t_pow, 1>((a[i] - b[i]) * scales_[i]);
      }
      return distance_ineq;
    }

    /** @brief Computes the distance metric between two points, raised
     *         to a particular power.
     *
     * This might be faster so that you could get, for instance, squared
     * L2 distance.
     */
    double DistanceSq(
      const arma::vec &a, const arma::vec &b) const {

      return core::metric_kernels::LMetricDistanceSqTrait<t_pow>::Compute(
               *this, a, b);
    }
};
};
};

#endif
