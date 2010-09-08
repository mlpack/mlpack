#ifndef CORE_METRIC_LMETRIC_H
#define CORE_METRIC_LMETRIC_H

#include <armadillo>
#include "core/math/math_lib.h"

namespace core {
namespace metric_kernels {

/**
 * An L_p metric for vector spaces.
 *
 * A generic Metric class should simply compute the distance between
 * two points.  An LMetric operates for integer powers on arma::vec spaces.
 */
template<int t_pow>
class LMetric {
  public:
    /**
     * Computes the distance metric between two points.
     */
    static double Distance(const arma::vec& a, const arma::vec& b) {
      return arma::norm(a - b, t_pow);
    }

    /**
     * Computes the distance metric between two points, raised to a
     * particular power.
     *
     * This might be faster so that you could get, for instance, squared
     * L2 distance.
     */
    static double DistanceSq(const arma::vec &a, const arma::vec &b) {
      return core::math::Pow<2, 1>(Distance(a, b));
    }
};
};
};

#endif
