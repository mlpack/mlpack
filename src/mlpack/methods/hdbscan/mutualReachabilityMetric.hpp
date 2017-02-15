/**
 * @file lmetric.hpp
 * @author Ryan Curtin
 *
 * Generalized L-metric, allowing both squared distances to be returned as well
 * as non-squared distances.  The squared distances are faster to compute.
 *
 * This also gives several convenience typedefs for commonly used L-metrics.
 */
#ifndef MLPACK_CORE_METRICS_MUTUALREACHABILITYMETRIC_HPP
#define MLPACK_CORE_METRICS_MUTUALREACHABILITYMETRIC_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace hdbscan {

class mutualReachabilityMetric
{
 public:
  
  mutualReachabilityMetric() { }

  /**
   * Computes the distance between two points.
   *
   * @tparam VecTypeA Type of first vector (generally arma::vec or
   *      arma::sp_vec).
   * @tparam VecTypeB Type of second vector.
   * @param a First vector.
   * @param b Second vector.
   * @return Distance between vectors a and b.
   */
  template<typename VecTypeA, typename VecTypeB>
  static typename VecTypeA::elem_type Evaluate(const VecTypeA& a,
                                               const VecTypeB& b);
};

} // namespace metric
} // namespace mlpack

// Include implementation.
#include "mlpack/methods/hdbscan/mutualReachabilityMetric_impl.hpp"

#endif
