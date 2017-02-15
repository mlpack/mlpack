/**
 * @file lmetric.hpp
 * @author Ryan Curtin
 *
 * Generalized L-metric, allowing both squared distances to be returned as well
 * as non-squared distances.  The squared distances are faster to compute.
 *
 * This also gives several convenience typedefs for commonly used L-metrics.
 */
#ifndef MLPACK_CORE_METRICS_MUTUALREACHABILITYMETRIC_IMPL_HPP
#define MLPACK_CORE_METRICS_MUTUALREACHABILITYMETRIC_IMPL_HPP

#include <mlpack/methods/hdbscan/mutualReachabilityMetric.hpp>

namespace mlpack {
namespace hdbscan {
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
typename VecTypeA::elem_type mutualReachabilityMetric::Evaluate(const VecTypeA& a,
    const VecTypeB& b)
{
  typename VecTypeA::elem_type dcoreA = a(a.n_rows-1);
  typename VecTypeB::elem_type dcoreB = b(b.n_rows-1);
  typename VecTypeB::elem_type dist = sqrt(arma::accu(arma::square(a - b) - arma::square(a(a.n_rows-1) - b(b.n_rows-1))));

  return std::max(std::max(dcoreA, dcoreB), dist);
}

} // namespace metric
} // namespace mlpack


#endif
