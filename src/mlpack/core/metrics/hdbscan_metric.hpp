/**
 * @file hdbscanMetric.hpp
 * @author Sudhanshu Ranjan
 *
 * An implementation of the HDBSCAN metric.
 *  
 */

#ifndef MLPACK_CORE_HDBSCAN_METRIC_HPP
#define MLPACK_CORE_HDBSCAN_METRIC_HPP

namespace mlpack {
namespace metric {

class HdbscanMetric
{
 public:
  /***
   * Default constructor does nothing, but is required to satisfy the Metric
   * policy.
   */
  HdbscanMetric() { }

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
}; // class HdbscanMetric ends

} // namespace metric
} // namespace mlpack

#include "hdbscan_metric_impl.hpp"
#endif
