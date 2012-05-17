/**
 * @file ip_metric.hpp
 * @author Ryan Curtin
 *
 * Inner product induced metric.  If given a kernel function, this gives the
 * complementary metric.
 */
#ifndef __MLPACK_METHODS_MAXIP_IP_METRIC_HPP
#define __MLPACK_METHODS_MAXIP_IP_METRIC_HPP

namespace mlpack {
namespace maxip /** The maximum inner product problem. */ {

template<typename KernelType>
class IPMetric
{
 public:
  typedef KernelType Kernel;

  /**
   * Create the IPMetric.
   */
  IPMetric() { }

  /**
   * Evaluate the metric.
   */
  template<typename Vec1Type, typename Vec2Type>
  static double Evaluate(const Vec1Type& a, const Vec2Type& b);
};

}; // namespace maxip
}; // namespace mlpack

// Include implementation.
#include "ip_metric_impl.hpp"

#endif
