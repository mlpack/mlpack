/**
 * @file ip_metric_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the IPMetric.
 */
#ifndef __MLPACK_METHODS_MAXIP_IP_METRIC_IMPL_HPP
#define __MLPACK_METHODS_MAXIP_IP_METRIC_IMPL_HPP

// In case it hasn't been included yet.
#include "ip_metric_impl.hpp"

#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/core/kernels/linear_kernel.hpp>

namespace mlpack {
namespace maxip {

template<typename KernelType>
template<typename Vec1Type, typename Vec2Type>
inline double IPMetric<KernelType>::Evaluate(const Vec1Type& a, const Vec2Type& b)
{
  // This is the metric induced by the kernel function.
  // Maybe we can do better by caching some of this?
  ++distanceEvaluations;
  return KernelType::Evaluate(a, a) + KernelType::Evaluate(b, b) -
      2 * KernelType::Evaluate(a, b);
}

template<>
template<typename Vec1Type, typename Vec2Type>
inline double IPMetric<kernel::LinearKernel>::Evaluate(const Vec1Type& a,
                                                       const Vec2Type& b)
{
  ++distanceEvaluations;
  return metric::LMetric<2>::Evaluate(a, b);
}

}; // namespace maxip
}; // namespace mlpack

#endif
