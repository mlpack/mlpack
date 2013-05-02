/**
 * @file ip_metric_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the IPMetric.
 */
#ifndef __MLPACK_METHODS_FASTMKS_IP_METRIC_IMPL_HPP
#define __MLPACK_METHODS_FASTMKS_IP_METRIC_IMPL_HPP

// In case it hasn't been included yet.
#include "ip_metric_impl.hpp"

#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/core/kernels/linear_kernel.hpp>

namespace mlpack {
namespace fastmks {

// Constructor with no instantiated kernel.
template<typename KernelType>
IPMetric<KernelType>::IPMetric() :
    localKernel(new KernelType()),
    kernel(*localKernel)
{
  // Nothing to do.
}

// Constructor with instantiated kernel.
template<typename KernelType>
IPMetric<KernelType>::IPMetric(KernelType& kernel) :
    localKernel(NULL),
    kernel(kernel)
{
  // Nothing to do.
}

// Destructor for the IPMetric.
template<typename KernelType>
IPMetric<KernelType>::~IPMetric()
{
  if (localKernel != NULL)
    delete localKernel;
}

template<typename KernelType>
template<typename Vec1Type, typename Vec2Type>
inline double IPMetric<KernelType>::Evaluate(const Vec1Type& a,
                                             const Vec2Type& b)
{
  // This is the metric induced by the kernel function.
  // Maybe we can do better by caching some of this?
  return sqrt(kernel.Evaluate(a, a) + kernel.Evaluate(b, b) -
      2 * kernel.Evaluate(a, b));
}

// A specialization for the linear kernel, which actually just turns out to be
// the Euclidean distance.
template<>
template<typename Vec1Type, typename Vec2Type>
inline double IPMetric<kernel::LinearKernel>::Evaluate(const Vec1Type& a,
                                                       const Vec2Type& b)
{
  return metric::LMetric<2, true>::Evaluate(a, b);
}

}; // namespace fastmks
}; // namespace mlpack

#endif
