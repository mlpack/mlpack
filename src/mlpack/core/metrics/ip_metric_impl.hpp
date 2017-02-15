/**
 * @file ip_metric_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the IPMetric.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_FASTMKS_IP_METRIC_IMPL_HPP
#define MLPACK_METHODS_FASTMKS_IP_METRIC_IMPL_HPP

// In case it hasn't been included yet.
#include "ip_metric_impl.hpp"

#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/core/kernels/linear_kernel.hpp>

namespace mlpack {
namespace metric {

// Constructor with no instantiated kernel.
template<typename KernelType>
IPMetric<KernelType>::IPMetric() :
    kernel(new KernelType()),
    kernelOwner(true)
{
  // Nothing to do.
}

// Constructor with instantiated kernel.
template<typename KernelType>
IPMetric<KernelType>::IPMetric(KernelType& kernel) :
    kernel(&kernel),
    kernelOwner(false)
{
  // Nothing to do.
}

// Destructor for the IPMetric.
template<typename KernelType>
IPMetric<KernelType>::~IPMetric()
{
  if (kernelOwner)
    delete kernel;
}

template<typename KernelType>
template<typename Vec1Type, typename Vec2Type>
inline typename Vec1Type::elem_type IPMetric<KernelType>::Evaluate(
    const Vec1Type& a,
    const Vec2Type& b)
{
  // This is the metric induced by the kernel function.
  // Maybe we can do better by caching some of this?
  return sqrt(kernel->Evaluate(a, a) + kernel->Evaluate(b, b) -
      2 * kernel->Evaluate(a, b));
}

// Serialize the kernel.
template<typename KernelType>
template<typename Archive>
void IPMetric<KernelType>::Serialize(Archive& ar,
                                     const unsigned int /* version */)
{
  // If we're loading, we need to allocate space for the kernel, and we will own
  // the kernel.
  if (Archive::is_loading::value)
    kernelOwner = true;

  ar & data::CreateNVP(kernel, "kernel");
}

// A specialization for the linear kernel, which actually just turns out to be
// the Euclidean distance.
template<>
template<typename Vec1Type, typename Vec2Type>
inline typename Vec1Type::elem_type IPMetric<kernel::LinearKernel>::Evaluate(
    const Vec1Type& a,
    const Vec2Type& b)
{
  return metric::LMetric<2, true>::Evaluate(a, b);
}

} // namespace metric
} // namespace mlpack

#endif
