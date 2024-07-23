/**
 * @file core/distances/ip_metric_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the IPMetric.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DISTANCES_IP_METRIC_IMPL_HPP
#define MLPACK_CORE_DISTANCES_IP_METRIC_IMPL_HPP

// In case it hasn't been included yet.
#include "ip_metric.hpp"

#include <mlpack/core/distances/lmetric.hpp>
#include <mlpack/core/kernels/linear_kernel.hpp>

namespace mlpack {

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
IPMetric<KernelType>::IPMetric(const IPMetric& other) :
  kernel(new KernelType(*other.kernel)),
  kernelOwner(true)
{
  // Nothing to do.
}

template<typename KernelType>
IPMetric<KernelType>& IPMetric<KernelType>::operator=(const IPMetric& other)
{
  if (this == &other)
    return *this;

  if (kernelOwner)
    delete kernel;

  kernel = new KernelType(*other.kernel);
  kernelOwner = true;
  return *this;
}

template<typename KernelType>
template<typename Vec1Type, typename Vec2Type>
inline typename Vec1Type::elem_type IPMetric<KernelType>::Evaluate(
    const Vec1Type& a,
    const Vec2Type& b)
{
  // This is the metric induced by the kernel function.
  // Maybe we can do better by caching some of this?
  return std::sqrt(kernel->Evaluate(a, a) + kernel->Evaluate(b, b) -
      2 * kernel->Evaluate(a, b));
}

// Serialize the kernel.
template<typename KernelType>
template<typename Archive>
void IPMetric<KernelType>::serialize(Archive& ar,
                                     const uint32_t /* version */)
{
  // If we're loading, we need to allocate space for the kernel, and we will own
  // the kernel.
  if (cereal::is_loading<Archive>())
  {
    if (kernelOwner)
      delete kernel;
    kernelOwner = true;
  }

  ar(CEREAL_POINTER(kernel));
}

// A specialization for the linear kernel, which actually just turns out to be
// the Euclidean distance.
template<>
template<typename Vec1Type, typename Vec2Type>
inline typename Vec1Type::elem_type IPMetric<LinearKernel>::Evaluate(
    const Vec1Type& a,
    const Vec2Type& b)
{
  return LMetric<2, true>::Evaluate(a, b);
}

} // namespace mlpack

#endif
