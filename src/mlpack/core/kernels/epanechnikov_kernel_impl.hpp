/**
 * @file core/kernels/epanechnikov_kernel_impl.hpp
 * @author Neil Slagle
 *
 * Implementation of template-based Epanechnikov kernel functions.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_KERNELS_EPANECHNIKOV_KERNEL_IMPL_HPP
#define MLPACK_CORE_KERNELS_EPANECHNIKOV_KERNEL_IMPL_HPP

// In case it hasn't already been included.
#include "epanechnikov_kernel.hpp"
#include <mlpack/core/util/log.hpp>

#include <mlpack/core/distances/lmetric.hpp>

namespace mlpack {

template<typename VecTypeA, typename VecTypeB>
inline double EpanechnikovKernel::Evaluate(const VecTypeA& a, const VecTypeB& b)
    const
{
  return std::max(0.0, 1.0 - SquaredEuclideanDistance::Evaluate(a, b)
      * inverseBandwidthSquared);
}

/**
 * Compute the normalizer of this Epanechnikov kernel for the given dimension.
 *
 * @param dimension Dimension to calculate the normalizer for.
 */
inline double EpanechnikovKernel::Normalizer(const size_t dimension)
{
  return 2.0 * std::pow(bandwidth, (double) dimension) *
      std::pow(M_PI, dimension / 2.0) /
      (std::tgamma(dimension / 2.0 + 1.0) * (dimension + 2.0));
}

/**
 * Evaluate the kernel not for two points but for a numerical value.
 */
inline double EpanechnikovKernel::Evaluate(const double distance) const
{
  return std::max(0.0, 1 - std::pow(distance, 2.0) * inverseBandwidthSquared);
}

/**
 * Evaluate gradient of the kernel not for two points
 * but for a numerical value.
 */
inline double EpanechnikovKernel::Gradient(const double distance) const
{
  if (std::abs(bandwidth) < std::abs(distance))
  {
    return 0;
  }
  else if (std::abs(bandwidth) > std::abs(distance))
  {
    return -2 * inverseBandwidthSquared * distance;
  }
  else
  {
    // The gradient doesn't exist.
    return arma::datum::nan;
  }
}

//! Serialize the kernel.
template<typename Archive>
void EpanechnikovKernel::serialize(Archive& ar,
                                   const uint32_t /* version */)
{
  ar(CEREAL_NVP(bandwidth));
  ar(CEREAL_NVP(inverseBandwidthSquared));
}

} // namespace mlpack

#endif
