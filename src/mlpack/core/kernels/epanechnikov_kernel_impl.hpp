/**
 * @file epanechnikov_kernel_impl.hpp
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

#include <mlpack/core/metrics/lmetric.hpp>

namespace mlpack {
namespace kernel {

template<typename VecTypeA, typename VecTypeB>
inline double EpanechnikovKernel::Evaluate(const VecTypeA& a, const VecTypeB& b)
    const
{
  return std::max(0.0, 1.0 - metric::SquaredEuclideanDistance::Evaluate(a, b)
      * inverseBandwidthSquared);
}

/**
 * Obtains the convolution integral [integral of K(||x-a||) K(||b-x||) dx]
 * for the two vectors.
 *
 * @tparam VecTypeA Type of first vector (arma::vec, arma::sp_vec should be
 *      expected).
 * @tparam VecTypeB Type of second vector (arma::vec, arma::sp_vec).
 * @param a First vector.
 * @param b Second vector.
 * @return the convolution integral value.
 */
template<typename VecTypeA, typename VecTypeB>
double EpanechnikovKernel::ConvolutionIntegral(const VecTypeA& a,
                                               const VecTypeB& b)
{
  double distance = sqrt(metric::SquaredEuclideanDistance::Evaluate(a, b));
  if (distance >= 2.0 * bandwidth)
    return 0.0;

  double volumeSquared = std::pow(Normalizer(a.n_rows), 2.0);

  switch (a.n_rows)
  {
    case 1:
      return 1.0 / volumeSquared *
          (16.0 / 15.0 * bandwidth - 4.0 * distance * distance /
          (3.0 * bandwidth) + 2.0 * distance * distance * distance /
          (3.0 * bandwidth * bandwidth) -
          std::pow(distance, 5.0) / (30.0 * std::pow(bandwidth, 4.0)));
      break;
    case 2:
      return 1.0 / volumeSquared *
          ((2.0 / 3.0 * bandwidth * bandwidth - distance * distance) *
          asin(sqrt(1.0 - std::pow(distance / (2.0 * bandwidth), 2.0))) +
          sqrt(4.0 * bandwidth * bandwidth - distance * distance) *
          (distance / 6.0 + 2.0 / 9.0 * distance *
          std::pow(distance / bandwidth, 2.0) - distance / 72.0 *
          std::pow(distance / bandwidth, 4.0)));
      break;
    default:
      Log::Fatal << "EpanechnikovKernel::ConvolutionIntegral(): dimension "
          << a.n_rows << " not supported.";
      return -1.0; // This line will not execute.
      break;
  }
}

//! Serialize the kernel.
template<typename Archive>
void EpanechnikovKernel::Serialize(Archive& ar,
                                   const unsigned int /* version */)
{
  ar & data::CreateNVP(bandwidth, "bandwidth");
  ar & data::CreateNVP(inverseBandwidthSquared, "inverseBandwidthSquared");
}

} // namespace kernel
} // namespace mlpack

#endif
