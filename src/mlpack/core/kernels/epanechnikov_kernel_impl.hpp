/**
 * @file epanechnikov_kernel_impl.hpp
 * @author Neil Slagle
 *
 * Implementation of template-based Epanechnikov kernel functions.
 *
 * This file is part of MLPACK 1.0.7.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_CORE_KERNELS_EPANECHNIKOV_KERNEL_IMPL_HPP
#define __MLPACK_CORE_KERNELS_EPANECHNIKOV_KERNEL_IMPL_HPP

// In case it hasn't already been included.
#include "epanechnikov_kernel.hpp"

#include <mlpack/core/metrics/lmetric.hpp>

namespace mlpack {
namespace kernel {

template<typename Vec1Type, typename Vec2Type>
inline double EpanechnikovKernel::Evaluate(const Vec1Type& a, const Vec2Type& b)
    const
{
  return std::max(0.0, 1.0 - metric::SquaredEuclideanDistance::Evaluate(a, b)
      * inverseBandwidthSquared);
}

/**
 * Obtains the convolution integral [integral of K(||x-a||) K(||b-x||) dx]
 * for the two vectors.
 *
 * @tparam VecType Type of vector (arma::vec, arma::spvec should be expected).
 * @param a First vector.
 * @param b Second vector.
 * @return the convolution integral value.
 */
template<typename VecType>
double EpanechnikovKernel::ConvolutionIntegral(const VecType& a,
                                               const VecType& b)
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

}; // namespace kernel
}; // namespace mlpack

#endif
