/**
 * @file epanechnikov_kernel.hpp
 * @author Neil Slagle
 *
 * Definition of the Epanechnikov kernel.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_KERNELS_EPANECHNIKOV_KERNEL_HPP
#define MLPACK_CORE_KERNELS_EPANECHNIKOV_KERNEL_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace kernel {

/**
 * The Epanechnikov kernel, defined as
 *
 * @f[
 * K(x, y) = \max \{0, 1 - || x - y ||^2_2 / b^2 \}
 * @f]
 *
 * where @f$ b @f$ is the bandwidth the of the kernel (defaults to 1.0).
 */
class EpanechnikovKernel
{
 public:
  /**
   * Instantiate the Epanechnikov kernel with the given bandwidth (default 1.0).
   *
   * @param bandwidth Bandwidth of the kernel.
   */
  EpanechnikovKernel(const double bandwidth = 1.0) :
      bandwidth(bandwidth),
      inverseBandwidthSquared(1.0 / (bandwidth * bandwidth))
  {  }

  /**
   * Evaluate the Epanechnikov kernel on the given two inputs.
   *
   * @tparam VecTypeA Type of first vector.
   * @tparam VecTypeB Type of second vector.
   * @param a One input vector.
   * @param b The other input vector.
   */
  template<typename VecTypeA, typename VecTypeB>
  double Evaluate(const VecTypeA& a, const VecTypeB& b) const;

  /**
   * Evaluate the Epanechnikov kernel given that the distance between the two
   * input points is known.
   */
  double Evaluate(const double distance) const;

  /**
   * Evaluate the Gradient of Epanechnikov kernel
   * given that the distance between the two
   * input points is known.
   */
  double Gradient(const double distance) const;

  /**
   * Evaluate the Gradient of Epanechnikov kernel
   * given that the squared distance between the two
   * input points is known.
   */
  double GradientForSquaredDistance(const double distanceSquared) const;
  /**
   * Obtains the convolution integral [integral of K(||x-a||) K(||b-x||) dx]
   * for the two vectors.
   *
   * @tparam VecType Type of vector (arma::vec, arma::spvec should be expected).
   * @param a First vector.
   * @param b Second vector.
   * @return the convolution integral value.
   */
  template<typename VecTypeA, typename VecTypeB>
  double ConvolutionIntegral(const VecTypeA& a, const VecTypeB& b);

  /**
   * Compute the normalizer of this Epanechnikov kernel for the given dimension.
   *
   * @param dimension Dimension to calculate the normalizer for.
   */
  double Normalizer(const size_t dimension);

  /**
   * Serialize the kernel.
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int version);

 private:
  //! Bandwidth of the kernel.
  double bandwidth;
  //! Cached value of the inverse bandwidth squared (to speed up computation).
  double inverseBandwidthSquared;

};

//! Kernel traits for the Epanechnikov kernel.
template<>
class KernelTraits<EpanechnikovKernel>
{
 public:
  //! The Epanechnikov kernel is normalized: K(x, x) = 1 for all x.
  static const bool IsNormalized = true;
  //! The Epanechnikov kernel includes a squared distance.
  static const bool UsesSquaredDistance = true;
};

} // namespace kernel
} // namespace mlpack

// Include implementation.
#include "epanechnikov_kernel_impl.hpp"

#endif
