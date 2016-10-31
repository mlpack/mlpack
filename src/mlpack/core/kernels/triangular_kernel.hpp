/**
 * @file triangular_kernel.hpp
 * @author Ryan Curtin
 *
 * Definition and implementation of the trivially simple triangular kernel.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_KERNELS_TRIANGULAR_KERNEL_HPP
#define MLPACK_CORE_KERNELS_TRIANGULAR_KERNEL_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/metrics/lmetric.hpp>

namespace mlpack {
namespace kernel {

/**
 * The trivially simple triangular kernel, defined by
 *
 * @f[
 * K(x, y) = \max \{ 0, 1 - \frac{|| x - y ||_2}{b} \}
 * @f]
 *
 * where \f$ b \f$ is the bandwidth of the kernel.
 */
class TriangularKernel
{
 public:
  /**
   * Initialize the triangular kernel with the given bandwidth (default 1.0).
   *
   * @param bandwidth Bandwidth of the triangular kernel.
   */
  TriangularKernel(const double bandwidth = 1.0) : bandwidth(bandwidth) { }

  /**
   * Evaluate the triangular kernel for the two given vectors.
   *
   * @tparam VecTypeA Type of first vector.
   * @tparam VecTypeB Type of second vector.
   * @param a First vector.
   * @param b Second vector.
   */
  template<typename VecTypeA, typename VecTypeB>
  double Evaluate(const VecTypeA& a, const VecTypeB& b) const
  {
    return std::max(0.0, (1 - metric::EuclideanDistance::Evaluate(a, b) /
        bandwidth));
  }

  /**
   * Evaluate the triangular kernel given that the distance between the two
   * points is known.
   *
   * @param distance The distance between the two points.
   */
  double Evaluate(const double distance) const
  {
    return std::max(0.0, (1 - distance) / bandwidth);
  }

  /**
   * Evaluate the gradient of triangular kernel
   * given that the distance between the two
   * points is known.
   *
   * @param distance The distance between the two points.
   */
  double Gradient(const double distance) const {
    if (distance < 1) {
      return -1.0 / bandwidth;
    } else if (distance > 1) {
      return 0;
    } else {
      return arma::datum::nan;
    }
  }

  //! Get the bandwidth of the kernel.
  double Bandwidth() const { return bandwidth; }
  //! Modify the bandwidth of the kernel.
  double& Bandwidth() { return bandwidth; }

  //! Serialize the kernel.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(bandwidth, "bandwidth");
  }

 private:
  //! The bandwidth of the kernel.
  double bandwidth;
};

//! Kernel traits for the triangular kernel.
template<>
class KernelTraits<TriangularKernel>
{
 public:
  //! The triangular kernel is normalized: K(x, x) = 1 for all x.
  static const bool IsNormalized = true;
  //! The triangular kernel doesn't include a squared distance.
  static const bool UsesSquaredDistance = false;
};

} // namespace kernel
} // namespace mlpack

#endif
