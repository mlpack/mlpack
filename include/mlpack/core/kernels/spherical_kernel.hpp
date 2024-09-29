/**
 * @file core/kernels/spherical_kernel.hpp
 * @author Neil Slagle
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_KERNELS_SPHERICAL_KERNEL_HPP
#define MLPACK_CORE_KERNELS_SPHERICAL_KERNEL_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * The spherical kernel, which is 1 when the distance between the two argument
 * points is less than or equal to the bandwidth, or 0 otherwise.
 */
class SphericalKernel
{
 public:
  /**
   * Construct the SphericalKernel with the given bandwidth.
   */
  SphericalKernel(const double bandwidth = 1.0) :
    bandwidth(bandwidth),
    bandwidthSquared(std::pow(bandwidth, 2.0))
  { /* Nothing to do. */ }

  // Get the bandwidth.
  double Bandwidth() const { return bandwidth; }
  // Modify the bandwidth.
  void Bandwidth(const double bandwidth)
  {
    this->bandwidth = bandwidth;
    this->bandwidthSquared = bandwidth * bandwidth;
  }

  /**
   * Evaluate the spherical kernel with the given two vectors.
   *
   * @tparam VecTypeA Type of first vector.
   * @tparam VecTypeB Type of second vector.
   * @param a First vector.
   * @param b Second vector.
   * @return The kernel evaluation between the two vectors.
   */
  template<typename VecTypeA, typename VecTypeB>
  double Evaluate(const VecTypeA& a, const VecTypeB& b) const
  {
    return (SquaredEuclideanDistance::Evaluate(a, b) <= bandwidthSquared) ?
        1.0 : 0.0;
  }

  double Normalizer(size_t dimension) const
  {
    return std::pow(bandwidth, (double) dimension) *
        std::pow(M_PI, dimension / 2.0) / std::tgamma(dimension / 2.0 + 1.0);
  }

  /**
   * Evaluate the kernel when only a distance is given, not two points.
   *
   * @param t Argument to kernel.
   */
  double Evaluate(const double t) const
  {
    return (t <= bandwidth) ? 1.0 : 0.0;
  }

  double Gradient(double t)
  {
    return t == bandwidth ? arma::datum::nan : 0.0;
  }

  //! Serialize the object.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(bandwidth));
    ar(CEREAL_NVP(bandwidthSquared));
  }

 private:
  double bandwidth;
  double bandwidthSquared;
};

//! Kernel traits for the spherical kernel.
template<>
class KernelTraits<SphericalKernel>
{
 public:
  //! The spherical kernel is normalized: K(x, x) = 1 for all x.
  static const bool IsNormalized = true;
  //! The spherical kernel doesn't include a squared distance.
  static const bool UsesSquaredDistance = false;
};

} // namespace mlpack

#endif
