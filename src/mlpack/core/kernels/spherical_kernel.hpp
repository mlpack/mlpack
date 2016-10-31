/**
 * @file spherical_kernel.hpp
 * @author Neil Slagle
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_KERNELS_SPHERICAL_KERNEL_HPP
#define MLPACK_CORE_KERNELS_SPHERICAL_KERNEL_HPP

#include <boost/math/special_functions/gamma.hpp>
#include <mlpack/core.hpp>

namespace mlpack {
namespace kernel {

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
    return
        (metric::SquaredEuclideanDistance::Evaluate(a, b) <= bandwidthSquared) ?
        1.0 : 0.0;
  }
  /**
   * Obtains the convolution integral [integral K(||x-a||)K(||b-x||)dx]
   * for the two vectors.
   *
   * @tparam VecTypeA Type of first vector (arma::vec, arma::sp_vec should be
   *       expected).
   * @tparam VecTypeB Type of second vector.
   * @param a First vector.
   * @param b Second vector.
   * @return the convolution integral value.
   */
  template<typename VecTypeA, typename VecTypeB>
  double ConvolutionIntegral(const VecTypeA& a, const VecTypeB& b) const
  {
    double distance = sqrt(metric::SquaredEuclideanDistance::Evaluate(a, b));
    if (distance >= 2.0 * bandwidth)
    {
      return 0.0;
    }
    double volumeSquared = pow(Normalizer(a.n_rows), 2.0);

    switch(a.n_rows)
    {
      case 1:
        return 1.0 / volumeSquared * (2.0 * bandwidth - distance);
        break;
      case 2:
        return 1.0 / volumeSquared *
          (2.0 * bandwidth * bandwidth * acos(distance/(2.0 * bandwidth)) -
          distance / 4.0 * sqrt(4.0*bandwidth*bandwidth-distance*distance));
        break;
      default:
        Log::Fatal << "The spherical kernel does not support convolution\
          integrals above dimension two, yet..." << std::endl;
        return -1.0;
        break;
    }
  }
  double Normalizer(size_t dimension) const
  {
    return pow(bandwidth, (double) dimension) * pow(M_PI, dimension / 2.0) /
        boost::math::tgamma(dimension / 2.0 + 1.0);
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
  double Gradient(double t) {
    return t == bandwidth ? arma::datum::nan : 0.0;
  }

  //! Serialize the object.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(bandwidth, "bandwidth");
    ar & data::CreateNVP(bandwidthSquared, "bandwidthSquared");
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

} // namespace kernel
} // namespace mlpack

#endif
