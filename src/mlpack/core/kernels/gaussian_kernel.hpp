/**
 * @file core/kernels/gaussian_kernel.hpp
 * @author Wei Guan
 * @author James Cline
 * @author Ryan Curtin
 *
 * Implementation of the Gaussian kernel (GaussianKernel).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_KERNELS_GAUSSIAN_KERNEL_HPP
#define MLPACK_CORE_KERNELS_GAUSSIAN_KERNEL_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/distances/lmetric.hpp>
#include <mlpack/core/kernels/kernel_traits.hpp>

namespace mlpack {

/**
 * The standard Gaussian kernel.  Given two vectors @f$ x @f$, @f$ y @f$, and a
 * bandwidth @f$ \mu @f$ (set in the constructor),
 *
 * @f[
 * K(x, y) = \exp(-\frac{|| x - y ||^2}{2 \mu^2}).
 * @f]
 *
 * The implementation is all in the header file because it is so simple.
 */
class GaussianKernel
{
 public:
  /**
   * Default constructor; sets bandwidth to 1.0.
   */
  GaussianKernel() : bandwidth(1.0), gamma(-0.5)
  { }

  /**
   * Construct the Gaussian kernel with a custom bandwidth.
   *
   * @param bandwidth The bandwidth of the kernel (@f$\mu@f$).
   */
  GaussianKernel(const double bandwidth) :
      bandwidth(bandwidth),
      gamma(-0.5 * std::pow(bandwidth, -2.0))
  { }

  /**
   * Evaluation of the Gaussian kernel.  This could be generalized to use any
   * distance metric, not the Euclidean distance, but for now, the Euclidean
   * distance is used.
   *
   * @tparam VecType Type of vector (likely arma::vec or arma::spvec).
   * @param a First vector.
   * @param b Second vector.
   * @return K(a, b) using the bandwidth (@f$\mu@f$) specified in the
   *   constructor.
   */
  template<typename VecTypeA, typename VecTypeB>
  double Evaluate(const VecTypeA& a, const VecTypeB& b) const
  {
    // The precalculation of gamma saves us a little computation time.
    return std::exp(gamma * SquaredEuclideanDistance::Evaluate(a, b));
  }

  /**
   * Evaluation of the Gaussian kernel given the distance between two points.
   *
   * @param t The distance between the two points the kernel is evaluated on.
   * @return K(t) using the bandwidth (@f$\mu@f$) specified in the
   *     constructor.
   */
  double Evaluate(const double t) const
  {
    // The precalculation of gamma saves us a little computation time.
    return std::exp(gamma * std::pow(t, 2.0));
  }

  /**
   * Evaluation of the gradient of Gaussian kernel
   * given the distance between two points.
   *
   * @param t The distance between the two points the kernel is evaluated on.
   * @return K(t) using the bandwidth (@f$\mu@f$) specified in the
   *     constructor.
   */
  double Gradient(const double t) const {
    return 2 * t * gamma * std::exp(gamma * std::pow(t, 2.0));
  }

  /**
   * Obtain the normalization constant of the Gaussian kernel.
   *
   * @param dimension
   * @return the normalization constant
   */
  double Normalizer(const size_t dimension)
  {
    return std::pow(std::sqrt(2.0 * M_PI) * bandwidth, (double) dimension);
  }

  //! Get the bandwidth.
  double Bandwidth() const { return bandwidth; }

  //! Modify the bandwidth.  This takes an argument because we must update the
  //! precalculated constant (gamma).
  void Bandwidth(const double bandwidth)
  {
    this->bandwidth = bandwidth;
    this->gamma = -0.5 * std::pow(bandwidth, -2.0);
  }

  //! Get the precalculated constant.
  double Gamma() const { return gamma; }

  //! Serialize the kernel.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(bandwidth));
    ar(CEREAL_NVP(gamma));
  }

 private:
  //! Kernel bandwidth.
  double bandwidth;

  //! Precalculated constant depending on the bandwidth;
  //! @f$ \gamma = -\frac{1}{2 \mu^2} @f$.
  double gamma;
};

//! Kernel traits for the Gaussian kernel.
template<>
class KernelTraits<GaussianKernel>
{
 public:
  //! The Gaussian kernel is normalized: K(x, x) = 1 for all x.
  static const bool IsNormalized = true;
  //! The Gaussian kernel includes a squared distance.
  static const bool UsesSquaredDistance = true;
};

} // namespace mlpack

#endif
