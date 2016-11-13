/**
 * @file laplacian_kernel.hpp
 * @author Ajinkya Kale <kaleajinkya@gmail.com>
 *
 * Implementation of the Laplacian kernel (LaplacianKernel).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_KERNELS_LAPLACIAN_KERNEL_HPP
#define MLPACK_CORE_KERNELS_LAPLACIAN_KERNEL_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace kernel {

/**
 * The standard Laplacian kernel.  Given two vectors @f$ x @f$, @f$ y @f$, and a
 * bandwidth @f$ \mu @f$ (set in the constructor),
 *
 * @f[
 * K(x, y) = \exp(-\frac{|| x - y ||}{\mu}).
 * @f]
 *
 * The implementation is all in the header file because it is so simple.
 */
class LaplacianKernel
{
 public:
  /**
   * Default constructor; sets bandwidth to 1.0.
   */
  LaplacianKernel() : bandwidth(1.0)
  { }

  /**
   * Construct the Laplacian kernel with a custom bandwidth.
   *
   * @param bandwidth The bandwidth of the kernel (@f$\mu@f$).
   */
  LaplacianKernel(double bandwidth) :
      bandwidth(bandwidth)
  { }

  /**
   * Evaluation of the Laplacian kernel.  This could be generalized to use any
   * distance metric, not the Euclidean distance, but for now, the Euclidean
   * distance is used.
   *
   * @tparam VecTypeA Type of first vector (likely arma::vec or arma::sp_vec).
   * @tparam VecTypeB Type of second vector (arma::vec / arma::sp_vec).
   * @param a First vector.
   * @param b Second vector.
   * @return K(a, b) using the bandwidth (@f$\mu@f$) specified in the
   *      constructor.
   */
  template<typename VecTypeA, typename VecTypeB>
  double Evaluate(const VecTypeA& a, const VecTypeB& b) const
  {
    // The precalculation of gamma saves us a little computation time.
    return exp(-metric::EuclideanDistance::Evaluate(a, b) / bandwidth);
  }

  /**
   * Evaluation of the Laplacian kernel given the distance between two points.
   *
   * @param t The distance between the two points the kernel should be evaluated
   *     on.
   * @return K(t) using the bandwidth (@f$\mu@f$) specified in the
   *     constructor.
   */
  double Evaluate(const double t) const
  {
    // The precalculation of gamma saves us a little computation time.
    return exp(-t / bandwidth);
  }

  /**
   * Evaluation of the gradient of the Laplacian kernel
   * given the distance between two points.
   *
   * @param t The distance between the two points the kernel should be evaluated
   *     on.
   * @return K(t) using the bandwidth (@f$\mu@f$) specified in the
   *     constructor.
   */
  double Gradient(const double t) const  {
    return exp(-t / bandwidth) / -bandwidth;
  }

  //! Get the bandwidth.
  double Bandwidth() const { return bandwidth; }
  //! Modify the bandwidth.
  double& Bandwidth() { return bandwidth; }

  //! Serialize the kernel.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(bandwidth, "bandwidth");
  }

 private:
  //! Kernel bandwidth.
  double bandwidth;
};

//! Kernel traits of the Laplacian kernel.
template<>
class KernelTraits<LaplacianKernel>
{
 public:
  //! The Laplacian kernel is normalized: K(x, x) = 1 for all x.
  static const bool IsNormalized = true;
  //! The Laplacian kernel doesn't include a squared distance.
  static const bool UsesSquaredDistance = false;
};

} // namespace kernel
} // namespace mlpack

#endif
