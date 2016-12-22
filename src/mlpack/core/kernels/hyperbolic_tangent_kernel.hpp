/**
 * @file hyperbolic_tangent_kernel.hpp
 * @author Ajinkya Kale <kaleajinkya@gmail.com>
 *
 * Implementation of the hyperbolic tangent kernel.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_KERNELS_HYPERBOLIC_TANGENT_KERNEL_HPP
#define MLPACK_CORE_KERNELS_HYPERBOLIC_TANGENT_KERNEL_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace kernel {

/**
 * Hyperbolic tangent kernel.  For any two vectors @f$ x @f$, @f$ y @f$ and a
 * given scale @f$ s @f$ and offset @f$ t @f$
 *
 * @f[
 * K(x, y) = \tanh(s <x, y> + t)
 * @f]
 */
class HyperbolicTangentKernel
{
 public:
  /**
   * This constructor sets the default scale to 1.0 and offset to 0.0.
   */
  HyperbolicTangentKernel() : scale(1.0), offset(0.0)
  { }

  /**
   * Construct the hyperbolic tangent kernel with custom scale factor and
   * offset.
   *
   * @param scale Scaling factor for <x, y>.
   * @param offset Kernel offset.
   */
  HyperbolicTangentKernel(double scale, double offset) :
      scale(scale), offset(offset)
  { }

  /**
   * Evaluate the hyperbolic tangent kernel.  This evaluation uses Armadillo's
   * dot() function.
   *
   * @tparam VecTypeA Type of first vector (should be arma::vec or
   *      arma::sp_vec).
   * @tparam VecTypeB Type of second vector (arma::vec / arma::sp_vec).
   * @param a First vector.
   * @param b Second vector.
   * @return K(a, b).
   */
  template<typename VecTypeA, typename VecTypeB>
  double Evaluate(const VecTypeA& a, const VecTypeB& b)
  {
    return tanh(scale * arma::dot(a, b) + offset);
  }

  //! Get scale factor.
  double Scale() const { return scale; }
  //! Modify scale factor.
  double& Scale() { return scale; }

  //! Get offset for the kernel.
  double Offset() const { return offset; }
  //! Modify offset for the kernel.
  double& Offset() { return offset; }

  //! Serialize the kernel.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(scale, "scale");
    ar & data::CreateNVP(offset, "offset");
  }

 private:
  double scale;
  double offset;
};

} // namespace kernel
} // namespace mlpack

#endif
