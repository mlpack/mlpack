/**
 * @file hyperbolic_tangent_kernel.hpp
 * @author Ajinkya Kale <kaleajinkya@gmail.com>
 *
 * Implementation of the hyperbolic tangent kernel
 */
#ifndef __MLPACK_CORE_KERNELS_HYPERBOLIC_TANGENT_KERNEL_HPP
#define __MLPACK_CORE_KERNELS_HYPERBOLIC_TANGENT_KERNEL_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace kernel {

/**
 * Hyperbolic tangent kernel.  For any two vectors @f$ x @f$,
 * @f$ y @f$ and @f$ scale @f$ and @f$ offset @f$
 *
 * @f[
 * k(x, y) = \tanh(scale <x, y> + offset)
 * @f]
 *
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
   * @tparam VecType Type of vector (should be arma::vec or arma::spvec).
   * @param a First vector.
   * @param b Second vector.
   * @return K(a, b).
   */
  template<typename VecType>
  double Evaluate(const VecType& a, const VecType& b)
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

 private:
  double scale;
  double offset;
};

}; // namespace kernel
}; // namespace mlpack

#endif
