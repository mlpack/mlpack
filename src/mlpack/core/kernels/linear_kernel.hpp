/**
 * @file linear_kernel.hpp
 * @author Wei Guan
 * @author James Cline
 * @author Ryan Curtin
 *
 * Implementation of the linear kernel (just the standard dot product).
 *
 * This file is part of mlpack 1.0.12.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef __MLPACK_CORE_KERNELS_LINEAR_KERNEL_HPP
#define __MLPACK_CORE_KERNELS_LINEAR_KERNEL_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace kernel {

/**
 * The simple linear kernel (dot product).  For any two vectors @f$ x @f$ and
 * @f$ y @f$,
 *
 * @f[
 * K(x, y) = x^T y
 * @f]
 *
 * This kernel has no parameters and therefore the evaluation can be static.
 */
class LinearKernel
{
 public:
  /**
   * This constructor does nothing; the linear kernel has no parameters to
   * store.
   */
  LinearKernel() { }

  /**
   * Simple evaluation of the dot product.  This evaluation uses Armadillo's
   * dot() function.
   *
   * @tparam VecType Type of vector (should be arma::vec or arma::spvec).
   * @param a First vector.
   * @param b Second vector.
   * @return K(a, b).
   */
  template<typename VecType>
  static double Evaluate(const VecType& a, const VecType& b)
  {
    return arma::dot(a, b);
  }

  //! Return a string representation of the kernel.
  std::string ToString() const
  {
    std::ostringstream convert;
    convert << "LinearKernel [" << this << "]" << std::endl;
    return convert.str();
  }
};

}; // namespace kernel
}; // namespace mlpack

#endif
