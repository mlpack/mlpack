/**
 * @file polynomial_kernel.hpp
 * @author Ajinkya Kale <kaleajinkya@gmail.com>
 *
 * Implementation of the polynomial kernel (just the standard dot product).
 *
 * This file is part of mlpack 1.0.12.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef __MLPACK_CORE_KERNELS_POLYNOMIAL_KERNEL_HPP
#define __MLPACK_CORE_KERNELS_POLYNOMIAL_KERNEL_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace kernel {

/**
 * The simple polynomial kernel.  For any two vectors @f$ x @f$, @f$ y @f$,
 * @f$ degree @f$ and @f$ offset @f$,
 *
 * @f[
 * K(x, y) = (x^T * y + offset) ^ {degree}.
 * @f]
 */
class PolynomialKernel
{
 public:
  /**
   * Construct the Polynomial Kernel with the given offset and degree.  If the
   * arguments are omitted, the default degree is 2 and the default offset is 0.
   *
   * @param offset Offset of the dot product of the arguments.
   * @param degree Degree of the polynomial.
   */
  PolynomialKernel(const double degree = 2.0, const double offset = 0.0) :
      degree(degree),
      offset(offset)
  { }

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
  double Evaluate(const VecType& a, const VecType& b) const
  {
    return pow((arma::dot(a, b) + offset), degree);
  }

  //! Get the degree of the polynomial.
  const double& Degree() const { return degree; }
  //! Modify the degree of the polynomial.
  double& Degree() { return degree; }

  //! Get the offset of the dot product of the arguments.
  const double& Offset() const { return offset; }
  //! Modify the offset of the dot product of the arguments.
  double& Offset() { return offset; }

  //! Return a string representation of the kernel.
  std::string ToString() const
  {
    std::ostringstream convert;
    convert << "PolynomialKernel [" << this << "]" << std::endl;
    convert << "  Degree: " << degree << std::endl;
    convert << "  Offset: " << offset << std::endl;
    return convert.str();
  }

 private:
  //! The degree of the polynomial.
  double degree;
  //! The offset of the dot product of the arguments.
  double offset;
};

}; // namespace kernel
}; // namespace mlpack

#endif
