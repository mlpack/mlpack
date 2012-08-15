/**
 * @file polynomial_kernel.hpp
 * @author Ajinkya Kale <kaleajinkya@gmail.com>
 *
 * Implementation of the polynomial kernel (just the standard dot product).
 * This file is part of MLPACK 1.0.2.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_CORE_KERNELS_POLYNOMIAL_KERNEL_HPP
#define __MLPACK_CORE_KERNELS_POLYNOMIAL_KERNEL_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace kernel {

/**
 * The simple polynomial kernel.  For any two vectors @f$ x @f$,
 * @f$ y @f$, @f$ degree @f$ and @f$ offset @f$
 *
 * @f[
 * K(x, y) = (x^T * y + offset) ^ {degree}
 * @f]
 *
 */
class PolynomialKernel
{
 public:
  /**
   * Default constructor; sets offset to 0.0 and degree to 1.0
   */
  PolynomialKernel() :
    offset(0.0),
    degree(1.0)
  { }

  /* Construct the Polynomial Kernel with custom
   * offset and degree
   *
   * @param offset offset to the polynomial
   * @param degree degree of the polynomial
   */
  PolynomialKernel(double offset, double degree) :
    offset(offset),
    degree(degree)
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
  double Evaluate(const VecType& a, const VecType& b)
  {
    return pow((arma::dot(a, b) + offset), degree);
  }

  //! Get the offset
  const double& Offset() const { return offset; }
  //! Get the degree of the polynomial
  const double& Degree() const { return degree; }

 private:
  double offset;
  double degree;
};

}; // namespace kernel
}; // namespace mlpack

#endif
