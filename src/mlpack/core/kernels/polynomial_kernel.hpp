/**
 * @file polynomial_kernel.hpp
 * @author Ajinkya Kale <kaleajinkya@gmail.com>
 *
 * Implementation of the polynomial kernel (just the standard dot product).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_KERNELS_POLYNOMIAL_KERNEL_HPP
#define MLPACK_CORE_KERNELS_POLYNOMIAL_KERNEL_HPP

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
   * @tparam VecTypeA Type of first vector (should be arma::vec or
   *      arma::sp_vec).
   * @tparam VecTypeB Type of second vector (arma::vec / arma::sp_vec).
   * @param a First vector.
   * @param b Second vector.
   * @return K(a, b).
   */
  template<typename VecTypeA, typename VecTypeB>
  double Evaluate(const VecTypeA& a, const VecTypeB& b) const
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

  //! Serialize the kernel.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(degree, "degree");
    ar & data::CreateNVP(offset, "offset");
  }

 private:
  //! The degree of the polynomial.
  double degree;
  //! The offset of the dot product of the arguments.
  double offset;
};

} // namespace kernel
} // namespace mlpack

#endif
