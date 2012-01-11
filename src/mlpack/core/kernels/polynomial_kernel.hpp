/**
 * @file polynomial_kernel.hpp
 * @author Ajinkya Kale <kaleajinkya@gmail.com>
 *
 * Implementation of the polynomial kernel (just the standard dot product).
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
