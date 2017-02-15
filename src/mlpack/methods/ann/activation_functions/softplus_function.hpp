/**
 * @file softplus_function.hpp
 * @author Vivek Pal
 *
 * Definition and implementation of the softplus function as described by
 * Charles Dugas, Yoshua Bengio, Franc¸ois Belisle, Claude Nadeau & Rene Garcia.
 *
 * For more information, please see the following paper:
 *
 * @code
 * @INPROCEEDINGS{secondorder:2001:nips,
 *   author = {Dugas, Charles and Bengio, Yoshua and Belisle, Francois and
 *   Nadeau, Claude and Garcia, Rene},
 *   title = {Incorporating Second-Order Functional Knowledge for Better Option
 *   Pricing},
 *   booktitle = {Advances in Neural Information Processing Systems},
 *   year = {2001}
 * }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_SOFTPLUS_FUNCTION_HPP
#define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_SOFTPLUS_FUNCTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The softplus function, defined by
 *
 * @f{eqnarray*}{
 * f(x) &=& \ln(1 + e^{x}) \\
 * f'(x) &=& \frac{1}{1 + e^{-x}} \\
 * f^{-1}(y) &=& \ln(e^{y} - 1)
 * @f}
 */
class SoftplusFunction
{
  public:
  
  /**
   * Computes the softplus function.
   *
   * @param x Input data.
   * @return f(x).
   */
  static double fn(const double x)
  {
    if (x < DBL_MAX)
      return x > -DBL_MAX ? std::log(1 + std::exp(x)) : 0;
    return 1.0;
  }

  /**
   * Computes the softplus function.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template<typename InputVecType, typename OutputVecType>
  static void fn(const InputVecType& x, OutputVecType& y)
  {
    y = x;

    for (size_t i = 0; i < x.n_elem; i++)
      y(i) = fn(x(i));
  }

  /**
   * Computes the first derivative of the softplus function.
   *
   * @param y Input data.
   * @return f'(x)
   */
  static double deriv(const double y)
  {
    return 1.0 / (1 + std::exp(-y));
  }

  /**
   * Computes the first derivatives of the softplus function.
   *
   * @param y Input activations.
   * @param x The resulting derivatives.
   */
  template<typename InputVecType, typename OutputVecType>
  static void deriv(const InputVecType& y, OutputVecType& x)
  {
    x = 1.0 / (1 + arma::exp(-y));
  }

  /**
   * Computes the inverse of the softplus function.
   *
   * @param y Input data.
   * @return f^{-1}(y)
   */
  static double inv(const double y)
  {
    return y > 0 ? arma::trunc_log(arma::trunc_exp(y) - 1) : 0;
  }

  /**
   * Computes the inverse of the softplus function.
   *
   * @param y Input data.
   * @param x The resulting inverse of the input data.
   */
  template<typename InputVecType, typename OutputVecType>
  static void inv(const InputVecType& y, OutputVecType& x)
  {
    x = y;

    for (size_t i = 0; i < y.n_elem; i++)
      x(i) = inv(y(i));
  }
}; // class SoftplusFunction

} // namespace ann
} // namespace mlpack

#endif
