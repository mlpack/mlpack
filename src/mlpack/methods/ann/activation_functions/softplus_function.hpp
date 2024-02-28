/**
 * @file methods/ann/activation_functions/softplus_function.hpp
 * @author Vivek Pal
 *
 * Definition and implementation of the softplus function as described by
 * Charles Dugas, Yoshua Bengio, Franc¸ois Belisle, Claude Nadeau & Rene Garcia.
 *
 * For more information, please see the following paper:
 *
 * @code
 * @inproceedings{Dugas2001,
 *   author    = {Dugas, Charles and Bengio, Yoshua and Belisle, Francois and
 *                Nadeau, Claude and Garcia, Rene},
 *   title     = {Incorporating Second-Order Functional Knowledge for Better
 *                Option Pricing},
 *   booktitle = {Advances in Neural Information Processing Systems},
 *   year      = {2001}
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
  static double Fn(const double x)
  {
    const double val = std::log(1 + std::exp(x));
    if (std::isfinite(val))
      return val;
    return x;
  }

  /**
   * Computes the softplus function.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template<typename InputType, typename OutputType>
  static void Fn(const InputType& x, OutputType& y)
  {
    y.set_size(arma::size(x));

    for (size_t i = 0; i < x.n_elem; ++i)
      y(i) = Fn(x(i));
  }

  /**
   * Computes the first derivative of the softplus function.
   *
   * @param x Input activation.
   * @param y Result of Fn(x).
   * @return f'(x)
   */
  static double Deriv(const double x, const double /* y */)
  {
    return 1.0 / (1 + std::exp(-x));
  }

  /**
   * Computes the first derivatives of the softplus function.
   *
   * @param x Input activation.
   * @param y Result of Fn(x).
   * @param dy The resulting derivatives.
   */
  template<typename InputType, typename OutputType, typename DerivType>
  static void Deriv(const InputType& x,
                    const OutputType& /* y */,
                    DerivType& dy)
  {
    dy = 1.0 / (1 + exp(-x));
  }

  /**
   * Computes the inverse of the softplus function.
   *
   * @param y Input data.
   * @return f^{-1}(y)
   */
  static double Inv(const double y)
  {
    const double val = std::log(std::exp(y) - 1);
    if (std::isfinite(val))
      return val;
    return y;
  }

  /**
   * Computes the inverse of the softplus function.
   *
   * @param y Input data.
   * @param x The resulting inverse of the input data.
   */
  template<typename InputType, typename OutputType>
  static void Inv(const InputType& y, OutputType& x)
  {
    x.set_size(arma::size(y));

    for (size_t i = 0; i < y.n_elem; ++i)
      x(i) = Inv(y(i));
  }
}; // class SoftplusFunction

} // namespace mlpack

#endif
