/**
 * @file methods/ann/activation_functions/mish_function.hpp
 * @author Kartik Dutt
 *
 * Definition and implementation of the Mish function as described by
 * Diganta Misra.
 *
 * For more information, see the following paper.
 *
 * @code
 * @misc{
 *   author = {Diganta Misra},
 *   title = {Mish: Self Regularized Non-Monotonic Neural Activation Function},
 *   year = {2019}
 * }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_MISH_FUNCTION_HPP
#define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_MISH_FUNCTION_HPP

#include <mlpack/prereqs.hpp>
#include <algorithm>

namespace mlpack {

/**
 * The Mish function, defined by
 *
 * @f{eqnarray*}{
 * f(x) = x * tanh(ln(1+e^x))
 * f'(x) = tanh(ln(1+e^x)) + x * ((1 - tanh^2(ln(1+e^x))) * frac{1}{1 + e^{-x}})
 * @f}
 */
class MishFunction
{
 public:
  /**
   * Computes the Mish function.
   *
   * @param x Input data.
   * @return f(x).
   */
  static double Fn(const double x)
  {
    return x * (std::exp(2 * x) + 2 * std::exp(x)) /
           (2 + 2 * std::exp(x) + std::exp(2 * x));
  }

  /**
   * Computes the Mish function.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template <typename InputVecType, typename OutputVecType>
  static void Fn(const InputVecType &x, OutputVecType &y)
  {
    y = x % (arma::exp(2 * x) + 2 * arma::exp(x)) /
        (2 + 2 * arma::exp(x) + arma::exp(2 * x));
  }

  /**
   * Computes the first derivative of the Mish function.
   *
   * @param y Input data.
   * @return f'(x)
   */
  static double Deriv(const double y)
  {
    return std::exp(y) * (4 * (y + 1) + std::exp(y) * (4 * y + 6) +
           4 * std::exp(2 * y) + std::exp(3 * y)) /
           std::pow(std::exp(2 * y) + 2 * std::exp(y) + 2, 2);
  }

  /**
   * Computes the first derivatives of the Mish function.
   *
   * @param y Input data.
   * @param x The resulting derivatives.
   */
  template <typename InputVecType, typename OutputVecType>
  static void Deriv(const InputVecType &y, OutputVecType &x)
  {
    x = arma::exp(y) % (4 * (y + 1) + arma::exp(y) % (4 * y + 6) +
        4 * arma::exp(2 * y) + arma::exp(3 * y)) /
        arma::pow(arma::exp(2 * y) + 2 * arma::exp(y) + 2, 2);
  }
}; // class MishFunction

} // namespace mlpack

#endif
