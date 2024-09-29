/**
 * @file methods/ann/activation_functions/tanh_function.hpp
 * @author Marcus Edel
 *
 * Definition and implementation of the Tangens Hyperbolic function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_TANH_FUNCTION_HPP
#define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_TANH_FUNCTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * The tanh function, defined by
 *
 * @f{eqnarray*}{
 * f(x) &=& \frac{e^x - e^{-x}}{e^x + e^{-x}} \\
 * f'(x) &=& 1 - \tanh^2(x) \\
 * f^{-1}(x) &=& \arctan(x)
 * @f}
 */
class TanhFunction
{
 public:
  /**
   * Computes the tanh function.
   *
   * @param x Input data.
   * @return f(x).
   */
  static double Fn(const double x)
  {
    return std::tanh(x);
  }

  /**
   * Computes the tanh function.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template<typename InputVecType, typename OutputVecType>
  static void Fn(const InputVecType& x, OutputVecType& y)
  {
    y = arma::tanh(x);
  }

  /**
   * Computes the first derivative of the tanh function.
   *
   * @param x Input activation.
   * @param y Result of Fn(x).
   * @return f'(x)
   */
  static double Deriv(const double /* x */, const double y)
  {
    return 1 - std::pow(y, 2);
  }

  /**
   * Computes the first derivatives of the tanh function.
   *
   * @param x Input activation.
   * @param y Result of Fn(x).
   * @param dy The resulting derivatives.
   */
  template<typename InputVecType, typename OutputVecType, typename DerivVecType>
  static void Deriv(const InputVecType& /* x */,
                    const OutputVecType& y,
                    DerivVecType& dy)
  {
    dy = 1 - pow(y, 2);
  }

  /**
   * Computes the inverse of the tanh function.
   *
   * @param y Input data.
   * @return f^{-1}(x)
   */
  static double Inv(const double y)
  {
    return std::atanh(y);
  }

  /**
   * Computes the inverse of the tanh function.
   *
   * @param y Input data.
   * @param x The resulting inverse of the input data.
   */
  template<typename InputVecType, typename OutputVecType>
  static void Inv(const InputVecType& y, OutputVecType& x)
  {
    x = arma::atanh(y);
  }
}; // class TanhFunction

} // namespace mlpack

#endif
