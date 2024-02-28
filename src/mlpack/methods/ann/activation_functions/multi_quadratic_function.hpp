/**
 * @file methods/ann/activation_functions/multi_quadratic_function.hpp
 * @author Himanshu Pathak
 * @author Adam Kropp
 *
 * Definition and implementation of multi quadratic function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_MULTIQUAD_FUNCTION_HPP
#define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_MULTIQUAD_FUNCTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * The Multi Quadratic function, defined by
 *
 * @f{eqnarray*}{
 * f(x) = \sqrt(1 + x^2) \\
 * f'(x) = x / f(x) \\
 * @f}
 */
class MultiQuadFunction
{
 public:
  /**
   * Computes the Multi Quadratic function.
   *
   * @param x Input data.
   * @return f(x).
   */
  static double Fn(const double x)
  {
    return std::pow(1 + x * x, 0.5);
  }

  /**
   * Computes the Multi Quadratic function.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template<typename InputVecType, typename OutputVecType>
  static void Fn(const InputVecType& x, OutputVecType& y)
  {
    y = pow((1 + pow(x, 2)), 0.5);
  }

  /**
   * Computes the first derivative of the Multi Quadratic function.
   *
   * Can be computed as x / sqrt(1 + x^2), but since f(x)=sqrt(1+x^2),
   * we can also use x / f(x)
   *
   * @param x Input activation.
   * @param y Result of Fn(x).
   * @return f'(x)
   */
  static double Deriv(const double x, const double y)
  {
    return x / y;
  }

  /**
   * Computes the first derivatives of the Multi Quadratic function.
   *
   * @param x Input activation.
   * @param y Result of Fn(x).
   * @param dy The resulting derivatives.
   */
  template<typename InputVecType, typename OutputVecType, typename DerivVecType>
  static void Deriv(const InputVecType& x,
                    const OutputVecType& y,
                    DerivVecType &dy)
  {
    dy = x / y;
  }
}; // class MultiquadFunction

} // namespace mlpack

#endif
