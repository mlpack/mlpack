/**
 * @file methods/ann/activation_functions/quadratic_function.hpp
 * @author Himanshu Pathak
 *
 * Definition and implementation of quadratic function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_QUAD_FUNCTION_HPP
#define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_QUAD_FUNCTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * The Quadratic function, defined by
 *
 * @f{eqnarray*}{
 * f(x) = x^2 \\
 * f'(x) = 2 * x \\
 * @f}
 */
class QuadraticFunction
{
 public:
  /**
   * Computes the Quadratic function.
   *
   * @param x Input data.
   * @return f(x).
   */
  static double Fn(const double x)
  {
    return std::pow(x, 2);
  }

  /**
   * Computes the Quadratic function.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template<typename InputVecType, typename OutputVecType>
  static void Fn(const InputVecType& x, OutputVecType& y)
  {
    y = pow(x, 2);
  }

  /**
   * Computes the first derivative of the Quadratic function.
   *
   * @param x Input activation.
   * @param y Result of Fn(x).
   * @return f'(x)
   */
  static double Deriv(const double x, const double /* y */)
  {
    return 2 * x;
  }

  /**
   * Computes the first derivatives of the Quadratic function.
   *
   * @param x Input activation.
   * @param y Result of Fn(x).
   * @param dy The resulting derivatives.
   */
  template<typename InputVecType, typename OutputVecType, typename DerivVecType>
  static void Deriv(const InputVecType& x,
                    const OutputVecType& /* y */,
                    DerivVecType& dy)
  {
    dy = 2 * x;
  }
}; // class QUADFunction

} // namespace mlpack

#endif
