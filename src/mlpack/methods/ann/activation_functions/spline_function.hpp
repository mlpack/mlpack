/**
 * @file methods/ann/activation_functions/spline_function.hpp
 * @author Himanshu Pathak
 * @author Adam Kropp
 *
 * Definition and implementation of Spline function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_SPLINE_FUNCTION_HPP
#define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_SPLINE_FUNCTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * The Spline function, defined by
 *
 * @f{eqnarray*}{
 * f(x) = x^2 * log(1 + x) \\
 * f'(x) = 2 * x * log(1 + x) + x^2 / (1 + x)\\
 * @f}
 */
class SplineFunction
{
 public:
  /**
   * Computes the Spline function.
   *
   * @param x Input data.
   * @return f(x).
   */
  static double Fn(const double x)
  {
    return std::pow(x, 2) * std::log(1 + x);
  }

  /**
   * Computes the Spline function.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template<typename InputVecType, typename OutputVecType>
  static void Fn(const InputVecType& x, OutputVecType& y)
  {
    y = pow(x, 2) % log(1 + x);
  }

  /**
   * Computes the first derivative of the Spline function.
   *
   * @param x Input activation.
   * @param y Result of Fn(x).
   * @return f'(x)
   */
  static double Deriv(const double x, const double y)
  {
    return  x != 0 ? 2 * y / x + std::pow(x, 2) / (1 + x) : 0;
  }

  /**
   * Computes the first derivatives of the Spline function.
   *
   * @param x Input activation.
   * @param y Result of Fn(x).
   * @param dy The resulting derivatives.
   */
  template<typename InputVecType, typename OutputVecType, typename DerivVecType>
  static void Deriv(const InputVecType& x,
                    const OutputVecType& y,
                    DerivVecType& dy)
  {
    dy = 2 * y / x + pow(x, 2) / (1 + x);
    // the expression above is indeterminate at 0, even though
    // the expression solely in terms of x is defined (= 0)
    dy(arma::find(x == 0)).zeros();
  }
}; // class SplineFunction

} // namespace mlpack

#endif
