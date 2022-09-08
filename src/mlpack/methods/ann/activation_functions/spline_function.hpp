/**
 * @file methods/ann/activation_functions/spline_function.hpp
 * @author Himanshu Pathak
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
    y = arma::pow(x, 2) % arma::log(1 + x);
  }

  /**
   * Computes the first derivative of the Spline function.
   *
   * @param y Input data.
   * @return f'(x)
   */
  static double Deriv(const double y)
  {
    return  2 * y * std::log(1 + y) + std::pow(y, 2) / (1 + y);
  }

  /**
   * Computes the first derivatives of the Spline function.
   *
   * @param y Input data.
   * @param x The resulting derivatives.
   */
  template<typename InputVecType, typename OutputVecType>
  static void Deriv(const InputVecType& x, OutputVecType& y)
  {
    y = 2 * x % arma::log(1 + x) + arma::pow(x, 2) / (1 + x);
  }
}; // class SplineFunction

} // namespace mlpack

#endif
