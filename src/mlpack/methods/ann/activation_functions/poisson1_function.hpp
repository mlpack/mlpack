/**
 * @file methods/ann/activation_functions/spline_function.hpp
 * @author Himanshu Pathak
 *
 * Definition and implementation of Poisson one function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_POISSON1_FUNCTION_HPP
#define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_POISSON1_FUNCTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * The Poisson one function, defined by
 *
 * @f{eqnarray*}{
 * f(x) = (x - 1) * e^-x \\
 * f'(x) = e^-x + (1 - x) * e^-x \\
 * @f}
 */
class Poisson1Function
{
 public:
  /**
   * Computes the Poisson one function.
   *
   * @param x Input data.
   * @return f(x).
   */
  static double Fn(const double x)
  {
    return (x - 1) * std::exp(-x);
  }

  /**
   * Computes the Poisson one function.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template<typename InputVecType, typename OutputVecType>
  static void Fn(const InputVecType& x, OutputVecType& y)
  {
    y = (x - 1) % arma::exp(-x);
  }

  /**
   * Computes the first derivative of the Poisson one function.
   *
   * @param y Input data.
   * @return f'(x)
   */
  static double Deriv(const double y)
  {
    return  std::exp(-y) + (1 - y) * std::exp(-y);
  }

  /**
   * Computes the first derivatives of the Poisson one function.
   *
   * @param y Input data.
   * @param x The resulting derivatives.
   */
  template<typename InputVecType, typename OutputVecType>
  static void Deriv(const InputVecType& x, OutputVecType& y)
  {
    y = arma::exp(-x) + (1 - x) % arma::exp(-x);
  }
}; // class Poisson1Function

} // namespace mlpack

#endif
