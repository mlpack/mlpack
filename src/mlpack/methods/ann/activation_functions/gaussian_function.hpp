/**
 * @file gaussian_function.hpp
 * @author Himanshu Pathak
 * @author Adam Kropp
 *
 * Definition and implementation of the gaussian function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_GAUSSIAN_FUNCTION_HPP
#define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_GAUSSIAN_FUNCTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * The gaussian function, defined by
 *
 * @f{eqnarray*}{
 * f(x) &=& e^{-1 * x^2} \\
 * f'(x) &=& 2 * -x * f(x) 
 * @f}
 */
class GaussianFunction
{
 public:
  /**
   * Computes the gaussian function.
   *
   * @param x Input data.
   * @return f(x).
   */
  template<typename eT>
  static double Fn(const eT x)
  {
    return std::exp(-1 * std::pow(x, 2));
  }

  /**
   * Computes the gaussian function.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template<typename InputVecType, typename OutputVecType>
  static void Fn(const InputVecType& x, OutputVecType& y)
  {
    y = exp(-1 * pow(x, 2));
  }

  /**
   * Computes the first derivative of the gaussian function.
   *
   * @param x Input activation.
   * @param y Result of Fn(x).
   * @return f'(x)
   */
  static double Deriv(const double x, const double y)
  {
    return -2 * x * y;
  }

  /**
   * Computes the first derivatives of the gaussian function.
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
    dy = -2 * x % y;
  }
}; // class GaussianFunction

} // namespace mlpack

#endif
