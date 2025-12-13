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
  template<typename ElemType>
  static ElemType Fn(const ElemType x)
  {
    return std::exp(-std::pow(x, ElemType(2)));
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
    y = exp(-square(x));
  }

  /**
   * Computes the first derivative of the gaussian function.
   *
   * @param x Input activation.
   * @param y Result of Fn(x).
   * @return f'(x)
   */
  template<typename ElemType>
  static ElemType Deriv(const ElemType x, const ElemType y)
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
