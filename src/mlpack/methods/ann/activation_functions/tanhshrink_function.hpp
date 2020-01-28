/**
 * @file tanhshrink_function.hpp
 * @author Mrityunjay Tripathi
 *
 * Definition and implementation of the Tanhshrink function, f(x) = x − tanh(x).
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
namespace ann /** Artificial Neural Network. */ {

/**
 * The tanhshrink function, defined by
 *
 * @f{eqnarray*}{
 * f(x) = x - tanh(x) \\
 * f'(x) = tanh^2(x) \\
 * f^{-1}(x) = f(x) + sqrt(f'(x))\\
 * @f}
 */
class TanhshrinkFunction
{
 public:
  /**
   * Computes the tanhshrink function.
   *
   * @param x Input data.
   * @return f(x).
   */
  static double Fn(const double x)
  {
    return x - std::tanh(x);
  }

  /**
   * Computes the tanhshrink function.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template<typename InputVecType, typename OutputVecType>
  static void Fn(const InputVecType& x, OutputVecType& y)
  {
    y = x - arma::tanh(x);
  }

  /**
   * Computes the first derivative of the tanhshrink function.
   *
   * @param y Input data.
   * @return f'(x)
   */
  static double Deriv(const double y)
  {
    return std::pow(y, 2);
  }

  /**
   * Computes the first derivatives of the tanhshrink function.
   *
   * @param y Input data.
   * @param x The resulting derivatives.
   */
  template<typename InputVecType, typename OutputVecType>
  static void Deriv(const InputVecType& y, OutputVecType& x)
  {
    x = arma::pow(y, 2);
  }

  /**
   * Computes the inverse of the tanhshrink function.
   *
   * @param y Input data.
   * @return f^{-1}(x)
   */
  static double Inv(const double y)
  {
    return y + std::sqrt(Deriv(y)) ;
  }

  /**
   * Computes the inverse of the tanhshrink function.
   *
   * @param y Input data.
   * @param x The resulting inverse of the input data.
   */
  template<typename InputVecType, typename OutputVecType>
  static void Inv(const InputVecType& y, OutputVecType& x)
  {
    x = y + armd::sqrt(Deriv(y));
  }
}; // class TanhshrinkFunction

} // namespace ann
} // namespace mlpack

#endif
