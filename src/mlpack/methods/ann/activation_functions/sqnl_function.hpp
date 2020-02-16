/**
 * @file sqnl_function.hpp
 * @author Prince Gupta
 *
 * Definition and implementation of the SQNL (Square Non-Linearity)
 * activation function.
 *
 * For more information, see the following paper.
 *
 * @code
 * @misc{
 *   author = {Adedamola Wuraola, Nitish Patel},
 *   title = {SQNL: A New Computationally Efficient Activation Function},
 *   year = {2018}
 * }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_SQNL_FUNCTION_HPP
#define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_SQNL_FUNCTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artifical Neural Network. */ {

/**
 * The SQNL(Square Non Linearity) activation function, defined by
 * 
 * @f{eqnarray*}{
 * f(x) &=& 1, if x > 2,
 *          x + x^{2}/4, if -2 <= x < 0,
 *          x - x^{2}/4, if 0 <= x <= 2,
 *          -1, if x < -2
 * f'(x) &=& 0, if x > 2 or x < -2,
 *           1 - x / 2, if 0 <= x <= 2,
 *           1 + x / 2, if -2 <= x < 0
 * @f}
 */
class SQNLFunction
{
 public:
  /**
   * Computes the SQNL(Square Non Linearity) function.
   * 
   * @param x Input data.
   * @return f(x).
   */
  static double Fn(const double x)
  {
      if (x > 2)
        return 1;
      else if (x >= 0 && x <= 2)
        return x - std::pow(x, 2) / 4;
      else if (x >= -2 && x < 0)
        return x + std::pow(x, 2) / 4;
      else
        return -1;
  }

  /**
   * Computes the SQNL(Square Non Linearity) function.
   * 
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template<typename InputVecType, typename OutputVecType>
  static void Fn(const InputVecType& x, OutputVecType& y)
  {
    y.set_size(arma::size(x));
    for (size_t i = 0; i < x.n_elem; i++)
    {
      y(i) = Fn(x(i));
    }
  }

  /**
   * Computes the inverse of SQNL function y = f(x).
   * Since y = 1 for all x >= 2 and y = -1 for all x <= -2 it returns values
   * 2 and -2 for y == 1 and y == -1 respectively.
   * Since y can only take values in range [-1,1] if input is some value out of bounds,
   * value returned is 0.
   * 
   * @param y Input Data.
   * @return f^{-1}(y).
   */
  static double Inv(const double y)
  {
    if (y >= 0 && y <= 1)
      return 2 * (1 - std::sqrt(1 - y));
    else if (y >= -1 && y < 0)
      return -2 * (1 - std::sqrt(1 + y));
    else
      return 0;
  }

  /**
   * Computes the inverse of SQNL function y = f(x).
   * Since y = 1 for all x > 2 and y = -1 for all x < -2 it returns value
   * 3 and -3 respectively for such input y.
   * Since y can only take values in range [-1,1] if input is some value out of bounds,
   * value returned is 0.
   * 
   * @param y Input Data.
   * @param x The resulting invers of the input y.
   */
  template<typename InputVecType, typename OutputVecType>
  static void Inv(const InputVecType& y, OutputVecType& x)
  {
    x.set_size(arma::size(y));
    for (size_t i = 0; i < x.n_elem; i++)
    {
      x(i) = Inv(y(i));
    }
  }

  /**
   * Computes the first derivate of SQNL function.
   * 
   * @param y Input activation.
   * @return f'(x) where f(x) = y.
   */
  static double Deriv(const double y)
  {
    return 1 - std::abs(Inv(y)) / 2;
  }

  /**
   * Computes the first derivate of SQNL function.
   * 
   * @param y Input activations.
   * @param x The resulting derivatives.
   */
  template<typename InputVecType, typename OutputVecType>
  static void Deriv(const InputVecType& y, OutputVecType& x)
  {
    x.set_size(arma::size(y));
    for (size_t i = 0; i < x.n_elem; i++)
    {
      x(i) = Deriv(y(i));
    }
  }
}; // class SQNL function

} // namespace ann
} // namespace mlpack

#endif
