/**
 * @file tanh_function.hpp
 * @author Marcus Edel
 *
 * Definition and implementation of the Tangens Hyperbolic function.
 *
 * This file is part of mlpack 2.0.0.
 *
 * mlpack is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * mlpack is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * mlpack.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_TANH_FUNCTION_HPP
#define __MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_TANH_FUNCTION_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

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
  static double fn(const double x)
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
  static void fn(const InputVecType& x, OutputVecType& y)
  {
    y = arma::tanh(x);
  }

  /**
   * Computes the first derivative of the tanh function.
   *
   * @param y Input data.
   * @return f'(x)
   */
  static double deriv(const double y)
  {
    return 1 - std::pow(y, 2);
  }

  /**
   * Computes the first derivatives of the tanh function.
   *
   * @param y Input data.
   * @param x The resulting derivatives.
   */
  template<typename InputVecType, typename OutputVecType>
  static void deriv(const InputVecType& y, OutputVecType& x)
  {
    x = 1 - arma::pow(y, 2);
  }

  /**
   * Computes the inverse of the tanh function.
   *
   * @param y Input data.
   * @return f^{-1}(x)
   */
  static double inv(const double y)
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
  static void inv(const InputVecType& y, OutputVecType& x)
  {
    x = arma::atanh(y);
  }
}; // class TanhFunction

} // namespace ann
} // namespace mlpack

#endif
