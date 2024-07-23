/**
 * @file methods/ann/activation_functions/rectifier_function.hpp
 * @author Marcus Edel
 *
 * Definition and implementation of the rectifier function as described by
 * V. Nair and G. E. Hinton.
 *
 * For more information, see the following paper.
 *
 * @code
 * @misc{NairHinton2010,
 *   author = {Vinod Nair, Geoffrey E. Hinton},
 *   title = {Rectified Linear Units Improve Restricted Boltzmann Machines},
 *   year = {2010}
 * }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_RECTIFIER_FUNCTION_HPP
#define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_RECTIFIER_FUNCTION_HPP

#include <mlpack/prereqs.hpp>
#include <algorithm>

namespace mlpack {

/**
 * The rectifier function, defined by
 *
 * @f{eqnarray*}{
 * f(x) &=& \max(0, x) \\
 * f'(x) &=& \left\{
 *   \begin{array}{lr}
 *     1 & : x > 0 \\
 *     0 & : x \le 0
 *   \end{array}
 * \right.
 * @f}
 */
class RectifierFunction
{
 public:
  /**
   * Computes the rectifier function.
   *
   * @param x Input data.
   * @return f(x).
   */
  static double Fn(const double x)
  {
    return std::max(0.0, x);
  }

  /**
   * Computes the rectifier function using a 2nd /3rd-order tensor as input.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template<typename MatType>
  static void Fn(const MatType& x, MatType& y)
  {
    y.set_size(size(x));
    y.zeros();
    y = max(y, x);
  }

  /**
   * Computes the first derivative of the rectifier function.
   *
   * @param x Input activation.
   * @param y Result of Fn(x).
   * @return f'(x)
   */
  static double Deriv(const double x, const double /* y */)
  {
    return (double)(x > 0);
  }

  /**
   * Computes the first derivatives of the rectifier function.
   *
   * @param x Input activation.
   * @param y Result of Fn(x).
   * @param dy The resulting derivatives.
   */
  template<typename InputType, typename OutputType, typename DerivType>
  static void Deriv(const InputType& x,
                    const OutputType& /* y */,
                    DerivType& dy)
  {
    dy = ConvTo<DerivType>::From(x > 0);
  }
}; // class RectifierFunction

} // namespace mlpack

#endif
