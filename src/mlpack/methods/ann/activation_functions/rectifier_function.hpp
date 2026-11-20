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
  template<typename ElemType>
  static ElemType Fn(const ElemType x)
  {
    return std::max(ElemType(0), x);
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
    y = clamp(x, 0, std::numeric_limits<typename MatType::elem_type>::max());
  }

  /**
   * Computes the first derivative of the rectifier function.
   *
   * @param x Input activation.
   * @param y Result of Fn(x).
   * @return f'(x)
   */
  template<typename ElemType>
  static ElemType Deriv(const ElemType x, const ElemType /* y */)
  {
    return ElemType(x > 0);
  }

  /**
   * Computes the first derivatives of the rectifier function.
   *
   * @param x Input activation.
   * @param y Result of Fn(x).
   * @param dy The resulting derivatives.
   */
  template<typename InputType, typename OutputType, typename DerivType>
  static void Deriv(const InputType& /* x */,
                    const OutputType& y,
                    DerivType& dy)
  {
    dy = sign(y);
  }
}; // class RectifierFunction

} // namespace mlpack

#endif
