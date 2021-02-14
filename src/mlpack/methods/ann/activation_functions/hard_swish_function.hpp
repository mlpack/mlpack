/**
 * @file methods/ann/activation_functions/hard_swish_function.hpp
 * @author Anush Kini
 *
 * Definition and implementation of the Hard Swish function as described by
 * Howard A, Sandler M, Chu G, Chen LC, Chen B, Tan M, Wang W, Zhu Y, Pang R,
 * Vasudevan V and Le QV.
 * For more information, see the following paper.
 *
 * @code
 * @misc{
 *   author = {Howard A, Sandler M, Chu G, Chen LC, Chen B, Tan M, Wang W,
 *            Zhu Y, Pang R, Vasudevan V and Le QV},
 *   title = {Searching for MobileNetV3},
 *   year = {2019}
 * }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_HARD_SWISH_FUNCTION_HPP
#define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_HARD_SWISH_FUNCTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {
/**
 * The Hard Swish function, defined by
 *
 * @f{eqnarray*}{
 *   f(x) &=& \begin{cases}
 *     0 & x \leq -3\\
 *     x & x \geq +3\\
 *     \frac{x * (x + 3)}{6} & otherwise\\
 *   \end{cases} \\
 *   f'(x) &=& \begin{cases}
 *     0 & x \leq -3\\
 *     1 & x \geq +3\\
 *     \frac{2x + 3}{6} & otherwise\\
 *   \end{cases}
 * @f}
 */
class HardSwishFunction
{
 public:
  /**
   * Computes the Hard Swish function.
   *
   * @param x Input data.
   * @return f(x).
   */
  static double Fn(const double x)
  {
    double x2 = x + 3.0;
    x2 = x2 > 0.0 ? x2 : 0.0;
    x2 = x2 < 6.0 ? x2 : 6.0;
    x2 = x * x2 / 6.0;

    return x2;
  }

  /**
   * Computes the Hard Swish function.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template <typename InputVecType, typename OutputVecType>
  static void Fn(const InputVecType &x, OutputVecType &y)
  {
    y.set_size(size(x));

    for (size_t i = 0; i < x.n_elem; i++)
      y(i) = Fn(x(i));
  }

  /**
   * Computes the first derivative of the Hard Swish function.
   *
   * @param y Input data.
   * @return f'(x).
   */
  static double Deriv(const double y)
  {
    if (y <= -3)
      return 0;
    else if (y >= 3)
      return 1;

    return (2*y + 3.0)/6.0;
  }

  /**
   * Computes the first derivatives of the Hard Swish function.
   *
   * @param y Input activations.
   * @param x The resulting derivatives.
   */
  template <typename InputVecType, typename OutputVecType>
  static void Deriv(const InputVecType &y, OutputVecType &x)
  {
    x.set_size(size(y));

    for (size_t i = 0; i < y.n_elem; i++)
      x(i) = Deriv(y(i));
  }
}; // class HardSwishFunction

} // namespace ann
} // namespace mlpack

#endif
