/**
 * @file methods/ann/activation_functions/hard_sigmoid_function.hpp
 * @author Bishwa Karki
 *
 * Definition and implementation of the hard sigmoid function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_HARD_SIGMOID_FUNCTION_HPP
#define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_HARD_SIGMOID_FUNCTION_HPP

#include <mlpack/prereqs.hpp>
#include <algorithm>

namespace mlpack {

/**
 * The hard sigmoid function, defined by
 *
 * @f{eqnarray*}{
 * f(x) &=& \min(1, \max(0, 0.2 * x + 0.5)) \\
 * f'(x) &=& \left\{
 *   \begin{array}{lr}
 *     0.0 & : x={0,1} \\
 *     0.2
 *   \end{array}
 * \right.
 * @f}
 */
class HardSigmoidFunction
{
 public:
  /**
   * Computes the hard sigmoid function.
   *
   * @param x Input data.
   * @return f(x).
   */
  static double Fn(const double x)
  {
    return std::min(1.0, std::max(0.0, 0.2 * x + 0.5));
  }

  /**
   * Computes the hard sigmoid function.
   *
   * @param x Input data.
   * @param y The resulting output activations.
   */
  template<typename InputVecType, typename OutputVecType>
  static void Fn(const InputVecType& x, OutputVecType& y)
  {
    y.set_size(size(x));

    #pragma omp for
    for (size_t i = 0; i < (size_t) x.n_elem; ++i)
      y(i) = Fn(x(i));
  }

  /**
   * Computes the first derivatives of hard sigmoid function.
   *
   * @param x Input activation.
   * @param y Result of Fn(x).
   * @return f'(x)
   */
  static double Deriv(const double /* x */, const double y)
  {
    if (y == 0.0 || y == 1.0)
    {
      return 0.0;
    }
    return 0.2;
  }

  /**
   * Computes the first derivatives of the hard sigmoid function.
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
    dy.set_size(size(y));

    #pragma omp for
    for (size_t i = 0; i < (size_t) y.n_elem; ++i)
    {
      dy(i) = Deriv(x(i), y(i));
    }
  }
}; // class HardSigmoidFunction

} // namespace mlpack

#endif
