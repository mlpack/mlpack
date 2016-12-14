/**
 * @file softsign_function.hpp
 * @author Marcus Edel
 *
 * Definition and implementation of the softsign function as described by
 * X. Glorot and Y. Bengio.
 *
 * For more information, see the following paper.
 *
 * @code
 * @inproceedings{GlorotAISTATS2010,
 *   title={title={Understanding the difficulty of training deep feedforward
 *   neural networks},
 *   author={Glorot, Xavier and Bengio, Yoshua},
 *   booktitle={Proceedings of AISTATS 2010},
 *   year={2010}
 * }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_SOFTSIGN_FUNCTION_HPP
#define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_SOFTSIGN_FUNCTION_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The softsign function, defined by
 *
 * @f{eqnarray*}{
 * f(x) &=& \frac{x}{1 + |x|} \\
 * f'(x) &=& (1 - |x|)^2 \\
 * f(x) &=& \left\{
 *   \begin{array}{lr}
 *     -\frac{y}{y-1} & : x > 0 \\
 *     \frac{x}{1 + x} & : x \le 0
 *   \end{array}
 * \right.
 * @f}
 */
class SoftsignFunction
{
  public:
  /**
   * Computes the softsign function.
   *
   * @param x Input data.
   * @return f(x).
   */
  static double fn(const double x)
  {
    if (x < DBL_MAX)
      return x > -DBL_MAX ? x / (1.0 + std::abs(x)) : -1.0;
    return 1.0;
  }

  /**
   * Computes the softsign function.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template<typename InputVecType, typename OutputVecType>
  static void fn(const InputVecType& x, OutputVecType& y)
  {
    y = x;

    for (size_t i = 0; i < x.n_elem; i++)
      y(i) = fn(x(i));
  }

  /**
   * Computes the first derivative of the softsign function.
   *
   * @param y Input data.
   * @return f'(x)
   */
  static double deriv(const double y)
  {
    return std::pow(1.0 - std::abs(y), 2);
  }

  /**
   * Computes the first derivatives of the softsign function.
   *
   * @param y Input activations.
   * @param x The resulting derivatives.
   */
  template<typename InputVecType, typename OutputVecType>
  static void deriv(const InputVecType& y, OutputVecType& x)
  {
    x = arma::pow(1.0 - arma::abs(y), 2);
  }

  /**
   * Computes the inverse of the softsign function.
   *
   * @param y Input data.
   * @return f^{-1}(y)
   */
  static double inv(const double y)
  {
    if (y > 0)
      return y < 1 ? -y / (y - 1) : DBL_MAX;
    else
      return y > -1 ? y / (1 + y) : -DBL_MAX;
  }

  /**
   * Computes the inverse of the softsign function.
   *
   * @param y Input data.
   * @param x The resulting inverse of the input data.
   */
  template<typename InputVecType, typename OutputVecType>
  static void inv(const InputVecType& y, OutputVecType& x)
  {
    x = y;

    for (size_t i = 0; i < y.n_elem; i++)
      x(i) = inv(y(i));
  }
}; // class SoftsignFunction

} // namespace ann
} // namespace mlpack

#endif
