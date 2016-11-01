/**
 * @file rectifier_function.hpp
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

#include <mlpack/core.hpp>
#include <algorithm>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

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
  static double fn(const double x)
  {
    return std::max(0.0, x);
  }

  /**
   * Computes the rectifier function using a dense matrix as input.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template<typename eT>
  static void fn(const arma::Mat<eT>& x, arma::Mat<eT>& y)
  {
    y = arma::max(arma::zeros<arma::Mat<eT> >(x.n_rows, x.n_cols), x);
  }

  /**
   * Computes the rectifier function using a 3rd-order tensor as input.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template<typename eT>
  static void fn(const arma::Cube<eT>& x, arma::Cube<eT>& y)
  {
    y = x;
    for (size_t s = 0; s < x.n_slices; s++)
      fn(x.slice(s), y.slice(s));
  }

  /**
   * Computes the first derivative of the rectifier function.
   *
   * @param x Input data.
   * @return f'(x)
   */
  static double deriv(const double y)
  {
    return y > 0;
  }

  /**
   * Computes the first derivatives of the rectifier function.
   *
   * @param y Input activations.
   * @param x The resulting derivatives.
   */
  template<typename InputType, typename OutputType>
  static void deriv(const InputType& y, OutputType& x)
  {
    x = y;

    for (size_t i = 0; i < y.n_elem; i++)
      x(i) = deriv(y(i));
  }
}; // class RectifierFunction

} // namespace ann
} // namespace mlpack

#endif
