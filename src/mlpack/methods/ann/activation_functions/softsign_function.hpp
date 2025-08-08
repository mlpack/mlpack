/**
 * @file methods/ann/activation_functions/softsign_function.hpp
 * @author Marcus Edel
 *
 * Definition and implementation of the softsign function as described by
 * X. Glorot and Y. Bengio.
 *
 * For more information, see the following paper.
 *
 * @code
 * @inproceedings{GlorotAISTATS2010,
 *   title={Understanding the difficulty of training deep feedforward
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

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * The softsign function, defined by
 *
 * @f{eqnarray*}{
 * f(x) &=& \frac{x}{1 + |x|} \\
 * f'(x) &=& (1 + |f(x)|)^2 \\
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
  template<typename ElemType>
  static ElemType Fn(const ElemType x)
  {
    return x / (1 + std::abs(x));
  }

  /**
   * Computes the softsign function.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template<typename InputVecType, typename OutputVecType>
  static void Fn(const InputVecType& x, OutputVecType& y)
  {
    y = x / (1 + abs(x));
  }

  /**
   * Computes the first derivative of the softsign function.
   *
   * @param x Input activation.
   * @param y Result of Fn(x).
   * @return f'(x)
   */
  template<typename ElemType>
  static ElemType Deriv(const ElemType x, const ElemType /* y */)
  {
    return 1 / std::pow(1 + std::abs(x), ElemType(2));
  }

  /**
   * Computes the first derivatives of the softsign function.
   *
   * @param x Input activation.
   * @param y Result of Fn(x).
   * @param dy The resulting derivatives.
   */
  template<typename InputVecType, typename OutputVecType, typename DerivVecType>
  static void Deriv(const InputVecType& x,
                    const OutputVecType& /* y */,
                    DerivVecType& dy)
  {
    dy = 1 / square(1 + abs(x));
  }

  /**
   * Computes the inverse of the softsign function.
   *
   * @param y Input data.
   * @return f^{-1}(y)
   */
  template<typename ElemType>
  static ElemType Inv(const ElemType y)
  {
    if (y > 0)
      return -y / (y - 1);
    else
      return y / (1 + y);
  }

  /**
   * Computes the inverse of the softsign function.
   *
   * @param y Input data.
   * @param x The resulting inverse of the input data.
   */
  template<typename InputVecType, typename OutputVecType>
  static void Inv(const InputVecType& y, OutputVecType& x)
  {
    x.set_size(arma::size(y));

    for (size_t i = 0; i < y.n_elem; ++i)
      x(i) = Inv(y(i));
  }
}; // class SoftsignFunction

} // namespace mlpack

#endif
