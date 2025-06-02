/**
 * @file methods/ann/activation_functions/swish_function.hpp
 * @author Vivek Pal
 * @author Adam Kropp
 *
 * Definition and implementation of the Swish function as described by
 * Prajit Ramachandran, Barret Zoph & Quoc V. Le
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_SWISH_FUNCTION_HPP
#define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_SWISH_FUNCTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * The swish function, defined by
 *
 * @f{eqnarray*}{
 * f(x) &=& x \cdot \sigma(x) \\
 * f'(x) &=& f(x) + \sigma(x) (1 - f(x)) \\
 * \sigma(x) &=& frac{1}{1 + e^{-x}}
 * @f}
 */
class SwishFunction
{
 public:
  /**
   * Computes the swish function.
   *
   * @param x Input data.
   * @return f(x).
   */
  static double Fn(const double x)
  {
    return x / (1.0 + std::exp(-x));
  }

  /**
   * Computes the swish function using a matrix as input.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template<typename MatType>
  static void Fn(const MatType& x, MatType& y,
     const typename std::enable_if_t<IsMatrix<MatType>::value>* = 0)
  {
    y = x / (1.0 + exp(-x));
  }

  /**
   * Computes the swish function.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template<typename VecType>
  static void Fn(const VecType& x, VecType& y,
      const typename std::enable_if_t<IsVector<VecType>::value>* = 0)
  {
    y.set_size(arma::size(x));

    for (size_t i = 0; i < x.n_elem; ++i)
      y(i) = Fn(x(i));
  }

  /**
   * Computes the first derivative of the swish function.
   *
   * @param x Input activation.
   * @param y Result of Fn(x).
   * @return f'(x)
   */
  static double Deriv(const double x, const double y)
  {
    double sigmoid = y / x; // save an exp
    return x == 0 ? 0.5 : sigmoid * (1.0 + x * (1.0 - sigmoid));
    // the expression above is indeterminate at 0, even though
    // the expression solely in terms of x is defined (= 0.5)
  }

  /**
   * Computes the first derivatives of the swish function.
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
    dy = (y / x) % (1.0 + x - y);
    // the expression above is indeterminate at 0, even though
    // the expression solely in terms of x is defined (= 0.5)
    dy(arma::find(x == 0)).fill(0.5);
  }
}; // class SwishFunction

} // namespace mlpack

#endif
