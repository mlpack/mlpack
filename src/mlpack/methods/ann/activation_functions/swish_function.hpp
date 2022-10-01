/**
 * @file methods/ann/activation_functions/swish_function.hpp
 * @author Vivek Pal
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
  template<typename eT>
  static void Fn(const arma::Mat<eT>& x, arma::Mat<eT>& y)
  {
    y = x / (1.0 + arma::exp(-x));
  }

  /**
   * Computes the swish function.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template<typename InputVecType, typename OutputVecType>
  static void Fn(const InputVecType& x, OutputVecType& y)
  {
    y.set_size(arma::size(x));

    for (size_t i = 0; i < x.n_elem; ++i)
      y(i) = Fn(x(i));
  }

  /**
   * Computes the first derivative of the swish function.
   *
   * @param y Input data.
   * @return f'(x)
   */
  static double Deriv(const double y)
  {
    return y / (1 + std::exp(-y)) + (1 - y / (1 + std::exp(-y))) /
                                             (1 + std::exp(-y));
  }

  /**
   * Computes the first derivatives of the swish function.
   *
   * @param y Input data.
   * @param x The resulting derivatives.
   */
  template<typename InputVecType, typename OutputVecType>
  static void Deriv(const InputVecType& y, OutputVecType& x)
  {
    x = y / (1 + arma::exp(-y)) + (1 - y / (1 + arma::exp(-y))) /
                                           (1 + arma::exp(-y));
  }
}; // class SwishFunction

} // namespace mlpack

#endif
