/**
 * @file identity_function.hpp
 * @author Marcus Edel
 *
 * Definition and implementation of the identity function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_IDENTITY_FUNCTION_HPP
#define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_IDENTITY_FUNCTION_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The identity function, defined by
 *
 * @f{eqnarray*}{
 * f(x) &=& x \\
 * f'(x) &=& 1
 * @f}
 */
class IdentityFunction
{
 public:
  /**
   * Computes the identity function.
   *
   * @param x Input data.
   * @return f(x).
   */
  static double fn(const double x)
  {
    return x;
  }

  /**
   * Computes the identity function.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template<typename InputVecType, typename OutputVecType>
  static void fn(const InputVecType& x, OutputVecType& y)
  {
    y = x;
  }

  /**
   * Computes the first derivative of the identity function.
   *
   * @param x Input data.
   * @return f'(x)
   */
  static double deriv(const double /* unused */)
  {
    return 1.0;
  }

  /**
   * Computes the first derivatives of the identity function.
   *
   * @param y Input activations.
   * @param x The resulting derivatives.
   */
  template<typename InputVecType, typename OutputVecType>
  static void deriv(const InputVecType& y, OutputVecType& x)
  {
    x.ones(y.n_elem);
  }

  /**
   * Computes the first derivatives of the identity function using a 3rd order
   * tensor as input.
   *
   * @param y Input activations.
   * @param x The resulting derivatives.
   */
  template<typename eT>
  static void deriv(const arma::Cube<eT>& y, arma::Cube<eT>& x)
  {
    x.ones(y.n_rows, y.n_cols, y.n_slices);
  }


}; // class IdentityFunction

} // namespace ann
} // namespace mlpack

#endif
