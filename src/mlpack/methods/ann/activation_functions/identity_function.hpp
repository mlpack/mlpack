/**
 * @file methods/ann/activation_functions/identity_function.hpp
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

#include <mlpack/prereqs.hpp>

namespace mlpack {

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
  static double Fn(const double x)
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
  static void Fn(const InputVecType& x, OutputVecType& y,
      const typename std::enable_if_t<IsVector<InputVecType>::value>* = 0,
      const typename std::enable_if_t<IsVector<OutputVecType>::value>* = 0)
  {
    y = x;
  }

  /**
   * Computes the first derivative of the identity function.
   *
   * @param * (x) Input activation.
   * @param * (y) Result of Fn(x).
   * @return f'(x)
   */
  static double Deriv(const double /* x */, const double /* y */)
  {
    return 1.0;
  }

  /**
   * Computes the first derivatives of the identity function.
   *
   * @param x Input activation.
   * @param * (y) Result of Fn(x).
   * @param dy The resulting derivatives.
   */
  template<typename InputVecType, typename OutputVecType, typename DerivVecType>
  static void Deriv(const InputVecType& x,
                    const OutputVecType& /* y */,
                    DerivVecType& dy)
  {
    dy.ones(arma::size(x));
  }

  /**
   * Computes the first derivatives of the identity function using a 3rd order
   * tensor as input.
   *
   * @param y Input activations.
   * @param dy The resulting derivatives.
   */
  template<typename CubeType>
  static void Deriv(const CubeType& y, CubeType& x,
      const typename std::enable_if_t<IsCube<CubeType>::value>* = 0)
  {
    x.ones(y.n_rows, y.n_cols, y.n_slices);
  }
}; // class IdentityFunction

} // namespace mlpack

#endif
