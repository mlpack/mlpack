/**
 * @file identity_function.hpp
 * @author Marcus Edel
 *
 * Definition and implementation of the identity function.
 *
 * This file is part of mlpack 2.0.0.
 *
 * mlpack is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * mlpack is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * mlpack.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_IDENTITY_FUNCTION_HPP
#define __MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_IDENTITY_FUNCTION_HPP

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
