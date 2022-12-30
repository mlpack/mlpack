/**
 * @file methods/ann/activation_functions/gcu_function.hpp
 * @author Tarek Elsayed
 *
 * Definition and implementation of the Growing Cosine Unit function.
 *
 * For more information, see the following paper.
 *
 * @code
 * @misc{
 *   authors = {Mathew Mithra Noel, Arunkumar L, Advait Trivedi, Praneet Dutta},
 *   title = {Growing Cosine Unit: A Novel Oscillatory Activation Function 
*    That Can Speedup Training and Reduce Parameters 
*    in Convolutional Neural Networks},
 *   year = {2021}
 * }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_GCU_FUNCTION_HPP
#define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_GCU_FUNCTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * The gcu function, defined by
 *
 * @f{eqnarray*}{
 * f(x) &=& x * cos(x) \\
 * f'(x) &=& cos(x) - x sin(x) \\
 * @f}
 */
class GCUFunction
{
 public:
  /**
   * Computes the gcu function.
   *
   * @param x Input data.
   * @return f(x).
   */
  static double Fn(const double x)
  {
    return x * std::cos(x);
  }

  /**
   * Computes the gcu function.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template<typename InputVecType, typename OutputVecType>
  static void Fn(const InputVecType& x, OutputVecType& y)
  {
    y = x % arma::cos(x);
  }

  /**
   * Computes the first derivative of the gcu function.
   *
   * @param y Input activation.
   * @return f'(x)
   */
  static double Deriv(const double y)
  {
    return std::cos(y) - y * std::sin(y);
  }

  /**
   * Computes the first derivatives of the gcu function.
   *
   * @param y Input activations.
   * @param x The resulting derivatives.
   */
  template<typename InputVecType, typename OutputVecType>
  static void Deriv(const InputVecType& y, OutputVecType& x)
  {
    x = arma::cos(y) - y % arma::sin(y);
  }
}; // class GCUFunction

} // namespace mlpack

#endif
