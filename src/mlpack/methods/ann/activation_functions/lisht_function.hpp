/**
 * @file lisht_function.hpp
 * @author Kartik Dutt
 *
 * Definition and implementation of the LiSHT function as described by
 * Swalpa K. Roy, Suvojit Manna, Shiv Ram Dubey and Bidyut B. Chaudhuri.
 *
 * For more information, see the following paper.
 *
 * @code
 * @misc{
 *   author = {Swalpa K. Roy, Suvojit Manna, Shiv R. Dubey and
 *            Bidyut B. Chaudhuri},
 *   title = {LiSHT: Non-Parametric Linearly Scaled Hyperbolic Tangent
 *           Activation Function for Neural Networks},
 *   year = {2019}
 * }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_LISHT_FUNCTION_HPP
#define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_LISHT_FUNCTION_HPP

#include <mlpack/prereqs.hpp>
#include <algorithm>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The Lisht function, defined by
 *
 * @f{eqnarray*}{
 * f(x) = x * tanh(x)
 * f'(x) = tanh(x) + x * (1 - tanh^{2}(x))
 * @f}
 */
class LishtFunction
{
 public:
  /**
   * Computes the Lisht function.
   *
   * @param x Input data.
   * @return f(x).
   */
  static double Fn(const double x)
  {
    return x * std::tanh(x);
  }

  /**
   * Computes the Lisht function.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template <typename InputVecType, typename OutputVecType>
  static void Fn(const InputVecType &x, OutputVecType &y)
  {
    y = x % arma::tanh(x);
  }

  /**
   * Computes the first derivative of the Lisht function.
   *
   * @param y Input data.
   * @return f'(x)
   */
  static double Deriv(const double y)
  {
    return (4 * y * std::exp(2 * y) + std::exp(4 * y) - 1) /
           (std::exp(4 * y) + 2 * std::exp(2 * y) + 1);
  }

  /**
   * Computes the first derivatives of the Lisht function.
   * 
   * @param y Input activations.
   * @param x The resulting derivatives.
   */
  template <typename InputVecType, typename OutputVecType>
  static void Deriv(const InputVecType &y, OutputVecType &x)
  {
    x = (4 * y % arma::exp(2 * y) + arma::exp(4 * y) - 1) /
        (arma::exp(4 * y) + 2 * arma::exp(2 * y) + 1);
  }
}; // class LishtFunction

} // namespace ann
} // namespace mlpack

#endif
