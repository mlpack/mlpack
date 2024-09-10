/**
 * @file methods/ann/activation_functions/lisht_function.hpp
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

/**
 * The LiSHT function, defined by
 *
 * @f{eqnarray*}{
 * f(x) = x * tanh(x)
 * f'(x) = tanh(x) + x * (1 - tanh^{2}(x))
 * @f}
 */
class LiSHTFunction
{
 public:
  /**
   * Computes the LiSHT function.
   *
   * @param x Input data.
   * @return f(x).
   */
  static double Fn(const double x)
  {
    return x * std::tanh(x);
  }

  /**
   * Computes the LiSHT function.
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
   * Computes the first derivative of the LiSHT function.
   *
   * @param x Input activation.
   * @param y Result of Fn(x).
   * @return f'(x)
   */
  static double Deriv(const double x, const double /* y */)
  {
    return std::tanh(x) + x * (1 - std::pow(std::tanh(x), 2));
  }

  /**
   * Computes the first derivatives of the LiSHT function.
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
    dy = arma::tanh(x) + x % (1 - pow(arma::tanh(x), 2));
  }
}; // class LishtFunction

} // namespace mlpack

#endif
