/**
 * @file methods/ann/activation_functions/bipolar_sigmoid_function.hpp
 * @author Satyam Shukla
 *
 * Definition and implementation of the bipolar sigmoid function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_BIPOLAR_SIGMOID_FUNCTION_HPP
#define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_BIPOLAR_SIGMOID_FUNCTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * The Bipolar Sigmoid function, defined by
 *
 * @f{eqnarray*}{
 * f(x) = 1 - exp(-x) / 1 + exp(-x)\\
 * f'(x) = 2*exp(x) / (1 + exp(x))^2\\
 * f'(x) = (1 - f(x)^2) / 2 \\
 * @f}
 */
class BipolarSigmoidFunction
{
 public:
  /**
   * Computes the Bipolar Sigmoid function.
   *
   * @param x Input data.
   * @return f(x).
   */
  static double Fn(const double x)
  {
    return (1 - std::exp(-x)) / (1 + std::exp(-x));
  }

  /**
   * Computes the bipolar sigmoid function.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template<typename InputVecType, typename OutputVecType>
  static void Fn(const InputVecType& x, OutputVecType& y)
  {
    y = (1 - exp(-x)) / (1 + exp(-x));
  }

  /**
   * Computes the first derivative of the Bipolar Sigmoid function.
   *
   * @param x Input activation.
   * @param y Result of Fn(x).
   * @return f'(x)
   */
  static double Deriv(const double /* x */, const double y)
  {
    return (1.0 - std::pow(y, 2)) / 2.0;
  }

  /**
   * Computes the first derivatives of the Bipolar Sigmoid function.
   *
   * @param x Input activation.
   * @param y Result of Fn(x).
   * @param dy The resulting derivatives.
   */
  template<typename InputVecType, typename OutputVecType, typename DerivVecType>
  static void Deriv(const InputVecType& /* x */,
                    const OutputVecType& y,
                    DerivVecType& dy)
  {
    dy = (1.0 - pow(y, 2)) / 2.0;
  }
}; // class BipolarSigmoidFunction

} // namespace mlpack

#endif
