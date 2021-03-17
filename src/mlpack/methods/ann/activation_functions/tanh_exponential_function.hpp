/**
 * @file methods/ann/activation_functions/tanh_exponential_function.hpp
 * @author Mayank Raj
 *
 * Definition and implementation of the Tanh exponential  function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef TANH_EXPONENTIAL_FUNCTION_HPP_INCLUDED
#define TANH_EXPONENTIAL_FUNCTION_HPP_INCLUDED

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The TanhExp function, defined by
 *
 * @f{eqnarray*}{
 * f(x) = x * tanh(e^x)\\
 * f'(x) = tanh(e^x) + x*e^x*(sech(e^x))^2\\
 * @f}
 */
 class TanhExpFunction
{
 public:
  /**
   * Computes the TanhExp function.
   *
   * @param x Input data.
   * @return f(x).
   */
  static double Fn(const double x)
  {
    return x*std::tanh(std::exp(x));
  }

  /**
   * Computes the TanhExp function.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template<typename InputVecType, typename OutputVecType>
  static void Fn(const InputVecType& x, OutputVecType& y)
  {
    y = x*arma::tanh(arma::exp(x));
  }

  /**
   * Computes the first derivative of the TanhExp function.
   *
   * @param y Input activation.
   * @return f'(x)
   */
  static double Deriv(const double y)
  {
    return std::tanh(std::exp(y)) +
    y*std::exp(y)*std::pow(std::sech(std::exp(y)),2);
  }

  /**
   * Computes the first derivatives of the tanh function.
   *
   * @param y Input activations.
   * @param x The resulting derivatives.
   */
  template<typename InputVecType, typename OutputVecType>
  static void Deriv(const InputVecType& y, OutputVecType& x)
  {
    x = arma::tanh(arma::exp(y)) +
    y*arma::exp(y)*arma::pow(arma::sech(arma::exp(y)),2);
  }
}; // class TanhExpFunction

} // namespace ann
} // namespace mlpack

#endif // TANH_EXPONENTIAL_FUNCTION_HPP_INCLUDED
