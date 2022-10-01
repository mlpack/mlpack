/**
 * @file methods/ann/activation_functions/tanh_exponential_function.hpp
 * @author Mayank Raj
 *
 * Definition and implementation of the Tanh exponential  function.
 *
 * For more information see the following paper
 *
 * @code
 * @misc{The Institution of Engineering and Technology 2015 ,
 *    title = {TanhExp: A Smooth Activation Function with High Convergence Speed
 *             for Lightweight Neural Networks},
 *    author = {Xinyu Liu and Xiaoguang Di},
 *    year = {2020},
 *    url = {https://arxiv.org/pdf/2003.09855v2.pdf},
 *    eprint = {2003.09855v2},
 *    archivePrefix = {arXiv},
 *    primaryClass = {cs.LG} }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_TANH_EXPONENTIAL_FUNCTION_HPP
#define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_TANH_EXPONENTIAL_FUNCTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * The TanhExp function, defined by
 *
 * @f{eqnarray*}{
 * f(x) = x * tanh(e^x)\\
 * f'(x) = tanh(e^x) - x*e^x*(tanh(e^x)^2 - 1)\\
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
    return x * std::tanh(std::exp(x));
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
    y = x % arma::tanh(arma::exp(x));
  }

  /**
   * Computes the first derivative of the TanhExp function.
   *
   * @param y Input activation.
   * @return f'(x)
   */
  static double Deriv(const double y)
  {
    return std::tanh(std::exp(y)) - y * std::exp(y) *
        (std::pow(std::tanh(std::exp(y)), 2) - 1);
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
    x = arma::tanh(arma::exp(y)) - y % arma::exp(y) %
        (arma::pow(arma::tanh(arma::exp(y)), 2) - 1);
  }
}; // class TanhExpFunction

} // namespace mlpack

#endif
