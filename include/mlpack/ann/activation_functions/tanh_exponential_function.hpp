/**
 * @file methods/ann/activation_functions/tanh_exponential_function.hpp
 * @author Mayank Raj
 * @author Adam Kropp
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
    y = x % arma::tanh(exp(x));
  }

  /**
   * Computes the first derivative of the TanhExp function.
   *
   * @param x Input activation.
   * @param y Result of Fn(x).
   * @return f'(x)
   */
  static double Deriv(const double x, const double y)
  {
    // leverage both y and x
    return x == 0 ? std::tanh(1) :
        y / x + x * std::exp(x) * (1 - std::pow(y / x, 2));
  }

  /**
   * Computes the first derivatives of the tanh function.
   *
   * @param x Input activation.
   * @param y Result of Fn(x).
   * @param dy The resulting derivatives.
   */
  template<typename InputVecType, typename OutputVecType, typename DerivVecType>
  static void Deriv(const InputVecType& x,
                    const OutputVecType& y,
                    DerivVecType& dy)
  {
    // leverage both y and x
    dy = y / x + x % exp(x) % (1 - pow(y / x, 2));
    // the expression above is indeterminate at 0, even though
    // the expression solely in terms of x is defined (= tanh(1))
    dy(arma::find(x == 0)).fill(std::tanh(1));
  }
}; // class TanhExpFunction

} // namespace mlpack

#endif
