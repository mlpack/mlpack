/**
 * @file methods/ann/activation_functions/silu_function.hpp
 * @author Fawwaz Mayda
 * @author Adam Kropp
 *
 * Definition and implementation of the Sigmoid Weighted Linear Unit function
 * (SILU).
 *
 * For more information see the following paper
 *
 * @code
 * @misc{elfwing2017sigmoidweighted ,
 *    title = {Sigmoid-Weighted Linear Units for Neural Network Function
 *             Approximation in Reinforcement Learning},
 *    author = {Stefan Elfwing and Eiji Uchibe and Kenji Doya},
 *    year = {2017},
 *    url = {https://arxiv.org/pdf/1702.03118.pdf},
 *    eprint = {1702.03118},
 *    archivePrefix = {arXiv},
 *    primaryClass = {cs.LG} }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_SILU_FUNCTION_HPP
#define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_SILU_FUNCTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * The SILU function, defined by
 *
 * @f{eqnarray*}{
 * f(x) &=& x * \frac{1}{1 + e^{-x}}\\
 * f'(x) &=& \frac{1}{1 + e^{-x}} * (1 + x * (1-\frac{1}{1 + e^{-x}}))\\
 * @f}
 */
class SILUFunction
{
 public:
  /**
   * Computes the SILU function.
   *
   * @param x Input data.
   * @return f(x).
   */
  static double Fn(const double x)
  {
    return x / (1.0 + std::exp(-x));
  }

  /**
   * Computes the SILU function.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template<typename InputVecType, typename OutputVecType>
  static void Fn(const InputVecType &x, OutputVecType &y)
  {
    y = x / (1.0 + exp(-x));
  }

  /**
   * Computes the first derivative of the SILU function.
   *
   * @param x Input activation.
   * @param y Result of Fn(x).
   * @return f'(x)
   */
  static double Deriv(const double x, const double y)
  {
    // since y = x * sigmoid(x)
    double sigmoid = y / x; // save an exp
    return x == 0 ? 0.5 : sigmoid * (1.0 + x * (1.0 - sigmoid));
    // the expression above is indeterminate at 0, even though
    // the expression solely in terms of x is defined (= 0.5)
  }

  /**
   * Computes the first derivatives of the SILU function.
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
    // since y = x * sigmoid(x)
    // DerivVecType sigmoid = y / x;
    // dy = sigmoid % (1.0 + x % (1.0 - sigmoid));
    dy = (y / x) % (1.0 + x - y);
    // the expression above is indeterminate at 0, even though
    // the expression solely in terms of x is defined (= 0.5)
    dy(arma::find(x == 0)).fill(0.5);
  }
}; // class SILUFunction

} // namespace mlpack

#endif
