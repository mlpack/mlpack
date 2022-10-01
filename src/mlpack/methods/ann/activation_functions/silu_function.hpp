/**
 * @file methods/ann/activation_functions/silu_function.hpp
 * @author Fawwaz Mayda
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
    y = x / (1.0 + arma::exp(-x));
  }

  /**
   * Computes the first derivative of the SILU function.
   *
   * @param y Input activation.
   * @return f'(x)
   */
  static double Deriv(const double x)
  {
    double sigmoid = 1.0 / (1.0 + std::exp(-x));
    return sigmoid * (1.0 + x * (1.0 - sigmoid));
  }

  /**
   * Computes the first derivatives of the SILU function.
   *
   * @param y Input activations.
   * @param x The resulting derivatives.
   */
  template<typename InputVecType, typename OutputVecType>
  static void Deriv(const InputVecType &x, OutputVecType &y)
  {
    OutputVecType sigmoid = 1.0 / (1.0 + arma::exp(-x));
    y = sigmoid % (1.0 + x % (1.0 - sigmoid));
  }
}; // class SILUFunction

} // namespace mlpack

#endif
