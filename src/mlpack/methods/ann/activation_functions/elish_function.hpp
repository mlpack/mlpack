/**
 * @file methods/ann/activation_functions/elish_function.hpp
 * @author Bisakh Mondal
 *
 * Definition and implementation of the ELiSH function as described by
 * Mina Basirat and Peter M. Roth.
 *
 * For more information see the following paper
 *
 * @code
 * @misc{Basirat2018,
 *    title = {The Quest for the Golden Activation Function},
 *    author = {Mina Basirat and Peter M. Roth},
 *    year = {2018},
 *    url = {https://arxiv.org/pdf/1808.00783.pdf},
 *    eprint = {1808.00783},
 *    archivePrefix = {arXiv},
 *    primaryClass = {cs.NE} }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_ELISH_FUNCTION_HPP
#define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_ELISH_FUNCTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * The ELiSH function, defined by
 *
 * @f{eqnarray*}{
 *   f(x) &=& \begin{cases}
 *      x / (1 + e^{-x}) & x \geq 0\\
 *     (e^{x} - 1) / (1 + e^{-x}) & x < 0.\\
 *   \end{cases} \\
 *   f'(x) &=& \begin{cases}
 *      1 / (1 + e^{-y}) + y * e^{-y} / (1 + e^{-y})^2 & x \geq 0\\
 *      e^y - 2 / (1 + e^y) + 2 / (1 + e^y)^2 & x < 0.\\
 *   \end{cases}
 * @f}
 */
class ElishFunction
{
 public:
  /**
   * Computes the ELiSH function.
   *
   * @param x Input data.
   * @return f(x).
   */
  static double Fn(const double x)
  {
    if (x < 0.0)
      return (std::exp(x) - 1) / (1 + std::exp(-x));

    return x / (1 + std::exp(-x));
  }

  /**
   * Computes the ELiSH function.
   *
   * @param x Input data.
   * @param y The resulting output activations.
   */
  template<typename InputVecType, typename OutputVecType>
  static void Fn(const InputVecType& x, OutputVecType& y)
  {
    y = ((x < 0.0) % ((arma::exp(x) -1) / (1 + arma::exp(-x))))
        + ((x >= 0.0) % (x / (1 + arma::exp(-x))));
  }

  /**
   * Computes the first derivatives of ELiSH function.
   *
   * @param y Input data.
   * @return f'(x).
   */
  static double Deriv(const double y)
  {
    if (y < 0.0)
    {
      return std::exp(y) - 2 / (1 + std::exp(y)) +
          2 / std::pow(1 + std::exp(y) , 2);
    }

    return 1 / (1 + std::exp(-y)) + y * std::exp(-y) /
        std::pow(1 + std::exp(-y) , 2);
  }

  /**
   * Computes the first derivatives of the ELiSH function.
   *
   * @param y Input data.
   * @param x The resulting derivatives.
   */
  template<typename InputVecType, typename OutputVecType>
  static void Deriv(const InputVecType& y, OutputVecType& x)
  {
    x = ((y < 0.0) % (arma::exp(y) - 2 / (1 + arma::exp(y)) + 2 / arma::pow(
        1 + arma::exp(y), 2))) + ((y >= 0.0) % (1 / (1 + arma::exp(-y)) + y %
        arma::exp(-y) / arma::pow(1 + arma::exp(-y), 2)));
  }
}; // class ElishFunction

} // namespace mlpack

#endif
