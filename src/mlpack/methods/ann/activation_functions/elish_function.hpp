/**
 * @file methods/ann/activation_functions/elish_function.hpp
 * @author Bisakh Mondal
 * @author Adam Kropp
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
    y = ((x < 0.0) % ((exp(x) - 1) / (1 + exp(-x))))
        + ((x >= 0.0) % (x / (1 + exp(-x))));
  }

  /**
   * Computes the first derivatives of ELiSH function.
   *
   * @param x Input activation.
   * @param y Result of Fn(x).
   * @return f'(x).
   */
  static double Deriv(const double x, const double y)
  {
    if (x < 0.0)
    {
      return std::exp(x) - 2 / (1 + std::exp(x)) +
          2 / std::pow(1 + std::exp(x) , 2);
    }
    else if (x == 0)
    {
      return 0.5; // the expression below is indeterminate at 0, even though
                  // the expression solely in terms of x is defined (= 0.5)
    }
    else
    {
      return (y / x) * (1 + x - y);
    }
  }

  /**
   * Computes the first derivatives of the ELiSH function.
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
    // simplified the x>=0 part to be in terms of x and y -- maybe
    // the x<0 part can be as well?
    // the expression is indeterminate at 0, even though
    // the expression solely in terms of x is defined (= 0.5)
    // only calculate exp(x) once for each element where x < 0
    // this gives approx 3x speedup, despite allocating the temp vector
    DerivVecType ex = (x < 0) % exp(x);
    dy = ((x < 0) % ((ex - 2 / (1 + ex) + 2 / pow(1 + ex, 2)))) +
         ((x > 0) % ((y / x) % (1.0 + x - y)));
    // need to do this here, because the /x above gives nans even when the
    // condition is not met (e.g. when x > 0 is false)
    dy(arma::find(x == 0)).fill(0.5);
  }
}; // class ElishFunction

} // namespace mlpack

#endif
