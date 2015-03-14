/**
 * @file softsign_function.hpp
 * @author Marcus Edel
 *
 * Definition and implementation of the softsign function as described by
 * X. Glorot and Y. Bengio.
 *
 * For more information, see the following paper.
 *
 * @code
 * @inproceedings{GlorotAISTATS2010,
 *   title={title={Understanding the difficulty of training deep feedforward
 *   neural networks},
 *   author={Glorot, Xavier and Bengio, Yoshua},
 *   booktitle={Proceedings of AISTATS 2010},
 *   year={2010}
 * }
 * @endcode
 */
#ifndef __MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_SOFTSIGN_FUNCTION_HPP
#define __MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_SOFTSIGN_FUNCTION_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The softsign function, defined by
 *
 * @f[
 * f(x) &=& \frac{x}{1 + \abs{x}} \\
 * f'(x) &=& (1 - \abs{x})^2
 * f(x) &=& \left\{
 *   \begin{array}{lr}
 *     -\frac{y}{y-1} & : x > 0 \\
 *     \frac{x}{1 + x} & : x \le 0
 *   \end{array}
 * \right
 * @f]
 */
class SoftsignFunction
{
  public:
  /**
   * Computes the softsign function.
   *
   * @param x Input data.
   * @return f(x).
   */
  static double fn(const double x)
  {
    if (x < DBL_MAX)
      return x > -DBL_MAX ? x / (1.0 + std::abs(x)) : -1.0;
    return 1.0;
  }

  /**
   * Computes the softsign function.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template<typename InputVecType, typename OutputVecType>
  static void fn(const InputVecType& x, OutputVecType& y)
  {
    y = x;
    y.transform( [](double x) { return fn(x); } );
  }

  /**
   * Computes the first derivative of the softsign function.
   *
   * @param y Input data.
   * @return f'(x)
   */
  static double deriv(const double y)
  {
    return std::pow(1.0 - std::abs(y), 2);
  }

  /**
   * Computes the first derivatives of the softsign function.
   *
   * @param y Input activations.
   * @param x The resulting derivatives.
   */
  template<typename InputVecType, typename OutputVecType>
  static void deriv(const InputVecType& y, OutputVecType& x)
  {
    x = arma::pow(1.0 - arma::abs(y), 2);
  }

  /**
   * Computes the inverse of the softsign function.
   *
   * @param y Input data.
   * @return f^{-1}(y)
   */
  static double inv(const double y)
  {
    if (y > 0)
      return y < 1 ? -y / (y - 1) : DBL_MAX;
    else
      return y > -1 ? y / (1 + y) : -DBL_MAX;
  }

  /**
   * Computes the inverse of the softsign function.
   *
   * @param y Input data.
   * @param x The resulting inverse of the input data.
   */
  template<typename InputVecType, typename OutputVecType>
  static void inv(const InputVecType& y, OutputVecType& x)
  {
    x = y;
    x.transform( [](double y) { return inv(y); } );
  }
}; // class SoftsignFunction

}; // namespace ann
}; // namespace mlpack

#endif
