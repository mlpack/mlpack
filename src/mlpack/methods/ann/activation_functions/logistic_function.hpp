/**
 * @file methods/ann/activation_functions/logistic_function.hpp
 * @author Marcus Edel
 *
 * Definition and implementation of the logistic function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_LOGISTIC_FUNCTION_HPP
#define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_LOGISTIC_FUNCTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * The logistic function, defined by
 *
 * @f{eqnarray*}{
 * f(x) &=& \frac{1}{1 + e^{-x}} \\
 * f'(x) &=& f(x) * (1 - f(x)) \\
 * f^{-1}(y) &=& ln(\frac{y}{1-y})
 * @f}
 */
class LogisticFunction
{
 public:
  /**
   * Computes the logistic function.
   *
   * @param x Input data.
   * @return f(x).
   */
  template<typename eT>
  static double Fn(const eT x)
  {
    if (x < arma::Datum<eT>::log_max)
    {
      if (x > -arma::Datum<eT>::log_max)
        return 1.0 / (1.0 + std::exp(-x));

      return 0.0;
    }

    return 1.0;
  }

  /**
   * Computes the logistic function.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template<typename InputVecType, typename OutputVecType>
  static void Fn(const InputVecType& x, OutputVecType& y)
  {
    y = (1.0 / (1 + exp(-x)));
  }

  /**
   * Computes the first derivative of the logistic function.
   *
   * @param x Input activation.
   * @param y Result of Fn(x).
   * @return f'(x)
   */
  static double Deriv(const double /* x */, const double y)
  {
    return y * (1.0 - y);
  }

  /**
   * Computes the first derivatives of the logistic function.
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
    dy = y % (1.0 - y);
  }

  /**
   * Computes the inverse of the logistic function.
   *
   * @param y Input data.
   * @return f^{-1}(y)
   */
  static double Inv(const double y)
  {
    return arma::trunc_log(y / (1 - y));
  }

  /**
   * Computes the inverse of the logistic function.
   *
   * @param y Input data.
   * @param x The resulting inverse of the input data.
   */
  template<typename InputVecType, typename OutputVecType>
  static void Inv(const InputVecType& y, OutputVecType& x)
  {
    x = arma::trunc_log(y / (1 - y));
  }
}; // class LogisticFunction

} // namespace mlpack

#endif
