/**
 * @file isru_function.hpp
 * @author Prince Gupta
 *
 * Definition and implementation of the ISRU function
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_ISRU_FUNCTION_HPP
#define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_ISRU_FUNCTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack{
namespace ann /** Artifical Neural Network. */{

/**
 * The ISRU function, defined by
 * 
 * @f{eqnarray*}{
 * f(x) &=& x / sqrt(1 + alpha* x ^ {2})
 * f'(x) &=& cube(1 / sqrt(1 + alpha * x ^ {2}))
 * @f
 */

class ISRUFunction
{
 public:
  /**
   * Computes the ISRU function
   * 
   * @param x Input data.
   * @param alpha paramter, default value = 0.1
   * @return f(x)
   */
  static double Fn(const double x, const double alpha = 0.1)
  {
    return x / (std::sqrt(1 + alpha*x*x));
  }

  /**
   * Computes the ISRU function using a matrix as a input
   * 
   * @param x Input data.
   * @param y The resulting output activation
   * @param alpha parameter, default value = 0.1
   */
  template<typename eT>
  static void Fn(const arma::Mat<eT>& x,
                 arma::Mat<eT>& y,
                 const double alpha = 0.1)
  {
    y = x / (arma::sqrt(1 + alpha * arma::pow(x, 2)));
  }

  /**
   * Computes the ISRU function.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   * @param alpha parameter, default value = 0.1
   */
  template<typename InputVecType, typename OutputVecType>
  static void Fn(const InputVecType& x,
                 OutputVecType& y,
                 const double alpha = 0.1)
  {
    y = x / (arma::sqrt(1 + alpha * arma::pow(x, 2)));
  }

  /**
   * Computes the inverse of the ISRU function
   * 
   * @param y
   * @param alpha
   * @return f^{-1}(y)
   */
  static double Inv(const double y, const double alpha = 0.1)
  {
    return y / std::sqrt(1 - alpha * y * y);
  }

  /**
   * Computes the inverse of the ISRU function
   * 
   * @param y Input data.
   * @param x The resulting inverse of the input data
   * @param alpha parameter, default value = 0.1
   */
  template<typename InputVecType, typename OutputVecType>
  static void Inv(const InputVecType& y,
                  OutputVecType& x,
                  const double alpha = 0.1)
  {
    x = y / arma::sqrt(1 - alpha * arma::pow(y, 2));
  }

  /**
   * Computes the first derivate of the ISRU function
   * 
   * @param y Input activation
   * @param alpha parameter, default value = 0.1
   * @return f'(x) where f(x) = y
   */
  static double Deriv(const double y, const double alpha = 0.1)
  {
    if (y == 0)
      return 1;
    double x = Inv(y, alpha);
    return std::pow(y / x, 3);
  }

  /**
   * Computes the first derivative of the ISRU function.
   *
   * @param y Input activations.
   * @param x The resulting derivatives. Should be the matrix used to calculate activation y 
   * @param alpha parameter, default value = 0.1
   */
  template<typename InputVecType, typename OutputVecType>
  static void Deriv(const InputVecType& y,
                    OutputVecType& x, const double alpha = 0.1)
  {
    Inv(y, x, alpha);
    x = arma::pow(y / x, 3);
    x.replace(arma::datum::nan, 1);
  }
}; // class ISRUFunction

} // namespace ann
} // namespace mlpack

#endif
