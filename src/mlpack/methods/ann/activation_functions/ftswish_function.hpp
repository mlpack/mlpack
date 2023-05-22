/**
 * @file methods/ann/activation_functions/ftswish_function.hpp
 * @author Mayank Raj
 *
 * Definition and implementation of the ftswish function as described by
 * Hock Hung Chieng, Noorhaniza Wahid, Ong Pauline,
 * Sai Raj Kishore Perla .
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_FTSWISH_FUNCTION_HPP
#define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_FTSWISH_FUNCTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * The FTSwish function, defined by:
 *
 * @f{eqnarray*}{
 *  f(x) &=& \begin{cases}
 * \frac{x}{1 + \exp(-x)} + T, & \text{for } x \geq 0 \\
 * T, & \text{for } x < 0
 * \end{cases}
 * 
 *  f'(x) &=&  = \begin{cases}
 * \sigma(x)(1 - f(x)) + f(x), & \text{for } x \geq 0 \\
 * 0, & \text{for } x < 0
 * \end{cases}
 * @f}
 */
class FTSwishFunction
{
 public:
  /**
   * Computes the FTSwish function.
   *
   * @param x Input data.
   * @param T The constant value for x < 0.
   * @return FTSwish(x).
   */
  static double Fn(const double x, const double T)
  {
    if (x >= 0)
      return x / (1.0 + std::exp(-x)) + T;
    else
      return T;
  }

  /**
   * Computes the FTSwish function using a matrix as input.
   *
   * @param x Input data.
   * @param T The constant value for x < 0.
   * @param y The resulting output activation.
   */
  template<typename eT>
  static void Fn(const arma::Mat<eT>& x, const double T,
                 arma::Mat<eT>& y)
  {
    y.set_size(x.n_rows, x.n_cols);

    for (size_t i = 0; i < x.n_elem; ++i)
      y(i) = Fn(x(i), T);
  }

  /**
   * Computes the FTSwish function.
   *
   * @param x Input data.
   * @param T The constant value for x < 0.
   * @param y The resulting output activation.
   */
  template<typename InputVecType, typename OutputVecType>
  static void Fn(const InputVecType& x, const double T,
                 OutputVecType& y)
  {
    y.set_size(arma::size(x));

    for (size_t i = 0; i < x.n_elem; ++i)
      y(i) = Fn(x(i), T);
  }

  /**
   * Computes the derivative of the FTSwish function.
   *
   * @param x Input data.
   * @return f'(x)
   */
  static double Deriv(const double x)
  {
    if (x >= 0)
    {
      const double sigmoid = 1.0 / (1.0 + std::exp(-x));
      return sigmoid * (1.0 - (x * sigmoid)) + (x * sigmoid);
    }
    else
    {
      return 0.0;
    }
  }

  /**
   * Computes the derivatives of the FTSwish function.
   *
   * @param x Input data.
   * @param y The resulting derivatives.
   */
  template<typename InputVecType, typename OutputVecType>
  static void Deriv(const InputVecType& x, OutputVecType& y)
  {
    y.set_size(arma::size(x));

    for (size_t i = 0; i < x.n_elem; ++i)
      y(i) = Deriv(x(i));
  }

};

} // namespace mlpack

#endif
