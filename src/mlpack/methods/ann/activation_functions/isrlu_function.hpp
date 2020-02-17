/**
 * @file isrlu_function.hpp
 * @author Pranav Reddy
 *
 * Definition and implementation of the ISRLU function
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

 #ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_ISRLU_FUNCTION_HPP
 #define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_ISRLU_FUNCTION_HPP

 #include <mlpack/prereqs.hpp>

namespace mlpack{
namespace ann /** Artifical Neural Network. */{

/**
 * The ISRLU function, defined by
 * 
 * @f{eqnarray*}{
 * f(x) &=& x/sqrt(1 + alpha*x*x) if x<0
 *      &=& x                     if x>=0
 *
 * f'(x) &=& cube(1/sqrt(1 + alpha*x*x)) if x<0
 *       &=& 1                           if x>=0
 * @f
 */

class ISRLUFunction
{

 public:
  /**
   * Computes the ISRLU function
   * 
   * @param x Input data.
   * @param alpha paramter, default value = 0.1
   * @return f(x)
   */
  static double Fn(const double x, const double alpha = 0.1)
  {
    if(x<0) 
      return x / (std::sqrt(1 + alpha*x*x));
    else
      return x;
  }

  /**
   * Computes the ISRLU function using a matrix as a input
   * 
   * @param x Input data.
   * @param y The resulting output activation
   * @param alpha parameter, default value = 0.1
   */
  template<typename eT>
  static void Fn(const arma::Mat<eT>& x, arma::Mat<eT>& y, const double alpha = 0.1)
  {
    if(x<0)
      y = x / (arma::sqrt(1 + alpha * arma::pow(x,2)));
    else
      y=x;
  }

  /**
   * Computes the ISRLU function.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   * @param alpha parameter, default value = 0.1
   */
  template<typename InputVecType, typename OutputVecType>
  static void Fn(const InputVecType& x, OutputVecType& y, const double alpha = 0.1)
  {
    y.set_size(arma::size(x));

    for (size_t i = 0; i < x.n_elem; i++)
      y(i) = Fn(x(i));
  }

  /**
   * Computes the inverse of the ISRLU function
   * 
   * @param y
   * @param alpha
   * @return f^{-1}(y)
   */
  static double Inv(const double y, const double alpha = 0.1)
  {
    if(y<0)
      return y / std::sqrt(1 - alpha * y * y);
    else
      return y;
  }

  /**
   * Computes the inverse of the ISRLU function
   * 
   * @param y Input data.
   * @param x The resulting inverse of the input data
   * @param alpha parameter, default value = 0.1
   */
  template<typename InputVecType, typename OutputVecType>
  static void Inv(const InputVecType& y, OutputVecType& x, const double alpha = 0.1)
  {
    if(y<0)
      x = y / arma::sqrt(1 - alpha * arma::pow(y, 2 ));
    else
      x = y;
  }

  /**
   * Computes the first derivate of the ISRLU function
   * 
   * @param y Input activation
   * @param alpha parameter, default value = 0.1
   * @return f'(x) where f(x) = y
   */
  static double Deriv(const double y, const double alpha = 0.1)
  {
    double x = Inv(y, alpha);
    if(y<0)
      return std::pow(y / x, 3); 
    else
      return 1;
  }

  /**
   * Computes the first derivative of the ISRLU function.
   *
   * @param y Input activations.
   * @param x The resulting derivatives. Should be the matrix used to calculate activation y 
   * @param alpha parameter, default value = 0.1
   */
  template<typename InputVecType, typename OutputVecType>
  static void Deriv(const InputVecType& y, OutputVecType& x, const double alpha = 0.1)
  {
    if(y<0)
      x = arma::pow(y / x, 3);
    else
      x = 1;
  }

}; // class ISRLUFunction

} // namespace ann
} // namespace mlpack

#endif 
