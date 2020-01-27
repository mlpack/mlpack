/**
 * @file elu_function.hpp
 * @author Mrityunjay Tripathi
 *
 * Definition and implementation of the Exponential Linear Unit activation function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_TANH_FUNCTION_HPP
#define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_TANH_FUNCTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The elu function, defined by
 *
 * @f{eqnarray*}{
 * f(x) = x if x>0 else a*(e^x-1) \\
 * f'(x) =  1 if x>0 else a*e^x\\
 * f^{-1}(x) = x if x>0 else ln(x/a + 1)\\
 * @f}
 */
class ELUFunction
{
 private:
    static float alpha;
 public:
  /**
   * Computes the ELU function.
   *
   * @param x Input data.
   * @return f(x).
   */
  ELUFunction(float a)
  {
      alpha = a;
  }
  ELUFunction()
  {
      alpha = 1.0;
  }
  static double Fn(const double x)
  {
    if(x>0)
      return x;
    return alpha*(std::exp(x)-1);
  }

  /**
   * Computes the elu function.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template<typename InputVecType, typename OutputVecType>
  static void Fn(const InputVecType& x, OutputVecType& y)
  {
    y.set_size(size(x));

    for(size_t i=0; i<x.n_elem; ++i)
    {
      y(i) = Fn(x(i));
      
    }
  }

  /**
   * Computes the first derivative of the elu function.
   *
   * @param y Input data.
   * @return f'(x)
   */
  static double Deriv(const double y)
  {
    if(y>0)
      return 1;
    else y + alpha;
    
  }

  /**
   * Computes the first derivatives of the elu function.
   *
   * @param y Input data.
   * @param x The resulting derivatives.
   */
  template<typename InputVecType, typename OutputVecType>
  static void Deriv(const InputVecType& y, OutputVecType& x)
  {
    x.set_size(size(y));

    for(size_t i=0; i<y.n_elem; ++i)
    {
      x(i) = Deriv(y(i));
    }
  }

  /**
   * Computes the inverse of the elu function.
   *
   * @param y Input data.
   * @return f^{-1}(x)
   */
  static double Inv(const double y)
  {
    if(y>0)
      return y;
    return std::log(y/alpha + 1);
  }

  /**
   * Computes the inverse of the elu function.
   *
   * @param y Input data.
   * @param x The resulting inverse of the input data.
   */
  template<typename InputVecType, typename OutputVecType>
  static void Inv(const InputVecType& y, OutputVecType& x)
  {
    x.set_size(size(y));

    for(size_t i=0; i<y.n_elem; ++i)
    {
      x(i) = Inv(y(i));
    }
  }
}; // class ELUFunction

} // namespace ann
} // namespace mlpack

#endif
