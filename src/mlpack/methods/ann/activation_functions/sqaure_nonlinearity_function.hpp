/**
 * @file methods/ann/activation_functions/square_nonlinearity_function.hpp
 * @author Shaikh Yusuf Niaz
 *
 * Definition and Implementation of Square NonLinearity (SQNL) function as described by
 * Wuraola, Adedamola and Patel, Nitish
 *
 *@code
 *@INPROCEEDINGS{8489043,
 * author={Wuraola, Adedamola and Patel, Nitish},
 * booktitle={2018 International Joint Conference on Neural Networks (IJCNN)}, 
 * title={SQNL: A New Computationally Efficient Activation Function}, 
 * year={2018},
 * volume={},
 * number={},
 * pages={1-7},
 * doi={10.1109/IJCNN.2018.8489043}}
 *@endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_SQUARE_NONLINEARITY_FUNCTION_HPP
#define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_SQUARE_NONLINEARITY_FUNCTION_HPP

#include <mlpack/prereqs.hpp>
#include <algorithm>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The Square NonLinearity function, defined by
 *
 * @f{eqnarray*}{
 *   f(x) &=& \begin{cases}
 *     1 & x>2.0\\
 *     x - \fraq{x^2}{4} & 0\leq x\leq 2.0\\
 *     x + \fraq{x^2}{4} & -2.0\leq x<0\\
 *    -1 & x<-2.0\\
 *   \end{cases} \\
 *   f'(x) &=& \begin{cases}
 *     0 & x>2.0\\
 *     1 - \fraq{x}{2} & 0\leq x\leq 2.0\\
 *     1 + \fraq{x}{2} & -2.0\leq x<0\\
 *     0 & x<-2.0\\
 *   \end{cases}
 * @f}
 */
class SquareNonLinearityFunction
{
 public:
  /**
   * Computes the Square NonLinearity function.
   *
   * @param x Input data.
   * @return f(x).
   */
  static double Fn(const double x)
  {
    if( x > 2.0) 
      return 1;
    
    else if( 0 <= x && x <= 2.0)
      return x - std::pow(x,2)/4.0;
    
    else if(-2.0 <= x && x < 0)
      return x + std::pow(x,2)/4.0;
    
    else 
      return -1.0;
  }

  /**
   * Computes the Square NonLinearity function.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template <typename InputVecType, typename OutputVecType>
  static void Fn(const InputVecType &x, OutputVecType &y)
  {
    y.set_size(size(x));

    for(size_t i =0; i < x.n_elem; ++i)
      y(i) = Fn(x(i));
  }

  /**
   * Computes the first derivative of the Square NonLinearity function.
   *
   * @param y Input data.
   * @return f'(x)
   */
  static double Deriv(const double y)
  {
    if(2.0 < y || y < -2.0)
	    return 0;

    else if(0 <= y && y <= 2.0)
	    return 1 - y/2.0;

    else 
	    return 1 + y/2.0;
  }

  /**
   * Computes the first derivatives of the Square NonLinearity function.
   *
   * @param y Input data.
   * @param x The resulting derivatives.
   */
  template <typename InputVecType, typename OutputVecType>
  static void Deriv(const InputVecType &y, OutputVecType &x)
  {
    x.set_size(size(y));

    for(size_t i = 0; i < y.n_elem; ++i)
      x(i) = Deriv(y(i));
  }
}; // class SquareNonLinearityFunction

} // namespace ann
} // namespace mlpack

#endif