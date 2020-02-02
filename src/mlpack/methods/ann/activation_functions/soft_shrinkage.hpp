/**
 * @file soft_shrinkage_function.hpp
 * @author Lakshya Ojha
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_SOFT_SHRINKAGE_FUNCTION_HPP
#define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_SOFT_SHRINKAGE_FUNCTION_HPP

#include <mlpack/prereqs.hpp>
#include <algorithm>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 *SoftShrinkage operator is defined as
 *         | x - lambda, if x >  lambda
 *  f(x) = | x + lambda, if x < -lambda
 *         | 0, otherwise
 *
 * lambda is set to 0.5 by default
 *
 *          | 1 , if x >  lambda
 *  f'(x) = | 1 , if x < -lambda
 *          | 0 , otherwise
 */
class SoftShrinkage
{
  public:
  static double Fn(const double x)
  {
    if(x > lambda)
    {
      return (x - lambda);
    }
    
    else if(x < -lambda)
    {
      return (x + lambda);
    }

    return 0;
  }
  template <typename InputVecType,typename OutputVecType>
  static void Fn(const InputVecType &x, OutputVecType &y)
  {
    y.set_size(x.size());
    for(size_t i=0;i<x.size();i++)
    {
      y(i)=Fn(x(i));  
    }
  }
}
