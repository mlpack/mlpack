/**
 * @file core/math/digamma.hpp
 * @author Gopi Tatiraju 
 *
 * Some parts of the implementation are inspired from boost.
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_MATH_DIGAMMA_HPP
#define MLPACK_CORE_MATH_DIGAMMA_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace math {

/**
 * This fucntion calculates and returns digamma(x).
 *
 * We have divided the implementation into two cases
 * 1. x > 0
 * digamma(x) = ln(x) - (1 / (2x))
 *
 * 2. x < 0
 * For this case we will be using reflection formula
 * digamma(1 - x) = digamma(x) + pi/tan(pi*x)
 *
 * @param x Input for which we have to calculate digamma. 
 */
template<typename T>
T digamma(T x)
{
  T result = 0;

  // Check for negative arguments and use reflection
  if (x <= -1)
  {
    // Reflect
    x = 1 - x;
    // Argument reduction for tan
    T reminder = x - floor(x);
    // Shift to negative if x > 0.5
    if (reminder > 0.5)
      reminder -= 1;
    
    // Check for evaluation at negative pole
    if (reminder == 0)
      throw std::runtime_error("Evaluation of function at pole");

    result = M_PI / tan(M_PI * reminder); 
  }

  if (x == 0)
    throw std::runtime_error("Evaluation of fucntion at pole");

  // Direct formula to calculate digamma
  result += std::log(x) - (1 / (2 * x));

  return result;
}

} // namespace math
} // namespace mlpack

#endif
