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

template<typename T>
T EvaluatePolyLarge(const T* a, const T& x)
{
  return 0;
}

template<std::size_t N, typename T>
typename std::enable_if<N == 7, T>::type
EvaluatePoly_1_2(const T(&a)[N], const T& x)
{
  T x2 = x * x;
  T t[2];

  t[0] = (a[6] * x2) + a[4];
  t[0] *= x2;
  t[0] += a[2];
  t[0] *= x2;
  t[0] += a[0];

  t[1] = (a[5] * x2) + a[3];
  t[1] *= x2;
  t[1] += a[1];
  t[1] *= x;

  return t[0] + t[1];
}

template<std::size_t N, typename T>
typename std::enable_if<N == 6, T>::type
EvaluatePoly_1_2(const T(&a)[N], const T& x)
{
  T x2 = x * x;
  T t[2];

  t[0] = (a[5] * x2) + a[3];
  t[0] *= x2;
  t[0] += a[1];
  t[0] *= x;

  t[1] = (a[4] * x2) + a[2];
  t[1] *= x2;
  t[1] += a[0];

  return t[0] + t[1];
}

/**
 * This function calculates and returns Digamma(x)
 * when x is in the interval [1, 2].
 *
 * @param x Input for which we have to calculate digamma.
 */
template<typename T>
T Digamma_1_2(T x)
{
  T y = 0.99558162689208984F;

  const T root_1 = T(1569415565) / 1073741824uL;
  const T root_2 = (T(381566830) / 1073741824uL) / 1073741824uL;
  const T root_3 = 0.9016312093258695918615325266959189453125e-19;

  const T P[] = {
                 0.25479851061131551,
                 -0.32555031186804491,
                 -0.65031853770896507,
                 -0.28919126444774784,
                 -0.045251321448739056,
                 -0.0020713321167745952
                };

  const T Q[] = {
                 1.0,
                 2.0767117023730469,
                 1.4606242909763515,
                 0.43593529692665969,
                 0.054151797245674225,
                 0.0021284987017821144,
                 -0.55789841321675513e-6
                };

  T g = x - root_1;
  g -= root_2;
  g -= root_3;

  T r = EvaluatePoly_1_2(P, T(x-1)) / EvaluatePoly_1_2(Q, T(x-1));

  T result = (g * y) + (g * r);

  return result;
}

/**
 * This function calculates and returns Digamma(x)
 * when x > 10.
 *
 * @param x Input for which we have to calculate digamma.
 */
template<typename T>
T DigammaLarge(T x)
{
  const T P[] = {
                 0.083333333333333333333333333333333333333333333333333,
                 -0.0083333333333333333333333333333333333333333333333333,
                 0.003968253968253968253968253968253968253968253968254,
                 -0.0041666666666666666666666666666666666666666666666667,
                 0.0075757575757575757575757575757575757575757575757576,
                 -0.021092796092796092796092796092796092796092796092796,
                 0.083333333333333333333333333333333333333333333333333,
                 -0.44325980392156862745098039215686274509803921568627
                };

  x -= 1;
  T result = std::log(x);
  result += 1 / (2 * x);
  T z = 1 / (x * x);
  result -= z * EvaluatePolyLarge(P, z); 

  return result;
}

/**
 * This function calculates and returns digamma(x).
 *
 * We have divided the implementation into two cases
 * 1. x > 10.
 *
 * 2. x in the interval [1, 2].
 *
 * For x < 0 we use reflection.
 *
 * For x > 2 we reduce itto the interval [1, 2].
 *
 * For x < 1 we bring it to the interval [1, 2].
 *
 * @param x Input for which we have to calculate digamma. 
 */
template<typename T>
T Digamma(T x)
{
  T result = 0;

  // Use reflection for negative x.
  if (x < 0)
  {
    // Use reflection.
    x = 1 - x;
    // Check for evaluation of the function on poles.
    T remainder = x - floor(x);

    if(remainder > 0.5)
    {
      remainder -= 1;
    }

    if (remainder == 0)
      throw std::runtime_error("Evaluation of the function at pole.");

    result = M_PI / tan(M_PI * remainder);
  }

  // Digamma is not defined at 0.
  if (x == 0)
    throw std::runtime_error("Evaluation of the function at pole.");

  if (x >= 10)
  {
    result += DigammaLarge(x);
  }
  else
  {
    // If x > 2, reduce to the interval [1, 2].
    while (x > 2)
    {
      x -= 1;
      result += 1 / x;
    }

    // If x < 1, shift to x > 1
    while (x < 1)
    {
      result -= 1 / x;
      x += 1;
    }

    result += Digamma_1_2(x);
  }

  return result;
}


} // namespace math
} // namespace mlpack

#endif
