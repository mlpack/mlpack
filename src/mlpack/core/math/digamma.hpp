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

/**
 * This function evaluates the polynomial based on the
 * constants passed to the function.
 * This function evaluates the polynomial when x > 10.
 *
 * @param a Array of constants.
 * @param x Input for which digamma will be calculated.
 */
template<std::size_t N, typename T>
std::enable_if_t<N == 8, T>
EvaluatePolyLarge(const T(&a)[N], const T& x)
{
  T x2 = x * x;
  T t[2];

  t[0] = a[7] * x2 + a[5];
  t[0] *= x2;
  t[0] += a[3];
  t[0] *= x2;
  t[0] += a[1];
  t[0] *= x;

  t[1] = a[6] * x2 + a[4];
  t[1] *= x2;
  t[1] += a[2];
  t[1] *= x2;
  t[1] += a[0];

  return t[0] + t[1];
}

/**
 * This function evaluates the polynomial based on the
 * constants passed to the function.
 * This gets executed when size of the array of constants is 7.
 * This function evaluates the polynomial
 * when x is in the interval [1, 2].
 *
 * @param a Array of constants.
 * @param x Input for which digamma will be calculated.
 */
template<std::size_t N, typename T>
std::enable_if_t<N == 7, T>
EvaluatePoly12(const T(&a)[N], const T& x)
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

/**
 * This function evaluates the polynomial based on the
 * constants passed to the function.
 * This gets executed when size of the array of constants is 6.
 * This function is to evaluate the polynomial when x
 * is in the interval [1, 2].
 *
 * @param a Array of constants.
 * @param x Input for which digamma will be calculated.
 */
template<std::size_t N, typename T>
std::enable_if_t<N == 6, T>
EvaluatePoly12(const T(&a)[N], const T& x)
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
 * @param x Input for which digamma will be calculated.
 */
template<typename T>
T Digamma12(T x)
{
  T y = 0.99558162689208984F;

  const T root1 = T(1569415565) / 1073741824uL;
  const T root2 = (T(381566830) / 1073741824uL) / 1073741824uL;
  const T root3 = 0.9016312093258695918615325266959189453125e-19;

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

  T g = x - root1;
  g -= root2;
  g -= root3;

  T r = EvaluatePoly12(P, T(x - 1)) / EvaluatePoly12(Q, T(x - 1));

  T result = (g * y) + (g * r);

  return result;
}

/**
 * This function calculates and returns Digamma(x)
 * when x > 10.
 *
 * @param x Input for which digamma will be calculated.
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
  const T z = 1 / (x * x);
  result -= z * EvaluatePolyLarge(P, z);

  return result;
}

/**
 * This function calculates and returns digamma(x).
 * We have divided the implementation as
 * 1. x > 10.
 * 2. x in the interval [1, 2].
 * For x < 0 we use reflection.
 * For x > 2 we reduce it to the interval [1, 2].
 * For x < 1 we bring it to the interval [1, 2].
 *
 * @param x Input for which digamma will be calculated.
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

    if (remainder > 0.5)
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

    // If x < 1, shift to x > 1.
    while (x < 1)
    {
      result -= 1 / x;
      x += 1;
    }

    result += Digamma12(x);
  }

  return result;
}

} // namespace mlpack

#endif
