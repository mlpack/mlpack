/**
 * @file core/math/trigamma.hpp
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
#ifndef MLPACK_CORE_MATH_TRIGAMMA_HPP
#define MLPACK_CORE_MATH_TRIGAMMA_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * This function evaluates the polynomial based on the
 * constants passed to the functions.
 * This function gets executed when the size of the array
 * of constants is 6.
 * This function is to evaluate the polynomial
 * when x >= 1.
 *
 * @param a Array of constants.
 * @param x Input for which we have to calculate trigamma.
 */
template<std::size_t N, typename T>
std::enable_if_t<N == 6, T>
EvaluatePolyPrec(const T(&a)[N], const T& x)
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
 * This function evaluates the polynomial based on the
 * constants passed to the functions.
 * This function gets executed when the size of the array
 * of constants is 7.
 * This function is to evaluate the polynomial
 * when x >= 1.
 *
 * @param a Array of constants.
 * @param x Input for which we have to calculate trigamma.
 */
template<std::size_t N, typename T>
std::enable_if_t<N == 7, T>
EvaluatePolyPrec(const T(&a)[N], const T& x)
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
 * This function calculates and returns Trigamma(x)
 * when x > 1.
 * Implementation is divided into 3 parts
 * 1. 1 < x <= 2
 * 2. 2 < x <= 4
 * 3. x > 4
 *
 * @param x Input for which we have to calculate trigamma.
 */
template<typename T>
T TrigammaPrec(T x)
{
  T offset = 2.1093254089355469;

  const T P12[] = {
                      -1.1093280605946045,
                      -3.8310674472619321,
                      -3.3703848401898283,
                      0.28080574467981213,
                      1.6638069578676164,
                      0.64468386819102836
                  };

  const T Q12[] = {
                      1.0,
                      3.4535389668541151,
                      4.5208926987851437,
                      2.7012734178351534,
                      0.64468798399785611,
                      -0.20314516859987728e-6
                  };

  const T P24[] = {
                        -0.13803835004508849e-7,
                        0.50000049158540261,
                        1.6077979838469348,
                        2.5645435828098254,
                        2.0534873203680393,
                        0.74566981111565923
                  };

  const T Q24[] = {
                      1.0,
                      2.8822787662376169,
                      4.1681660554090917,
                      2.7853527819234466,
                      0.74967671848044792,
                      -0.00057069112416246805
                  };

  const T P4INF[] = {
                        0.68947581948701249e-17L,
                        0.49999999999998975L,
                        1.0177274392923795L,
                        2.498208511343429L,
                        2.1921221359427595L,
                        1.5897035272532764L,
                        0.40154388356961734L
                    };

  const T Q4INF[] = {
                        1.0L,
                        1.7021215452463932L,
                        4.4290431747556469L,
                        2.9745631894384922L,
                        2.3013614809773616L,
                        0.28360399799075752L,
                        0.022892987908906897L
                    };

  if (x <= 2) // For 1 < x <= 2.
  {
    return (offset + EvaluatePolyPrec(P12, x) / EvaluatePolyPrec(Q12, x)) /
        (x * x);
  }
  else if (x <= 4) // For 2 < x <= 4.
  {
    T y = 1 / x;
    return (1 + EvaluatePolyPrec(P24, y) / EvaluatePolyPrec(Q24, y)) / x;
  }

  // For x > 4.
  T y = 1 / x;
  return (1 + EvaluatePolyPrec(P4INF, y) / EvaluatePolyPrec(Q4INF, y)) / x;
}

/**
 * This function calculates and returns Trigamma(x).
 * For x < 0 we use reflection.
 * For 0 < x < 1 bring x to interval [1, INF].
 *
 * @param x Input for which we have to calculate trigamma.
 */
template<typename T>
T Trigamma(T x)
{
  T result = 0;

  if (x <= 0)
  {
    // Reflect.
    T z = 1 - x;

    // Check for evaluation on pole.
    if (floor(x) == x)
      throw std::runtime_error("Evaluation of the function at pole.");

    T s = fabs(x) < fabs(z) ? sin(M_PI * x) : sin(M_PI * z);

    return -Trigamma(z) + std::pow(M_PI, 2) / (s * s);
  }

  if (x < 1)
  {
    // Make x > 1.
    result = 1 / (x * x);
    x += 1;
  }

  // For x >= 1.
  return result + TrigammaPrec(x);
}

} // namespace mlpack

#endif
