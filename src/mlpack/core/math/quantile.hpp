/**
 * @file core/math/quantile.hpp
 * @author Shubham Agrawal
 *
 * Miscellaneous math quantile-related routines.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_MATH_QUANTILE_HPP
#define MLPACK_CORE_MATH_QUANTILE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace math /** Miscellaneous math routines. */ {

/**
 * Computes the inverse erf function using the rational approximation from
 * Numerical Recipes.
 *
 * @param x Input value.
 */
double erfinverse(double x)
{
  const double a[] = {0.886226899, -1.645349621, 0.914624893, -0.140543331};
  const double b[] = {1.0, -2.118377725, 1.442710462, -0.329097515, 0.012229801};
  const double c[] = {-1.970840454, -1.62490649, 3.429567803, 1.641345311};
  const double d[] = {1.0, 3.543889200, 1.637067800};
  double x2, r, y;
  int  sign_x; 
  if (x < -1 || x > 1)
    return NAN;
  
  if (x == 0)
    return 0;

  if (x > 0) 
  {
    sign_x = 1;
  }
  else 
  {
    sign_x = -1;
    x = -x;
  }

  if (x <= 0.7) 
  {
    x2 = x * x;
    r = x * (((a[3] * x2 + a[2]) * x2 + a[1]) * x2 + a[0]);
    r /= (((b[4] * x2 + b[3]) * x2 + b[2]) * x2 + b[1]) * x2 + b[0];
  } 
  else 
  {
    y = std::sqrt (-std::log ((1 - x) / 2));
    r = (((c[3] * y + c[2]) * y + c[1]) * y + c[0]);
    r /= ((d[2] * y + d[1]) * y + d[0]);
  }

  r = r * sign_x;
  x = x * sign_x;

  r -= (std::erf (r) - x) / (2 / std::sqrt (M_PI) * std::exp (-r * r));
  r -= (std::erf (r) - x) / (2 / std::sqrt (M_PI) * std::exp (-r * r));

  return r;
}

/**
 * Computes the quantile function of Guassian distribution at given probability.
 *
 * @param p Probability value.
 * @param mu Mean of the distribution. (Default 0)
 * @param sigma Standard deviation of the distribution. (Default 1)
 */
double quantile(double p, double mu = 0.0, double sigma = 1.0)
{
  return mu + sigma * std::sqrt(2.0) * erfinverse(2 * p - 1);
}

} // namespace math
} // namespace mlpack

#endif
