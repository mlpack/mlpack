/**
 * @file clamp.hpp
 *
 * Miscellaneous math clamping routines.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_MATH_CLAMP_HPP
#define MLPACK_CORE_MATH_CLAMP_HPP

#include <stdlib.h>
#include <math.h>
#include <float.h>

namespace mlpack {
namespace math /** Miscellaneous math routines. */ {

/**
 * Forces a number to be non-negative, turning negative numbers into zero.
 * Avoids branching costs (this is a measurable improvement).
 *
 * @param d Double to clamp.
 * @return 0 if d < 0, d otherwise.
 */
inline double ClampNonNegative(const double d)
{
  return (d + fabs(d)) / 2;
}

/**
 * Forces a number to be non-positive, turning positive numbers into zero.
 * Avoids branching costs (this is a measurable improvement).
 *
 * @param d Double to clamp.
 * @param 0 if d > 0, d otherwise.
 */
inline double ClampNonPositive(const double d)
{
  return (d - fabs(d)) / 2;
}

/**
 * Clamp a number between a particular range.
 *
 * @param value The number to clamp.
 * @param rangeMin The first of the range.
 * @param rangeMax The last of the range.
 * @return max(rangeMin, min(rangeMax, d)).
 */
inline double ClampRange(double value,
                         const double rangeMin,
                         const double rangeMax)
{
  value -= rangeMax;
  value = ClampNonPositive(value) + rangeMax;
  value -= rangeMin;
  value = ClampNonNegative(value) + rangeMin;
  return value;
}

} // namespace math
} // namespace mlpack

#endif // MLPACK_CORE_MATH_CLAMP_HPP
