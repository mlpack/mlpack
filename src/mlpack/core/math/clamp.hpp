/**
 * @file clamp.hpp
 *
 * Miscellaneous math clamping routines.
 */
#ifndef __MLPACK_CORE_MATH_CLAMP_HPP
#define __MLPACK_CORE_MATH_CLAMP_HPP

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
inline double ClampNonNegative(double d)
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
inline double ClampNonPositive(double d)
{
  return (d - fabs(d)) / 2;
}

/**
 * Clamp a number between a particular range.
 *
 * @param value The number to clamp.
 * @param range_min The first of the range.
 * @param range_max The last of the range.
 * @return max(range_min, min(range_max, d)).
 */
inline double ClampRange(double value, double range_min, double range_max)
{
  value -= range_max;
  value = ClampNonPositive (value) + range_max;
  value -= range_min;
  value = ClampNonNegative (value) + range_min;
  return value;
}

}; // namespace math
}; // namespace mlpack

#endif // __MLPACK_CORE_MATH_CLAMP_HPP
