/**
 * @file math_misc.hpp
 *
 * Miscellaneous math routines.
 */

#ifndef __MLPACK_CORE_MATH_MATH_LIB_HPP
#define __MLPACK_CORE_MATH_MATH_LIB_HPP

#include "../io/cli.hpp"
#include "../io/log.hpp"

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

/**
 * Generates a uniform random number between 0 and 1.
 */
inline double Random()
{
  return rand() * (1.0 / RAND_MAX);
}

/**
 * Generates a uniform random number in the specified range.
 */
inline double Random(double lo, double hi)
{
  return Random() * (hi - lo) + lo;
}

/**
 * Generates a uniform random integer.
 */
inline int RandInt(int hi_exclusive)
{
  return rand() % hi_exclusive;
}

/**
 * Generates a uniform random integer.
 */
inline int RandInt(int lo, int hi_exclusive)
{
  return (rand() % (hi_exclusive - lo)) + lo;
}

}; // namespace math
}; // namespace mlpack

#endif // __MLPACK_CORE_MATH_MATH_LIB_HPP
