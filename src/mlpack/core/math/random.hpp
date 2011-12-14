/**
 * @file random.hpp
 *
 * Miscellaneous math random-related routines.
 */
#ifndef __MLPACK_CORE_MATH_RANDOM_HPP
#define __MLPACK_CORE_MATH_RANDOM_HPP

#include <stdlib.h>
#include <math.h>
#include <float.h>

namespace mlpack {
namespace math /** Miscellaneous math routines. */ {

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
