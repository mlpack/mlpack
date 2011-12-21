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

#include <boost/random.hpp>

namespace mlpack {
namespace math /** Miscellaneous math routines. */ {

// Annoying Boost versioning issues.
#include <boost/version.hpp>

#if BOOST_VERSION >= 104700
  // Global random object.
  extern boost::random::mt19937 randGen;
  // Global uniform distribution.
  extern boost::random::uniform_01<> randUniformDist;
#else
  // Global random object.
  extern boost::mt19937 randGen;
  // Global uniform distribution.
  extern boost::uniform_01<> randUniformDist;
#endif

/**
 * Generates a uniform random number between 0 and 1.
 */
inline double Random()
{
  return randUniformDist(randGen);
}

/**
 * Generates a uniform random number in the specified range.
 */
inline double Random(double lo, double hi)
{
  #if BOOST_VERSION >= 104700
    boost::random::uniform_real_distribution<> dist(lo, hi);
  #else
    boost::uniform_real<> dist(lo, hi);
  #endif

  return dist(randGen);
}

/**
 * Generates a uniform random integer.
 */
inline int RandInt(int hiExclusive)
{
  #if BOOST_VERSION >= 104700
    boost::random::uniform_int_distribution<> dist(0, hiExclusive - 1);
  #else
    boost::uniform_int<> dist(0, hiExclusive - 1);
  #endif

  return dist(randGen);
}

/**
 * Generates a uniform random integer.
 */
inline int RandInt(int lo, int hiExclusive)
{
  #if BOOST_VERSION >= 104700
    boost::random::uniform_int_distribution<> dist(lo, hiExclusive - 1);
  #else
    boost::uniform_int<> dist(0, hiExclusive - 1);
  #endif

  return dist(randGen);
}

}; // namespace math
}; // namespace mlpack

#endif // __MLPACK_CORE_MATH_MATH_LIB_HPP
