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
  // Global normal distribution.
  extern boost::random::normal_distribution<> randNormalDist;
#else
  // Global random object.
  extern boost::mt19937 randGen;
  // Global uniform distribution.
  extern boost::uniform_01<> randUniformDist;
  // Global normal distribution.
  extern boost::normal_distribution<> randNormalDist;
#endif

/**
 * Set the random seed used by the random functions (Random() and RandInt()).
 * The seed is casted to a 32-bit integer before being given to the random
 * number generator, but a size_t is taken as a parameter for API consistency.
 *
 * @param seed Seed for the random number generator.
 */
inline void RandomSeed(const size_t seed)
{
  randGen.seed((uint32_t) seed);
  srand((unsigned int) seed);
}

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
inline double Random(const double lo, const double hi)
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
inline int RandInt(const int hiExclusive)
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
inline int RandInt(const int lo, const int hiExclusive)
{
  #if BOOST_VERSION >= 104700
    boost::random::uniform_int_distribution<> dist(lo, hiExclusive - 1);
  #else
    boost::uniform_int<> dist(lo, hiExclusive - 1);
  #endif

  return dist(randGen);
}

/**
 * Generates a normally distributed random number with mean 0 and variance 1.
 */
inline double RandNormal()
{
  return randNormalDist(randGen);
}

/**
 * Generates a normally distributed random number with specified mean and
 * variance.
 *
 * @param mean Mean of distribution.
 * @param variance Variance of distribution.
 */
inline double RandNormal(const double mean, const double variance)
{
  return variance * randNormalDist(randGen) + mean;
}

}; // namespace math
}; // namespace mlpack

#endif // __MLPACK_CORE_MATH_MATH_LIB_HPP
