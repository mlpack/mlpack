/**
 * @file random.hpp
 *
 * Miscellaneous math random-related routines.
 *
 * This file is part of MLPACK 1.0.9.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_CORE_MATH_RANDOM_HPP
#define __MLPACK_CORE_MATH_RANDOM_HPP

#include <mlpack/prereqs.hpp>
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

  #if BOOST_VERSION >= 103900
    // Global uniform distribution.
    extern boost::uniform_01<> randUniformDist;
  #else
    // Pre-1.39 Boost.Random did not give default template parameter values.
    extern boost::uniform_01<boost::mt19937, double> randUniformDist;
  #endif

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
#if BOOST_VERSION >= 103900
  return randUniformDist(randGen);
#else
  // Before Boost 1.39, we did not give the random object when we wanted a
  // random number; that gets given at construction time.
  return randUniformDist();
#endif
}

/**
 * Generates a uniform random number in the specified range.
 */
inline double Random(const double lo, const double hi)
{
#if BOOST_VERSION >= 103900
  return lo + (hi - lo) * randUniformDist(randGen);
#else
  // Before Boost 1.39, we did not give the random object when we wanted a
  // random number; that gets given at construction time.
  return lo + (hi - lo) * randUniformDist();
#endif
}

/**
 * Generates a uniform random integer.
 */
inline int RandInt(const int hiExclusive)
{
#if BOOST_VERSION >= 103900
  return (int) std::floor((double) hiExclusive * randUniformDist(randGen));
#else
  // Before Boost 1.39, we did not give the random object when we wanted a
  // random number; that gets given at construction time.
  return (int) std::floor((double) hiExclusive * randUniformDist());
#endif
}

/**
 * Generates a uniform random integer.
 */
inline int RandInt(const int lo, const int hiExclusive)
{
#if BOOST_VERSION >= 103900
  return lo + (int) std::floor((double) (hiExclusive - lo)
                               * randUniformDist(randGen));
#else
  // Before Boost 1.39, we did not give the random object when we wanted a
  // random number; that gets given at construction time.
  return lo + (int) std::floor((double) (hiExclusive - lo)
                               * randUniformDist());
#endif

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
