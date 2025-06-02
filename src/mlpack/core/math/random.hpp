/**
 * @file core/math/random.hpp
 *
 * Miscellaneous math random-related routines.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_MATH_RANDOM_HPP
#define MLPACK_CORE_MATH_RANDOM_HPP

#include <mlpack/prereqs.hpp>
#include <random>

namespace mlpack {

// Because we have multiple RNGs for mlpack (one for each thread, as they are
// marked thread_local), we must ensure that the default seeds and user-set
// seeds do not cause each thread's RNG to have the exact same output.
// Therefore, we assign a different offset for each thread, and add this offset
// whenever we set a seed with RandGen() or RandomSeed().
inline size_t RandGenSeedOffset()
{
  // The use of `seedCounter` ensures that each individual RNG gets its own
  // separate seed.  Otherwise, in OpenMP-enabled loops that use random numbers,
  // it is possible that each individual thread could generate the same sequence
  // of random numbers.
  static std::atomic<size_t> seedCounter(0);
  static thread_local size_t threadSeed = (seedCounter++);
  return threadSeed;
}

//! Global random object.
inline std::mt19937& RandGen()
{
  static thread_local std::mt19937 randGen(std::mt19937::default_seed +
      RandGenSeedOffset());
  return randGen;
}

//! Global uniform distribution.
inline std::uniform_real_distribution<>& RandUniformDist()
{
  static thread_local std::uniform_real_distribution<> randUniformDist(0.0,
      1.0);
  return randUniformDist;
}

//! Global normal distribution.
inline std::normal_distribution<>& RandNormalDist()
{
  static thread_local std::normal_distribution<> randNormalDist(0.0, 1.0);
  return randNormalDist;
}

/**
 * Set the random seed used by the random functions (Random() and RandInt()).
 * The seed is casted to a 32-bit integer before being given to the random
 * number generator, but a size_t is taken as a parameter for API consistency.
 *
 * @param seed Seed for the random number generator.
 */
inline void RandomSeed(const size_t seed)
{
  #if (!defined(BINDING_TYPE) || BINDING_TYPE != BINDING_TYPE_TEST)
    RandGen().seed((uint32_t) (seed + RandGenSeedOffset()));
    #if (BINDING_TYPE == BINDING_TYPE_R)
      // To suppress Found 'srand', possibly from 'srand' (C).
      (void) seed;
    #else
      srand((unsigned int) seed);
    #endif
    arma::arma_rng::set_seed(seed + RandGenSeedOffset());
  #else
    (void) seed;
  #endif
}

/**
 * Set the random seed to a fixed number.
 * This function is used in binding tests to set a fixed random seed before
 * calling the binding. In this way we can test whether a certain parameter
 * makes a difference to execution of CLI binding.
 * Refer to pull request #1306 for discussion on this function.
 */
#if (BINDING_TYPE == BINDING_TYPE_TEST)
inline void FixedRandomSeed()
{
  std::mt19937 rng;
  std::uniform_int_distribution<size_t> dist;
  const size_t seed = dist(rng);
  RandGen().seed((uint32_t) seed + RandGenSeedOffset());
  srand((unsigned int) seed);
  arma::arma_rng::set_seed(seed + RandGenSeedOffset());
}

inline void CustomRandomSeed(const size_t seed)
{
  RandGen().seed((uint32_t) seed + RandGenSeedOffset());
  srand((unsigned int) seed);
  arma::arma_rng::set_seed(seed + RandGenSeedOffset());
}
#endif

/**
 * Generates a uniform random number between 0 and 1.
 */
inline double Random()
{
  return RandUniformDist()(RandGen());
}

/**
 * Generates a uniform random number in the specified range.
 */
inline double Random(const double lo, const double hi)
{
  return lo + (hi - lo) * RandUniformDist()(RandGen());
}

/**
 * Generates a 0/1 specified by the input.
 */
inline double RandBernoulli(const double input)
{
  if (Random() < input)
    return 1;
  else
    return 0;
}

/**
 * Generates a uniform random integer.
 */
inline int RandInt(const int hiExclusive)
{
  return (int) std::floor((double) hiExclusive * RandUniformDist()(RandGen()));
}

/**
 * Generates a uniform random integer.
 */
inline int RandInt(const int lo, const int hiExclusive)
{
  return lo + (int) std::floor((double) (hiExclusive - lo)
                               * RandUniformDist()(RandGen()));
}

/**
 * Generates a normally distributed random number with mean 0 and standard
 * deviation of 1.
 */
inline double RandNormal()
{
  return RandNormalDist()(RandGen());
}

/**
 * Generates a normally distributed random number with specified mean and
 * standard deviation.
 *
 * @param mean Mean of distribution.
 * @param stddev Standard deviation of distribution.
 */
inline double RandNormal(const double mean, const double stddev)
{
  return stddev * RandNormalDist()(RandGen()) + mean;
}

} // namespace mlpack

#endif // MLPACK_CORE_MATH_RANDOM_HPP
