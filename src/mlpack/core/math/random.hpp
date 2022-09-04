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

/**
 * MLPACK_EXPORT is required for global variables; it exports the symbols
 * correctly on Windows.
 */

//! Global random object.
inline std::mt19937& RandGen()
{
  static thread_local std::mt19937 randGen;
  return randGen;
}

//! Global uniform distribution.
inline std::uniform_real_distribution<>& RandUniformDist()
{
  static thread_local std::uniform_real_distribution<> randUniformDist(0.0, 1.0);
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
    RandGen().seed((uint32_t) seed);
    #if (BINDING_TYPE == BINDING_TYPE_R)
      // To suppress Found 'srand', possibly from 'srand' (C).
      (void) seed;
    #else
      srand((unsigned int) seed);
    #endif
    arma::arma_rng::set_seed(seed);
  #else
    (void) seed;
  #endif
}

/**
 * Set the random seed to a fixed number.
 * This function is used in binding tests to set a fixed random seed before
 * calling mlpack(). In this way we can test whether a certain parameter makes
 * a difference to execution of CLI binding.
 * Refer to pull request #1306 for discussion on this function.
 */
#if (BINDING_TYPE == BINDING_TYPE_TEST)
inline void FixedRandomSeed()
{
  const static size_t seed = rand();
  RandGen().seed((uint32_t) seed);
  srand((unsigned int) seed);
  arma::arma_rng::set_seed(seed);
}

inline void CustomRandomSeed(const size_t seed)
{
  RandGen().seed((uint32_t) seed);
  srand((unsigned int) seed);
  arma::arma_rng::set_seed(seed);
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
 * Generates a normally distributed random number with mean 0 and variance 1.
 */
inline double RandNormal()
{
  return RandNormalDist()(RandGen());
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
  return variance * RandNormalDist()(RandGen()) + mean;
}

/**
 * Obtains no more than maxNumSamples distinct samples. Each sample belongs to
 * [loInclusive, hiExclusive).
 *
 * @param loInclusive The lower bound (inclusive).
 * @param hiExclusive The high bound (exclusive).
 * @param maxNumSamples The maximum number of samples to obtain.
 * @param distinctSamples The samples that will be obtained.
 */
inline void ObtainDistinctSamples(const size_t loInclusive,
                                  const size_t hiExclusive,
                                  const size_t maxNumSamples,
                                  arma::uvec& distinctSamples)
{
  const size_t samplesRangeSize = hiExclusive - loInclusive;

  if (samplesRangeSize > maxNumSamples)
  {
    arma::Col<size_t> samples;

    samples.zeros(samplesRangeSize);

    for (size_t i = 0; i < maxNumSamples; ++i)
      samples [ (size_t) RandInt(samplesRangeSize) ]++;

    distinctSamples = arma::find(samples > 0);

    if (loInclusive > 0)
      distinctSamples += loInclusive;
  }
  else
  {
    distinctSamples.set_size(samplesRangeSize);
    for (size_t i = 0; i < samplesRangeSize; ++i)
      distinctSamples[i] = loInclusive + i;
  }
}

} // namespace mlpack

#endif // MLPACK_CORE_MATH_RANDOM_HPP
