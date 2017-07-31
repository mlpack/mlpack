/**
 * @file random.hpp
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
#include <mlpack/mlpack_export.hpp>
#include <random>

namespace mlpack {
namespace math /** Miscellaneous math routines. */ {

/**
 * MLPACK_EXPORT is required for global variables; it exports the symbols
 * correctly on Windows.
 */

// Global random object.
extern MLPACK_EXPORT std::mt19937 randGen;
// Global uniform distribution.
extern MLPACK_EXPORT std::uniform_real_distribution<> randUniformDist;
// Global normal distribution.
extern MLPACK_EXPORT std::normal_distribution<> randNormalDist;

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
  arma::arma_rng::set_seed(seed);
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
  return lo + (hi - lo) * randUniformDist(randGen);
}

/**
 * Generates a uniform random integer.
 */
inline int RandInt(const int hiExclusive)
{
  return (int) std::floor((double) hiExclusive * randUniformDist(randGen));
}

/**
 * Generates a uniform random integer.
 */
inline int RandInt(const int lo, const int hiExclusive)
{
  return lo + (int) std::floor((double) (hiExclusive - lo)
                               * randUniformDist(randGen));
}

/**
 * Generates a random integer from range [0, N) with given probabilities
 * P(0), P(1), ..., P(N-1) (arbitrary discrete distribution).
 * 
 * @param weights Probabilities of individual numbers. Note that the function
 *                accepts unnormalized probabilties as long as they are
 *                non-negative and sum to a positive number.
 * @return A random integer sampled from the specified distribution.
 */
inline int RandInt(const arma::vec& weights)
{
  // Check that constraints on weights parameter are satisfied.
  if (arma::min(weights) < 0) {
    std::ostringstream oss;
      oss << "Probabilities should be non-negative."
          << std::endl;
      throw std::invalid_argument(oss.str());
  }
  double sum = arma::accu(weights);
  if (sum <= 0) {
    std::ostringstream oss;
    oss << "Sum of probabilites should be positive."
        << std::endl;
    throw std::invalid_argument(oss.str());
  }
  // Build normalized cumulative probabilities from event probabilities.
  arma::vec slots = arma::cumsum(weights) / sum;
  return std::lower_bound(slots.begin(), slots.end(), Random()) - slots.begin();
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

    for (size_t i = 0; i < maxNumSamples; i++)
      samples [ (size_t) math::RandInt(samplesRangeSize) ]++;

    distinctSamples = arma::find(samples > 0);

    if (loInclusive > 0)
      distinctSamples += loInclusive;
  }
  else
  {
    distinctSamples.set_size(samplesRangeSize);
    for (size_t i = 0; i < samplesRangeSize; i++)
      distinctSamples[i] = loInclusive + i;
  }
}

} // namespace math
} // namespace mlpack

#endif // MLPACK_CORE_MATH_MATH_LIB_HPP
