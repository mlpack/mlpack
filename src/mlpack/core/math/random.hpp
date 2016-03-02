/**
 * @file random.hpp
 *
 * Miscellaneous math random-related routines.
 */
#ifndef MLPACK_CORE_MATH_RANDOM_HPP
#define MLPACK_CORE_MATH_RANDOM_HPP

#include <mlpack/prereqs.hpp>
#include <random>

namespace mlpack {
namespace math /** Miscellaneous math routines. */ {

// Global random object.
extern std::mt19937 randGen;
// Global uniform distribution.
extern std::uniform_real_distribution<> randUniformDist;
// Global normal distribution.
extern std::normal_distribution<> randNormalDist;

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
#if ARMA_VERSION_MAJOR > 3 || \
    (ARMA_VERSION_MAJOR == 3 && ARMA_VERSION_MINOR >= 930)
  // Armadillo >= 3.930 has its own random number generator internally that we
  // need to set the seed for also.
  arma::arma_rng::set_seed(seed);
#endif
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

} // namespace math
} // namespace mlpack

#endif // __MLPACK_CORE_MATH_MATH_LIB_HPP
