/**
 * @file random.cpp
 *
 * Declarations of global random number generators.
 */
#include <random>

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace math {

// Global random object.
MLPACK_API std::mt19937 randGen;
// Global uniform distribution.
MLPACK_API std::uniform_real_distribution<> randUniformDist(0.0, 1.0);
// Global normal distribution.
MLPACK_API std::normal_distribution<> randNormalDist(0.0, 1.0);

} // namespace math
} // namespace mlpack
