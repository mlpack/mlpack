/**
 * @file random.cpp
 *
 * Declarations of global random number generators.
 */
#include <random>
#include <mlpack_export.h>

namespace mlpack {
namespace math {

// Global random object.
MLPACK_EXPORT std::mt19937 randGen;
// Global uniform distribution.
MLPACK_EXPORT std::uniform_real_distribution<> randUniformDist(0.0, 1.0);
// Global normal distribution.
MLPACK_EXPORT std::normal_distribution<> randNormalDist(0.0, 1.0);

} // namespace math
} // namespace mlpack
